#!/usr/bin/env python3
"""Local Xiaomi-OCR-0 MCP service.

Inference is served by a local SGLang or vLLM process. This MCP process owns image
normalization, optional PP-DocLayoutV3 region detection/cropping, concurrent OCR,
task prompts, and result formatting. No hosted OCR or layout service is used.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing, nullcontext
from pathlib import Path

# A standalone Skill installation must point at a complete repository checkout.
REPO_ROOT = Path(os.environ.get("XIAOMI_OCR_REPO", Path(__file__).resolve().parents[2])).expanduser()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess.document import finalize_document
from postprocess.region import merge_page_to_markdown
from postprocess.otsl import convert_otsl_to_html
from runtime_monitor import CallMonitor, active_calls, recent_calls, compact_devices, device_display, gpu_placement, status_message, probe_ocr_runtime, reserve_protocol_stdout, write_rpc

OCR_BASE = os.environ.get("XIAOMI_OCR_LOCAL_URL", "http://127.0.0.1:8000/v1").strip().rstrip("/")
OCR_MODEL = os.environ.get("XIAOMI_OCR_MODEL", "SeerRay-Lab/Xiaomi-OCR-0").strip()
OCR_BACKEND = os.environ.get("XIAOMI_OCR_BACKEND", "openai").strip().lower()
MAX_TOKENS = int(os.environ.get("XIAOMI_OCR_MAX_TOKENS", "4096"))
CONCURRENCY = max(1, min(int(os.environ.get("XIAOMI_OCR_CONCURRENCY", "4")), 32))
LAYOUT_THRESHOLD = float(os.environ.get("XIAOMI_OCR_LAYOUT_THRESHOLD", "0.3"))

PROMPTS = {
    "document": (
        "Extract all information from the main body of the document image and represent it "
        "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
        "format, formulas in the document should be represented using LATEX format, and the "
        "parsing should be organized according to the reading order."
    ),
    "text": "Extract the text in the image.",
    "table": "Parse the table in the image into OTSL.",
    "formula": "Identify the formula in the image and represent it using LATEX format.",
}
LABEL_MAP = {
    "text": "text", "paragraph": "text", "plain text": "text", "paragraph_title": "title",
    "title": "title", "header": "title", "section_header": "title", "table": "table",
    "table_caption": "text", "formula": "formula", "equation": "formula",
    "isolate_formula": "formula", "figure": "figure", "image": "figure", "chart": "figure",
    "figure_caption": "text", "figure_title": "text", "chart_title": "text",
    "list": "text", "list_item": "text", "caption": "text", "abstract": "text",
    "number": "text", "reference": "text", "footnote": "text",
}
_INFERENCE_SLOTS = threading.BoundedSemaphore(CONCURRENCY)
_REQUEST_SLOTS = threading.BoundedSemaphore(2)


def _stage(monitor, component, stage, **context):
    return monitor.stage(component, stage, **context) if monitor else nullcontext()


def _format_device(kind, index):
    kind = str(kind).strip()
    return f"{kind}:{index}" if (index is not None and kind.lower() != "cpu") else kind


def layout_runtime_status(refresh=False):
    """Inspect the predictor, not Paddle's thread-global default device."""
    model = _LAYOUT_MODEL
    device, source = None, "not reported"
    if model is not None:
        predictor = getattr(model, "_predictor", model)
        runner_cfg = getattr(getattr(predictor, "runner", None), "_config", None)
        if isinstance(runner_cfg, dict) and runner_cfg.get("device_type"):
            device = _format_device(runner_cfg["device_type"], runner_cfg.get("device_id"))
            source = "PaddleX runner._config"
        if device is None:
            # Older PaddleX 3.x builds expose resolved options in pp_option/hpi_config.
            # Never fall back to predictor.device, which stores the unresolved request.
            for name in ("pp_option", "hpi_config"):
                options = getattr(predictor, name, None) or getattr(model, name, None)
                kind = options.get("device_type") if isinstance(options, dict) else getattr(options, "device_type", None)
                index = options.get("device_id") if isinstance(options, dict) else getattr(options, "device_id", None)
                if kind:
                    device = _format_device(kind, index)
                    source = "PaddleX predictor." + name
                    break
    info = {"model": "PP-DocLayoutV3", "pid": os.getpid(), "loaded": model is not None,
            "requested_device": os.environ.get("XIAOMI_OCR_LAYOUT_DEVICE") or "auto",
            "device": str(device) if device is not None else "unknown",
            "source": source,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if device is not None and str(device).lower().split(":")[0] in {"gpu", "cuda"}:
        info.update(gpu_placement({os.getpid()}, refresh=refresh))
    info["display_device"] = device_display(info)
    return info


def runtime_status(model_url=None, model=None, refresh=False, request_id=None, verbose=False):
    calls, recent = active_calls(), recent_calls()
    if request_id:
        call = next((c for c in calls if c["request_id"] == request_id), None)
        task = (call if verbose else call["summary"]) if call else next((c for c in recent if c["request_id"] == request_id), None)
        return {"scope": "this_process", "found": task is not None, "task": task}
    components = {"layout": layout_runtime_status(refresh=refresh),
                  "ocr": probe_ocr_runtime(_local_url(model_url), model or OCR_MODEL, refresh=refresh)}
    return {"scope": "this_process", "status": "busy" if len(calls) >= 2 else "working" if calls else "idle",
            "components": components if verbose else compact_devices(components),
            "active_calls": calls if verbose else [c["summary"] for c in calls],
            "recent_calls": [{k: v for k, v in c.items() if k not in {"devices", "active_pages", "last_activity_ms"}} for c in recent[:3]],
            "task_limit": 2, "ocr_concurrency_limit": CONCURRENCY}


def _local_url(base: str | None = None) -> str:
    """Refuse non-loopback inference endpoints: model requests must stay on this machine."""
    base = (base or OCR_BASE).rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("XIAOMI_OCR_LOCAL_URL must point to a local http://127.0.0.1 inference server")
    return base


def _http_json(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 400 and "maximum context length" in body.lower():
            raise RuntimeError(
                "The image/prompt plus requested output exceeds the model context. "
                "Reduce max_tokens or use smaller inputs."
            ) from e
        raise RuntimeError(f"Local inference returned HTTP {e.code}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Local model is unavailable at {_local_url()}: {e.reason}. Start SGLang or vLLM first.") from e


def _chat(image: bytes, prompt: str, max_tokens: int, *, model_url: str | None = None, model: str | None = None, temperature=0.0,
          monitor=None, **context) -> dict:
    url = _local_url(model_url) + "/chat/completions"
    image = _image_png(image)
    if OCR_BACKEND != "ollama":
        encoded = base64.b64encode(image).decode()
        payload = {"model": model or OCR_MODEL, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            {"type": "text", "text": prompt},
        ]}], "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.8, "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    # One shared limit prevents PDF page workers times region workers from
    # oversubscribing the inference server. Queue time is visible separately.
    if not _INFERENCE_SLOTS.acquire(blocking=False):
        with _stage(monitor, "ocr", "queue", **context):
            _INFERENCE_SLOTS.acquire()
    try:
        with _stage(monitor, "ocr", "inference", **context):
            if OCR_BACKEND == "ollama":
                from pipeline.ollama_backend import complete
                return complete(image, prompt, max_tokens, base_url=_local_url(model_url),
                                model=model or OCR_MODEL, temperature=temperature)
            if OCR_BACKEND not in {"openai", "vllm", "sglang"}:
                raise RuntimeError(f"Unsupported XIAOMI_OCR_BACKEND: {OCR_BACKEND}")
            data = _http_json(url, payload)
    finally:
        _INFERENCE_SLOTS.release()
    message = data["choices"][0]["message"]["content"]
    if isinstance(message, list):
        message = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in message)
    text = str(message or "").strip()
    if not text:
        raise RuntimeError("Local model returned an empty response")
    if data["choices"][0].get("finish_reason") == "length":
        raise RuntimeError(f"Model output was truncated (max_tokens={max_tokens}). Increase max_tokens within the model context limit, or use smaller inputs")
    return {"text": text, "usage": data.get("usage") or {}}


def _read_bytes(path: str | None, url: str | None, b64: str | None, kind: str) -> bytes:
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise RuntimeError(f"{kind} path not found: {path}")
        return p.read_bytes()
    if url:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    if b64:
        try:
            raw = base64.b64decode(b64.split(",", 1)[-1], validate=True)
        except Exception as exc:
            raise RuntimeError(f"Invalid or truncated {kind} base64; pass a local file path instead") from exc
        return raw
    raise RuntimeError(f"Provide {kind}_path or {kind}_base64/image_url")


def _image_png(raw: bytes) -> bytes:
    try:
        from PIL import Image
        from evaluation_pipeline.cropping import fit_image
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        fit_image(image)
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()
    except Exception as exc:
        raise RuntimeError(f"Cannot decode image: {exc}") from exc


def _layout_regions(png: bytes, monitor=None, **context) -> list[dict]:
    """Run layout detection in-process; never asks the caller to invoke a layout tool."""
    with _stage(monitor, "layout", "import", **context):
        try:
            import numpy as np
            import paddlex as pdx
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Region mode needs PaddleX/PaddlePaddle installed in this environment; install the PP-DocLayoutV3 runtime described in the Skill.") from exc
        from evaluation_pipeline.layout_detect_paddlex import serialize_regions
    img = Image.open(io.BytesIO(png)).convert("RGB")
    global _LAYOUT_MODEL
    if not _LAYOUT_LOCK.acquire(blocking=False):
        with _stage(monitor, "layout", "queue", **context):
            _LAYOUT_LOCK.acquire()
    try:
        if _LAYOUT_MODEL is None:
            options = {"threshold": LAYOUT_THRESHOLD}
            if os.environ.get("XIAOMI_OCR_LAYOUT_MODEL_DIR"):
                options["model_dir"] = os.environ["XIAOMI_OCR_LAYOUT_MODEL_DIR"]
            if os.environ.get("XIAOMI_OCR_LAYOUT_DEVICE"):
                options["device"] = os.environ["XIAOMI_OCR_LAYOUT_DEVICE"]
            with _stage(monitor, "layout", "load", **context):
                _LAYOUT_MODEL = pdx.create_model("PP-DocLayoutV3", **options)
        if monitor:
            monitor.component("layout", layout_runtime_status())
        with _stage(monitor, "layout", "inference", **context):
            result = list(_LAYOUT_MODEL.predict(np.asarray(img)))[0]
        if monitor:
            # Some versions resolve/fall back only on first prediction.
            monitor.component("layout", layout_runtime_status())
    finally:
        _LAYOUT_LOCK.release()
    # Preserve the detector's reading order; do not sort columns by y/x.
    return serialize_regions(result, *img.size)


_LAYOUT_MODEL = None
_LAYOUT_LOCK = threading.Lock()


def _crop(image, box: list[float], polygon=None) -> bytes:
    from evaluation_pipeline.cropping import crop_region
    crop = crop_region(image, box, polygon=polygon)
    out = io.BytesIO()
    crop.save(out, format="PNG")
    return out.getvalue()


def assemble_regions(regions: list[dict], texts: list[str]) -> str:
    """Use the two-stage pipeline's table conversion and full page assembly."""
    rows = []
    for region, text in zip(regions, texts):
        if region.get("task_type") == "table":
            # Match twostage_img2md.store_result: preserve the raw table if
            # conversion fails, then continue assembling the rest of the page.
            try:
                text = convert_otsl_to_html(text)
            except Exception:
                pass
        rows.append({**region, "merged": text})
    return merge_page_to_markdown(rows)


def merge_usage(a, b):
    result = dict(a)
    for key, value in b.items():
        if isinstance(value, dict):
            result[key] = merge_usage(result.get(key) if isinstance(result.get(key), dict) else {}, value)
        elif isinstance(value, (int, float)):
            previous = result.get(key)
            result[key] = (previous if isinstance(previous, (int, float)) else 0) + value
    return result


def ocr_image(image_path: str | None = None, image_url: str | None = None,
              image_base64: str | None = None, mode: str = "page", max_tokens: int = MAX_TOKENS,
              *, monitor=None, page=None, prompt=None, concurrency=None, chat_options=None) -> dict:
    if mode not in {"page", "region"}:
        raise RuntimeError("mode must be 'page' or 'region'")
    context = {"page": page} if page is not None else {}
    options = chat_options or {}
    with _stage(monitor, "pipeline", "prepare_image", **context):
        png = _image_png(_read_bytes(image_path, image_url, image_base64, "image"))
    if mode == "page":
        result = _chat(png, prompt or PROMPTS["document"], max_tokens, monitor=monitor, **options, **context)
        with _stage(monitor, "pipeline", "assemble", **context):
            return {"mode": "page", "markdown": finalize_document(result["text"]), "usage": result["usage"]}

    from PIL import Image
    image = Image.open(io.BytesIO(png)).convert("RGB")
    layout_started = time.monotonic()
    regions = _layout_regions(png, monitor=monitor, **context)
    layout_ms = int((time.monotonic() - layout_started) * 1000)
    regions = [r for r in regions if r.get("task_type") not in {"skip", "abandon"}]
    if not regions:
        if monitor:
            monitor.emit("layout", "fallback", "No usable regions; using whole-page OCR", **context)
        result = _chat(png, prompt or PROMPTS["document"], max_tokens, monitor=monitor, **options, **context)
        return {"mode": "page_fallback", "reason": "layout returned no usable regions",
                "markdown": finalize_document(result["text"]), "usage": result["usage"],
                "layout_ms": layout_ms, "region_results": []}
    results, details, usage = [""] * len(regions), [None] * len(regions), {}
    if monitor:
        monitor.emit("layout", "regions", f"Layout detected {len(regions)} usable regions",
                     total_regions=len(regions), **context)
        monitor.emit("ocr", "regions", f"Regions completed: 0/{len(regions)}",
                     completed=0, total=len(regions), **context)
    def run(i: int):
        region = regions[i]
        task_type = region.get("task_type", "text")
        started = time.monotonic()
        crop = _crop(image, region["bbox_2d"])
        result = _chat(crop, PROMPTS.get(task_type, PROMPTS["text"]), max_tokens,
                       monitor=monitor, region=i + 1, **options, **context)
        return i, result, int((time.monotonic() - started) * 1000)
    workers = min(CONCURRENCY, max(1, concurrency or CONCURRENCY), len(regions))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, future in enumerate(as_completed([pool.submit(run, i) for i in range(len(regions))]), 1):
            i, result, ms = future.result()
            results[i] = result["text"]
            usage = merge_usage(usage, result["usage"])
            details[i] = {"bbox": regions[i]["bbox_2d"], "label": regions[i]["label"],
                          "bucket": regions[i].get("task_type", "text"), "ok": True, "ms": ms,
                          "chars": len(result["text"])}
            if monitor:
                monitor.emit("ocr", "regions", f"Regions completed: {done}/{len(regions)} "
                             + (f"page={context['page']}" if context else ""),
                             completed=done, total=len(regions), **context)
    with _stage(monitor, "pipeline", "assemble", **context):
        return {"mode": "region", "regions": len(regions), "markdown": assemble_regions(regions, results),
                "region_results": details, "usage": usage, "layout_ms": layout_ms, "concurrency": workers}


_PDF_LOCK = threading.Lock()


def _render_pdf(raw: bytes, max_pages: int | None, dpi: int, monitor=None, on_render=None) -> list[bytes | None]:
    if dpi <= 0 or (max_pages is not None and max_pages < 1):
        raise ValueError("dpi and max_pages must be positive")
    if not raw.startswith(b"%PDF"):
        raise RuntimeError("Input is not a PDF")
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("PDF support needs pypdfium2; install requirements.txt") from exc
    # PDFium is not thread-safe, even for separate documents. Serialize rendering,
    # then release the lock before the much slower concurrent model requests.
    with _PDF_LOCK, pdfium.PdfDocument(raw) as doc:
        count = len(doc) if max_pages is None else min(len(doc), max_pages)
        pages = []
        for i in range(count):
            if monitor:
                monitor.check_cancelled()
            with closing(doc[i]) as page:
                bitmap = page.render(scale=dpi / 72)
                try:
                    image = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            # Only skip a uniformly light page, checked at render resolution.
            # Even one faint mark keeps the page on the normal OCR path.
            blank = all(low == high and low >= 240 for low, high in image.getextrema())
            if image.width > 2200:
                scale = 2200 / image.width
                image = image.resize((2200, int(image.height * scale)), __import__("PIL.Image", fromlist=["Image"]).Resampling.LANCZOS)
            out = io.BytesIO()
            image.save(out, "PNG")
            png = out.getvalue()
            pages.append(None if blank else png)
            if on_render:
                on_render(i + 1, png)
            if monitor:
                if blank:
                    monitor.count("blank_pages", total=count, page=i + 1)
                monitor.count("rendered_pages", total=count, page=i + 1)
        return pages



def ocr_pdf(pdf_path: str | None = None, pdf_base64: str | None = None,
            mode: str = "page", max_pages: int | None = None, dpi: int = 150,
            max_tokens: int = MAX_TOKENS, *, monitor=None, prompt=None, concurrency=None,
            chat_options=None, on_render=None, on_page=None, keep_partial=False) -> dict:
    if mode not in {"page", "region"}:
        raise RuntimeError("mode must be 'page' or 'region'")
    with _stage(monitor, "pipeline", "prepare_pdf"):
        raw = _read_bytes(pdf_path, None, pdf_base64, "pdf")
    with _stage(monitor, "pipeline", "render_pdf"):
        pages = _render_pdf(raw, max_pages, dpi, monitor=monitor, on_render=on_render)
    def run(i: int):
        started = time.monotonic()
        try:
            if monitor:
                monitor.check_cancelled()
            if pages[i] is None:
                return i, {"ok": True, "text": "", "usage": {}, "ms": 0, "error": ""}
            result = ocr_image(image_base64=base64.b64encode(pages[i]).decode(), mode=mode,
                               max_tokens=max_tokens, monitor=monitor, page=i + 1, prompt=prompt,
                               concurrency=concurrency, chat_options=chat_options)
            return i, {"ok": True, "text": result["markdown"], "usage": result.get("usage", {}),
                       "ms": int((time.monotonic() - started) * 1000), "error": ""}
        except Exception as exc:
            if not keep_partial:
                raise
            return i, {"ok": False, "text": f"<!-- page {i + 1} OCR failed: {exc} -->", "usage": {},
                       "ms": int((time.monotonic() - started) * 1000), "error": str(exc)}
    outputs, usage = [None] * len(pages), {}
    workers = min(CONCURRENCY, max(1, concurrency or CONCURRENCY))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run, i) for i in range(len(pages))]):
            i, result = future.result()
            outputs[i] = result
            usage = merge_usage(usage, result["usage"])
            if on_page:
                on_page(i + 1, result)
            if monitor:
                if not result["ok"]:
                    monitor.count("failed_pages", total=len(pages), page=i + 1)
                monitor.count("recognized_pages", total=len(pages), page=i + 1)
    with _stage(monitor, "pipeline", "assemble_pdf"):
        body = "\n\n---\n\n".join(f"<!-- page {i+1} -->\n\n{result['text']}" for i, result in enumerate(outputs))
    failed = sum(not result["ok"] for result in outputs)
    return {"pages": len(pages), "mode": mode, "markdown": body, "usage": usage,
            "blank_pages": sum(page is None for page in pages),
            "failed_pages": failed, "warning": "One or more PDF pages failed; partial output is retained." if failed else ""}


def _schema_prompt(fields: list[str] | None) -> str:
    if not fields:
        return "Extract the key information from the document image as a JSON object. Use concise field names. Return JSON only; use an empty string for missing values."
    schema = {str(field): "" for field in fields}
    return ("Extract key information in the image\n\nPlease output the key information in JSON format according to the following schema:\n"
            + json.dumps(schema, ensure_ascii=False, indent=4))


def kie_image(image_path: str | None = None, image_url: str | None = None,
              image_base64: str | None = None, fields: list[str] | None = None,
              max_tokens: int = MAX_TOKENS, *, monitor=None, chat_options=None) -> dict:
    with _stage(monitor, "pipeline", "prepare_image"):
        png = _image_png(_read_bytes(image_path, image_url, image_base64, "image"))
    raw = _chat(png, _schema_prompt(fields), max_tokens, monitor=monitor, **(chat_options or {}))["text"]
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {"json": None, "raw": raw, "warning": "Model output did not contain a JSON object"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"json": None, "raw": raw, "warning": "Model output contained invalid JSON"}
    if fields:
        data = {field: data.get(field, "") for field in fields}
    return {"json": data}


def vqa_image(question: str, image_path: str | None = None, image_url: str | None = None,
              image_base64: str | None = None, *, monitor=None, chat_options=None) -> dict:
    if not question.strip():
        raise RuntimeError("question must not be empty")
    with _stage(monitor, "pipeline", "prepare_image"):
        png = _image_png(_read_bytes(image_path, image_url, image_base64, "image"))
    prompt = question.strip() + "\nAnswer the question using a single word or phrase."
    return {"answer": _chat(png, prompt, 1024, monitor=monitor, **(chat_options or {}))["text"]}


def _props_image():
    return {"image_path": {"type": "string", "description": "File path on the MCP server host (preferred); a remote MCP cannot read paths on the client machine"},
            "image_url": {"type": "string"},
            "image_base64": {"type": "string", "description": "Base64/data URL; may be truncated by a harness"}}


TOOLS = [
    {"name": "runtime_status", "description": "Quick status for this MCP server process, not Mify: devices, task capacity, active tasks and the latest 3 finished task summaries (last 8 available by ID). Each task has request_id, phase, elapsed_ms, page/region counts, active_pages, last_activity_ms and error/warning. Filter by request_id from a progress notification or result. Progress notification numbers are event sequences, not percentages; use page/region counts for work completed. Unknown IDs return found=false. Device inspection uses a 30s cache; request-specific lookups only read memory. This does not report a separately launched Demo/MCP process.", "inputSchema": {"type": "object", "properties": {
        "request_id": {"type": "string", "description": "Look up one active or recently finished task"},
        "verbose": {"type": "boolean", "default": False, "description": "Include active stage and per-page details; finished tasks retain only summaries"},
        "refresh": {"type": "boolean", "default": False, "description": "Refresh device evidence for the service overview; ignored for request_id lookups"}}}},
    {"name": "ocr_image", "description": "Parse one image through the MCP host's loopback inference endpoint; this MCP does not call Mify. Use mode='region' for standard printed documents (MCP detects layout, crops and OCRs regions concurrently); use mode='page' for scene text, handwriting, calligraphy, or historic/irregular documents where segmentation could split content incorrectly. The default is page. The first content block is OCR text; the following Runtime block is status, not document content.", "inputSchema": {"type": "object", "properties": {**_props_image(), "mode": {"type": "string", "enum": ["page", "region"], "default": "page"}, "max_tokens": {"type": "integer", "minimum": 1, "default": MAX_TOKENS, "description": "Maximum output tokens per page in page mode or per region in region mode. Each page/region receives its own budget. Image/prompt plus requested output must fit the server context. Increase for truncated output; reduce for context-overflow errors."}}}},
    {"name": "ocr_pdf", "description": "Parse a PDF to Markdown through the MCP host's loopback inference endpoint; this MCP does not call Mify. Standard printed documents may use mode='region' for internal layout detection and concurrent region OCR; scene text, handwriting, calligraphy and historical documents should use mode='page'. The first content block is Markdown; the following Runtime block is status, not document content.", "inputSchema": {"type": "object", "properties": {"pdf_path": {"type": "string", "description": "File path on the MCP server host, not the client machine"}, "pdf_base64": {"type": "string"}, "mode": {"type": "string", "enum": ["page", "region"], "default": "page"}, "max_pages": {"type": "integer"}, "dpi": {"type": "integer", "default": 150}, "max_tokens": {"type": "integer", "minimum": 1, "default": MAX_TOKENS, "description": "Maximum output tokens per page in page mode or per region in region mode. Each page/region receives its own budget. Image/prompt plus requested output must fit the server context. Increase for truncated output; reduce for context-overflow errors."}}}},
    {"name": "kie_image", "description": "Extract key information as JSON through the MCP host's loopback inference endpoint; this MCP does not call Mify. Optionally provide field names as a schema. Image reading, prompting and JSON cleanup are handled by this MCP service. On success, parse only the first content block as JSON; the following Runtime block is separate status text with a request_id for runtime_status.", "inputSchema": {"type": "object", "properties": {**_props_image(), "fields": {"type": "array", "items": {"type": "string"}, "description": "Optional output field names"}, "max_tokens": {"type": "integer", "minimum": 1, "default": MAX_TOKENS, "description": "Maximum output tokens for this extraction. Image/prompt plus requested output must fit the server context. Increase for truncated output; reduce for context-overflow errors."}}}},
    {"name": "vqa_image", "description": "Answer a document-image question through the MCP host's loopback inference endpoint; this MCP does not call Mify. The MCP applies the concise OCR-VQA answer format; provide the image and question only. On success, parse only the first content block as JSON; the following Runtime block is separate status text with a request_id for runtime_status.", "inputSchema": {"type": "object", "properties": {**_props_image(), "question": {"type": "string"}}, "required": ["question"]}},
]


def _run_tool(name, args, monitor, on_render=None, on_page=None):
    options = {k: args[k] for k in ("model_url", "model", "temperature") if k in args}
    if name in {"ocr_image", "kie_image", "vqa_image"}:
        params = {k: args.get(k) for k in ("image_path", "image_url", "image_base64")}
        if name == "ocr_image":
            return ocr_image(**params, mode=args.get("mode", "page"), max_tokens=int(args.get("max_tokens", MAX_TOKENS)),
                             prompt=args.get("prompt"), concurrency=args.get("concurrency"), monitor=monitor, chat_options=options)
        if name == "kie_image":
            return kie_image(**params, fields=args.get("fields"), max_tokens=int(args.get("max_tokens", MAX_TOKENS)), monitor=monitor, chat_options=options)
        return vqa_image(**params, question=args.get("question", ""), monitor=monitor, chat_options=options)
    return ocr_pdf(pdf_path=args.get("pdf_path"), pdf_base64=args.get("pdf_base64"), mode=args.get("mode", "page"),
                   max_pages=args.get("max_pages"), dpi=int(args.get("dpi", 150)), max_tokens=int(args.get("max_tokens", MAX_TOKENS)),
                   prompt=args.get("prompt"), concurrency=args.get("concurrency"), monitor=monitor, chat_options=options,
                   on_render=on_render, on_page=on_page, keep_partial=args.get("keep_partial", False))


def execute_tool(name, args, notify=None, observe=None, cancelled=None, on_render=None, on_page=None):
    """One execution/monitoring path for the browser and MCP transports."""
    if name not in {"ocr_image", "ocr_pdf", "kie_image", "vqa_image"}:
        return {"ok": False, "error": f"unknown tool: {name}"}
    if not _REQUEST_SLOTS.acquire(blocking=False):
        return {"ok": False, "error": "Service busy: two tasks are already running; retry after one completes"}
    monitor = CallMonitor(name, notify, observe, cancelled)
    try:
        with monitor:
            with monitor.stage("pipeline", "inspect_runtime"):
                monitor.component("layout", layout_runtime_status())
                monitor.component("ocr", probe_ocr_runtime(_local_url(args.get("model_url")), args.get("model") or OCR_MODEL))
            result = _run_tool(name, args, monitor, on_render, on_page)
            monitor.warning = result.get("warning") or None
            if monitor.components["ocr"].get("loaded") is False:
                monitor.component("ocr", probe_ocr_runtime(_local_url(args.get("model_url")), args.get("model") or OCR_MODEL, refresh=True))
        return {"ok": True, "result": result, "monitor": monitor.snapshot()}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "monitor": monitor.snapshot()}
    finally:
        _REQUEST_SLOTS.release()


def _tool_call(name: str, args: dict, notify=None) -> dict:
    if name == "runtime_status":
        try:
            info = runtime_status(request_id=args.get("request_id"), verbose=args.get("verbose", False), refresh=args.get("refresh", False))
            return {"content": [{"type": "text", "text": json.dumps(info, ensure_ascii=False, indent=2)}]}
        except Exception as exc:
            return {"content": [{"type": "text", "text": f"ERROR: {exc}"}], "isError": True}
    outcome = execute_tool(name, args, notify=notify)
    if outcome["ok"]:
        result = outcome["result"]
        # Transport formatting is separate from shared execution and monitoring.
        text = json.dumps(result, ensure_ascii=False, indent=2) if name in {"kie_image", "vqa_image"} else result["markdown"]
        response = {"content": [{"type": "text", "text": text}]}
    else:
        response = {"content": [{"type": "text", "text": "ERROR: " + outcome["error"]}], "isError": True}
    if "monitor" in outcome:
        response["_meta"] = {"xiaomi_ocr": outcome["monitor"]}
        response["content"].append({"type": "text", "text": "Runtime: " + status_message(outcome["monitor"]["summary"])})
    return response


def main() -> None:
    # PaddleX and other dependencies can print to stdout. Reserve the original
    # stream for write_rpc so diagnostics cannot corrupt the stdio protocol.
    reserve_protocol_stdout()

    def run_call(msg_id, params):
        metadata = params.get("_meta")
        token = metadata.get("progressToken") if isinstance(metadata, dict) else None
        notify = None
        if isinstance(token, (str, int)) and not isinstance(token, bool):
            def notify(progress, message):
                write_rpc({"jsonrpc": "2.0", "method": "notifications/progress", "params": {
                    "progressToken": token, "progress": progress, "message": message}})
        result = _tool_call(params.get("name", ""), params.get("arguments") or {}, notify)
        write_rpc({"jsonrpc": "2.0", "id": msg_id, "result": result})

    # Keep reading control/status requests during long-running OCR calls.
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool, ThreadPoolExecutor(max_workers=2) as status_pool:
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            method, params, msg_id = msg.get("method"), msg.get("params") or {}, msg.get("id")
            if msg_id is None:
                continue
            if not isinstance(params, dict):
                write_rpc({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32602, "message": "params must be an object"}})
                continue
            if method == "initialize":
                result = {"protocolVersion": params.get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "xiaomi-ocr", "version": "0.3.0"}}
            elif method == "ping": result = {}
            elif method == "tools/list": result = {"tools": TOOLS}
            elif method == "tools/call":
                target_pool = status_pool if params.get("name") == "runtime_status" else pool
                target_pool.submit(run_call, msg_id, params)
                continue
            else:
                write_rpc({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"method not found: {method}"}})
                continue
            if msg_id is not None:
                write_rpc({"jsonrpc": "2.0", "id": msg_id, "result": result})


if __name__ == "__main__":
    main()
