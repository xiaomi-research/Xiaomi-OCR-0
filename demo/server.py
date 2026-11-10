#!/usr/bin/env python3
"""Xiaomi-OCR-0 Demo — local model-backed web app and static server.

Uses the repository requirements and optional PaddleX for region detection.
Run:
    python3 server.py --port 8787
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import mimetypes
import math
import os
import sys
import threading
import time
import uuid
import urllib.request
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SAMPLES = ROOT / "samples"
EXAMPLE_PICS = ROOT.parent / "example_pics"
BOOK_TMP = ROOT / "out" / "books"

MAX_BODY = 80 * 1024 * 1024  # 80 MB (PDF books)

LOCAL_MODEL_URL = os.environ.get("XIAOMI_OCR_LOCAL_URL", "http://127.0.0.1:8000/v1").strip()
LOCAL_MODEL = os.environ.get("XIAOMI_OCR_MODEL", "SeerRay-Lab/Xiaomi-OCR-0").strip()

_MCP_MODULE = None
_MCP_LOCK = threading.Lock()


def mcp_pipeline():
    """Load the shared Skill implementation so Demo and MCP use the same transforms."""
    global _MCP_MODULE
    with _MCP_LOCK:
        return _load_mcp_pipeline()


def _load_mcp_pipeline():
    global _MCP_MODULE
    if _MCP_MODULE is None:
        source = ROOT.parent / "skills" / "xiaomi-ocr" / "mcp_ocr_server.py"
        spec = importlib.util.spec_from_file_location("xiaomi_ocr_mcp_pipeline", source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the shared Xiaomi OCR MCP pipeline")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MCP_MODULE = module
    return _MCP_MODULE


def validate_request(req):
    """Reject malformed API input before opening a stream or launching a job."""
    if not isinstance(req, dict):
        raise ValueError("request body must be a JSON object")
    for key in ("task", "prompt", "fields", "question", "mode", "image_base64",
                "image_mime", "pdf_base64", "pdf_name", "sample"):
        if key in req and not isinstance(req[key], str):
            raise ValueError(f"{key} must be a string")
    for key, low, high in (("max_tokens", 1, 16384), ("concurrency", 1, 32),
                           ("dpi", 36, 300), ("max_pages", 1, 10000)):
        if key not in req or (key == "max_pages" and req[key] is None):
            continue
        value = req[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
    if "temperature" in req:
        value = req["temperature"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2:
            raise ValueError("temperature must be a number between 0 and 2")
    if "task" in req:
        req["task"] = req["task"].strip().lower()
        if req["task"] not in {"document", "kie", "vqa"}:
            raise ValueError("task must be document, kie, or vqa")
    return req


def region_runtime_status():
    """Check installed layout dependencies without loading the model or downloading weights."""
    dependencies = {"paddle": "PaddlePaddle", "paddlex": "PaddleX", "numpy": "NumPy",
                    "PIL": "Pillow", "cv2": "OpenCV", "shapely": "Shapely"}
    missing = [name for module, name in dependencies.items()
               if importlib.util.find_spec(module) is None]
    return {"available": not missing, "missing": missing}


def model_health():
    """Probe the configured model; a running demo alone does not imply inference is ready."""
    mcp = mcp_pipeline()
    base = mcp._local_url(LOCAL_MODEL_URL)
    base = base.removesuffix("/chat/completions").rstrip("/")
    url = base + ("/models" if base.endswith("/v1") else "/v1/models")
    with urllib.request.urlopen(url, timeout=3) as response:
        models = json.load(response).get("data", [])
    ready = any(model.get("id") == LOCAL_MODEL for model in models)
    return {"ready": ready, "model": LOCAL_MODEL,
            "error": "" if ready else "Configured model is not listed by the inference server"}


# The Demo owns HTTP and job storage only. Inference, counters and device
# inspection all run in the same execute_tool() used by the MCP transport.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def job_get(jid):
    with JOBS_LOCK:
        job = JOBS.get(jid)
        return dict(job) if job else None


def job_update(jid, **values):
    with JOBS_LOCK:
        JOBS.setdefault(jid, {}).update(values)


def demo_arguments(req, mode="page"):
    task = req.get("task", "document")
    if not req.get("image_base64"):
        raise ValueError("image is required")
    args = {"image_base64": req["image_base64"], "max_tokens": req.get("max_tokens", 4096),
            "model_url": LOCAL_MODEL_URL, "model": LOCAL_MODEL, "temperature": req.get("temperature", 0.0)}
    if task == "kie":
        args["fields"] = [x.strip() for x in req.get("fields", "").replace("，", ",").replace("\n", ",").split(",") if x.strip()]
        return "kie_image", args
    if task == "vqa":
        if not req.get("question", "").strip():
            raise ValueError("请输入关于图像内容的问题")
        args["question"] = req["question"]
        return "vqa_image", args
    args.update(mode=mode, prompt=req.get("prompt"), concurrency=req.get("concurrency"))
    return "ocr_image", args


def demo_result(outcome):
    if not outcome["ok"]:
        return {"error": outcome["error"], "monitor": outcome.get("monitor")}
    result, monitor = outcome["result"], outcome["monitor"]
    if "markdown" in result:
        content = result["markdown"]
    elif "answer" in result:
        content = result["answer"]
    else:
        content = json.dumps(result["json"] if result.get("json") is not None else result, ensure_ascii=False, indent=2)
    return {"content": content, "model": LOCAL_MODEL, "usage": result.get("usage", {}),
            "latency_ms": monitor["elapsed_ms"], "monitor": monitor,
            "regions": result.get("region_results", []), "layout_ms": result.get("layout_ms", 0),
            "concurrency": result.get("concurrency", mcp_pipeline().CONCURRENCY),
            "pages": result.get("pages"), "blank_pages": result.get("blank_pages", 0), "warning": result.get("warning", "")}


def run_book_job(jid, pdf_path, max_tokens, max_pages, dpi, concurrency, prompt, mode):
    try:
        _run_book_job(jid, pdf_path, max_tokens, max_pages, dpi, concurrency, prompt, mode)
    except Exception as exc:
        job_update(jid, status="error", error=str(exc))


def _run_book_job(jid, pdf_path, max_tokens, max_pages, dpi, concurrency, prompt, mode):
    pages_dir = BOOK_TMP / jid / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    job_update(jid, pages_dir=str(pages_dir), status="running", page_log=[])

    def observe(state):
        counts = state["progress_counts"]
        rendered = counts.get("pipeline/rendered_pages", {})
        recognized = counts.get("pipeline/recognized_pages", {})
        phase = "render" if any(s["stage"] == "render_pdf" for s in state["active_stages"]) else "ocr"
        job_update(jid, monitor=state, total=rendered.get("total", 0), done=recognized.get("completed", 0),
                   phase=phase, elapsed_ms=state["elapsed_ms"])

    def on_render(page, png):
        (pages_dir / f"p{page:04d}.png").write_bytes(png)

    def on_page(page, result):
        job = job_get(jid)
        log = job.get("page_log", []) + [{"page": page, "ok": result["ok"], "ms": result["ms"],
                                         "chars": len(result["text"]), "error": result["error"],
                                         "img": f"/api/book/{jid}/page/{page}"}]
        job_update(jid, page_log=sorted(log, key=lambda row: row["page"]),
                   usage=mcp_pipeline().merge_usage(job.get("usage", {}), result["usage"]))

    args = {"pdf_path": str(pdf_path), "max_tokens": max_tokens, "max_pages": max_pages, "dpi": dpi,
            "concurrency": concurrency, "prompt": prompt, "mode": mode, "keep_partial": True,
            "model_url": LOCAL_MODEL_URL, "model": LOCAL_MODEL}
    outcome = mcp_pipeline().execute_tool("ocr_pdf", args, observe=observe,
                                         cancelled=lambda: (job_get(jid) or {}).get("cancel", False),
                                         on_render=on_render, on_page=on_page)
    if (job_get(jid) or {}).get("cancel"):
        job_update(jid, status="cancelled", monitor=outcome.get("monitor"))
    elif outcome["ok"]:
        result = demo_result(outcome)
        out_path = BOOK_TMP / f"{jid}.md"
        out_path.write_text(result["content"], encoding="utf-8")
        job_update(jid, status="error" if result["warning"] else "done", error=result["warning"],
                   markdown=result["content"], result_path=str(out_path), usage=result["usage"],
                   elapsed_ms=result["latency_ms"], monitor=result["monitor"])
    else:
        job_update(jid, status="error", error=outcome["error"], monitor=outcome.get("monitor"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("[demo] %s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            health = {"region": region_runtime_status()}
            try:
                health.update(model_health())
            except Exception as exc:
                health.update(ready=False, model=LOCAL_MODEL, error=str(exc))
            return self._json(200, health)
        if path == "/api/runtime":
            return self._json(200, mcp_pipeline().runtime_status(LOCAL_MODEL_URL, LOCAL_MODEL, verbose=True))
        if path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()
        if path.startswith("/samples/"):
            return self._serve_samples()
        if path.startswith("/example_pics/"):
            return self._serve_example_pics()
        if path.startswith("/api/book/"):
            # /api/book/<id> | /api/book/<id>/raw | /api/book/<id>/page/<n>
            parts = path.strip("/").split("/")
            # ["api","book", id, ...]
            if len(parts) >= 4 and parts[3] == "raw":
                return self._book_raw(parts[2])
            if len(parts) >= 5 and parts[3] == "page":
                return self._book_page(parts[2], parts[4])
            return self._book_status(path)
        return super().do_GET()

    def _serve_samples(self):
        rel = self.path.split("/samples/", 1)[1]
        return self._serve_file_from(SAMPLES, rel, "sample")

    def _serve_example_pics(self):
        rel = self.path.split("/example_pics/", 1)[1]
        return self._serve_file_from(EXAMPLE_PICS, rel, "example")

    def _serve_file_from(self, root: Path, rel: str, label: str):
        rel = rel.split("?", 1)[0]
        base = root.resolve()
        fp = (base / rel).resolve()
        if not str(fp).startswith(str(base) + os.sep) and fp != base:
            self.send_error(403, "forbidden")
            return
        if not fp.is_file():
            self.send_error(404, f"{label} not found")
            return
        ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _book_status(self, path: str):
        # /api/book/<id>
        jid = path.rsplit("/", 1)[-1]
        job = job_get(jid)
        if not job:
            self._json(404, {"error": "job not found"})
            return
        # do not ship full markdown on every poll
        public = {k: v for k, v in job.items() if k != "markdown"}
        public["has_markdown"] = bool(job.get("markdown"))
        self._json(200, public)

    def _book_raw(self, jid: str):
        job = job_get(jid)
        if not job:
            self._json(404, {"error": "job not found"})
            return
        md = job.get("markdown") or ""
        self._json(200, {
            "id": jid,
            "status": job.get("status"),
            "markdown": md,
            "chars": len(md),
            "elapsed_ms": job.get("elapsed_ms"),
        })

    def _book_page(self, jid: str, page_s: str):
        job = job_get(jid)
        if not job:
            self.send_error(404, "job not found")
            return
        try:
            n = int(page_s)
        except ValueError:
            self.send_error(400, "bad page")
            return
        pages_dir = job.get("pages_dir")
        if not pages_dir:
            self.send_error(404, "no pages yet")
            return
        fp = Path(pages_dir) / f"p{n:04d}.png"
        if not fp.is_file():
            self.send_error(404, "page not found")
            return
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/export":
            return self._handle_export()
        if path == "/api/ocr":
            return self._handle_run()
        if path == "/api/ocr/stream":
            return self._handle_run(stream=True)
        if path == "/api/parse/regions":
            return self._handle_run(mode="region")
        if path == "/api/parse/regions/stream":
            return self._handle_run(stream=True, mode="region")
        if path == "/api/book":
            return self._handle_book_start()
        if path.startswith("/api/book/") and path.endswith("/cancel"):
            jid = path.split("/")[-2]
            if not job_get(jid):
                return self._json(404, {"error": "job not found"})
            job_update(jid, cancel=True)
            self._json(200, {"ok": True})
            return
        self.send_error(404, "not found")

    def _handle_export(self):
        """Serve an attachment over HTTP, including in browsers without blob downloads."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                raise ValueError("invalid export size")
            form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"), max_num_fields=4)
            content = form.get("content", [""])[0]
            extension = form.get("format", ["md"])[0]
            if not content or extension not in {"md", "json"}:
                raise ValueError("export requires content and md/json format")
        except (ValueError, UnicodeDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="xiaomi-ocr-result.{extension}"')
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._json(400, {"error": "invalid or missing Content-Length"})
            return None
        body = self.rfile.read(length)
        try:
            return validate_request(json.loads(body))
        except (ValueError, UnicodeDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return None

    def _handle_run(self, stream=False, mode="page"):
        req = self._read_json()
        if req is None:
            return
        try:
            name, args = demo_arguments(req, mode)
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        if not stream:
            outcome = mcp_pipeline().execute_tool(name, args)
            return self._json(200 if outcome["ok"] else 502, demo_result(outcome))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.connection.settimeout(10)
        condition, finished, disconnected = threading.Condition(), threading.Event(), threading.Event()
        pending, outcome = None, None

        def observe(state):
            nonlocal pending
            with condition:
                pending = state  # One latest snapshot: a slow browser cannot grow a queue.
                condition.notify()

        def run():
            nonlocal outcome
            try:
                outcome = mcp_pipeline().execute_tool(name, args, observe=observe, cancelled=disconnected.is_set)
            finally:
                finished.set()
                with condition:
                    condition.notify()

        def send(event, data):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        threading.Thread(target=run, daemon=True).start()
        try:
            while True:
                with condition:
                    if pending is None and not finished.is_set():
                        condition.wait(5)
                    state, pending = pending, None
                if state and not finished.is_set():
                    send("progress", state)
                if finished.is_set():
                    result = demo_result(outcome)
                    if result.get("monitor"):
                        send("progress", result["monitor"])
                    send("done" if outcome["ok"] else "error", result)
                    break
                finished.wait(.5)  # At most two intermediate browser updates per second.
        except (OSError, TimeoutError):
            disconnected.set()

    def _handle_book_start(self):
        req = self._read_json()
        if req is None:
            return

        model = LOCAL_MODEL
        max_tokens = int(req.get("max_tokens") or 4096)
        max_pages = req.get("max_pages")
        max_pages = int(max_pages) if max_pages else None
        dpi = int(req.get("dpi") or 150)
        mode = (req.get("mode") or "page").strip().lower()
        if mode not in {"page", "region"}:
            self._json(400, {"error": "PDF mode must be page or region"})
            return
        prompt = req.get("prompt")
        # Bound nested region OCR fan-out as well as page-level concurrency.
        concurrency = min(mcp_pipeline().CONCURRENCY, int(req.get("concurrency") or 2))
        sample = (req.get("sample") or "").strip()
        pdf_b64 = req.get("pdf_base64") or ""
        pdf_name = (req.get("pdf_name") or "book.pdf").strip()

        # resolve PDF source
        try:
            if sample:
                fp = (SAMPLES / os.path.basename(sample)).resolve()
                if not str(fp).startswith(str(SAMPLES.resolve()) + os.sep) or not fp.is_file():
                    self._json(404, {"error": f"sample not found: {sample}"})
                    return
                pdf_path = fp
                pdf_name = fp.name
            elif pdf_b64.startswith("data:"):
                try:
                    _, b64 = pdf_b64.split(",", 1)
                    pdf_b64 = b64
                except ValueError:
                    self._json(400, {"error": "malformed data URL"})
                    return
                pdf_path = self._save_upload(pdf_b64, pdf_name)
            elif pdf_b64:
                pdf_path = self._save_upload(pdf_b64, pdf_name)
            else:
                self._json(400, {"error": "provide sample or pdf_base64"})
                return
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": f"PDF save failed: {e}"})
            return

        sys.stderr.write(f"[book] start source={pdf_name} size={pdf_path.stat().st_size} model={model} max_pages={max_pages} conc={concurrency}\n")

        jid = uuid.uuid4().hex[:12]
        job_update(
            jid,
            status="queued",
            total=0,
            done=0,
            current=0,
            model=model,
            source=pdf_name,
            created=int(time.time()),
            cancel=False,
            concurrency=concurrency,
        )
        th = threading.Thread(
            target=run_book_job,
            args=(jid, pdf_path, max_tokens, max_pages, dpi, concurrency, prompt, mode),
            daemon=True,
        )
        th.start()
        self._json(200, {"id": jid, "source": pdf_name, "concurrency": concurrency, "mode": mode})

    def _save_upload(self, b64: str, name: str) -> Path:
        try:
            data = base64.b64decode(b64, validate=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"invalid base64: {e}") from e
        BOOK_TMP.mkdir(parents=True, exist_ok=True)
        safe = os.path.basename(name) or "book.pdf"
        fp = BOOK_TMP / f"upload_{uuid.uuid4().hex[:8]}_{safe}"
        fp.write_bytes(data)
        return fp

    def _json(self, status: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="Xiaomi-OCR-0 demo server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8787)))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    gen = ROOT / "generate_samples.py"
    if gen.is_file() and (not SAMPLES.exists() or not any(SAMPLES.iterdir())):
        try:
            import runpy

            runpy.run_path(str(gen), run_name="__main__")
        except Exception as e:  # noqa: BLE001
            print(f"[demo] sample generation skipped: {e}", file=sys.stderr)

    BOOK_TMP.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Xiaomi-OCR-0 Demo  →  {url}")
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
