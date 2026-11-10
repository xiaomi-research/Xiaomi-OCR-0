"""Opt-in Qwen3.5 Ollama transport, shared by the demo and MCP.

Use a verified Xiaomi-OCR-0 BF16 model. This path uses raw generation to
preserve the checkpoint's non-thinking prompt and Markdown whitespace.
It does not claim bitwise numerical equivalence between inference engines.
"""
from __future__ import annotations

import base64
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageOps


def prepare_image(raw: bytes) -> tuple[bytes, int]:
    """Match this checkpoint's 16px patch / 2x merge dynamic image grid."""
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    h, w = image.height, image.width
    if max(h, w) / min(h, w) > 200:
        raise ValueError("Image aspect ratio exceeds the Qwen3.5 limit of 200")
    factor, min_pixels, max_pixels = 32, 65536, 16777216
    rh, rw = round(h / factor) * factor, round(w / factor) * factor
    if rh * rw > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        rh, rw = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif rh * rw < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        rh, rw = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    if image.size != (rw, rh):
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), rh * rw // (factor * factor)


def render_prompt(prompt: str) -> str:
    """HF single-user, one-image template with enable_thinking=False.

    Ollama expands [img-0] into the vision delimiters and image embeddings.
    Its marker is deliberately not wrapped in a second pair of delimiters.
    """
    return ("<|im_start|>user\n[img-0]" + prompt.strip()
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def complete(image: bytes, prompt: str, max_tokens: int, *, base_url: str,
             model: str, temperature: float = 0, num_ctx: int = 32768) -> dict:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Ollama inference must use a loopback HTTP endpoint")
    # Accept the same /v1 endpoint used in existing repository configurations.
    base = base_url.rstrip("/").removesuffix("/chat/completions").removesuffix("/v1").removesuffix("/api")
    png, visual_tokens = prepare_image(image)
    if max_tokens < 1 or visual_tokens + max_tokens >= num_ctx:
        raise ValueError("Image and output budget exceed the Ollama context; reduce image size or max_tokens")
    payload = {"model": model, "prompt": render_prompt(prompt),
               "images": [base64.b64encode(png).decode()], "raw": True,
               "stream": False, "truncate": False,
               "options": {"temperature": temperature, "repeat_penalty": 1,
                           "seed": 0, "num_ctx": num_ctx, "num_predict": max_tokens}}
    request = urllib.request.Request(base + "/api/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama is unavailable at {base}: {exc.reason}") from exc
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    if data.get("done_reason") == "length":
        raise RuntimeError(f"Model output was truncated (max_tokens={max_tokens}). Increase max_tokens within the model context limit, or use smaller inputs")
    text = data.get("response", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Ollama returned an empty response")
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    return {"text": text, "usage": {"prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}}
