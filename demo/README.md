# Xiaomi-OCR-0 Browser Demo

A compact local web UI for document parsing, key information extraction (KIE), OCR-VQA, and PDF parsing. It calls a Xiaomi-OCR-0 inference server running on the same machine.

## Start

Start Xiaomi-OCR-0 with SGLang or vLLM on `127.0.0.1:8000`, then from the repository root run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo/server.py --port 8787
```

Open <http://127.0.0.1:8787>. Set `XIAOMI_OCR_LOCAL_URL` if the inference server uses another local port. The demo only accepts loopback inference URLs.

`requirements.txt` includes the PDF renderer.

## Interface language

Use the header's **English / 中文** button to switch languages. The initial language follows
the browser preference (Chinese for `zh`, English otherwise); your choice is remembered.
Open `/?lang=en` or `/?lang=zh` to share a language-specific view. Switching languages keeps
the selected document, entered prompts/questions, current task and recognition results.
Only interface text changes; OCR output retains the document's language.

## Use

- **Document parsing:** choose whole-page or region mode. Whole-page mode has an editable prompt and suits handwriting, calligraphy, scene text, historical documents, and irregular layouts. Region mode suits regular printed documents and runs layout detection, region OCR, and output assembly in the local server.
- **KIE:** choose fields separated by commas or newlines; leave the field box empty to extract general key information. The result is parsed and displayed as JSON.
- **VQA:** enter a question about text or content in the selected image. The demo uses the same concise-answer prompt as the MCP service.
- **PDF:** upload a PDF to preview it and parse its pages in whole-page or region mode. Page rendering and OCR progress are shown in the existing status bar. PDF supports document parsing; use image inputs for KIE and VQA.
- **Output length:** open **Settings → Maximum output tokens** and save the value
  before running. The default is 4096; the Demo accepts values from 1 to 16384.
  Page mode uses this limit per page, Region mode per region, and KIE per extraction.
  Every region type uses the selected value. Each page/region in a PDF receives its
  own budget. VQA uses a fixed 1024-token limit for concise answers.

The Demo and MCP both call `execute_tool()` in `skills/xiaomi-ocr/mcp_ocr_server.py`.
Image/PDF preprocessing, layout detection, region/page scheduling, inference limits,
prompts, result assembly, and `CallMonitor` are shared. The Demo only adapts HTTP
requests, stores PDF results, and displays the monitor snapshots.

The runtime panel displays the resolved layout device and OCR device, current stages
and elapsed time, PDF rendered/recognized page counts, and region completion counts
for each page. An unloaded predictor and an unresolved device remain explicitly labeled.
Hover over a device to inspect its source, requested placement, cache age, and observed
GPU process information. GPU memory residency is not a utilization measurement.
Both components use `display_device` from the shared monitor: `GPU 0` uses the host
index returned by `nvidia-smi`, not a process-local CUDA index. UUIDs are only an
internal lookup key. A confirmed CPU runtime displays `CPU`; an unloaded model,
an unknown device, and a GPU with an unconfirmed index remain distinct. Explicit
CPU evidence takes priority over GPU processes belonging to other components.
On macOS, Ollama-reported GPU placement displays as `GPU`; NVIDIA index inspection
is skipped.

Image requests use SSE for coalesced monitor snapshots (at most two intermediate
updates per second) and a final result, rather than a separate model-token transport.
Only the latest pending snapshot is retained; network writes run outside the inference
workers. PDF jobs use the same monitor and the existing job endpoint, polled once per
second only while the task runs. The idle page fetches `/api/runtime` once; it does not
poll hardware. The lightweight connection health check remains every 30 seconds.

Device inspection is cached for 30 seconds and concurrent cache misses are serialized.
Heartbeats every five seconds only read in-memory progress; they never launch device
probes. Each service process admits at most two active tasks and shares at most four
in-flight OCR requests by default (`XIAOMI_OCR_CONCURRENCY`). Excess active tasks return
an explicit busy error. These limits are per process; separately running Demo and MCP
processes must also respect the inference server's overall capacity.


## Region-mode dependencies

Whole-page mode does **not** need PaddleX or PP-DocLayoutV3. Region mode uses PP-DocLayoutV3 for page layout detection. Install a PaddlePaddle build compatible with your operating system, Python, and (if used) CUDA version, then install PaddleX. PP-DocLayoutV3 needs PaddleX's `cv` extras and Shapely, so use the repository's `requirements-paddlex.txt` (`paddlex[cv]` + `shapely`) rather than plain `paddlex`. On first use, `paddlex.create_model("PP-DocLayoutV3")` may download the layout model weights; later runs reuse PaddleX's local cache.

- [PaddlePaddle installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html)
- [PaddleX installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html)
- [PP-DocLayoutV3 model card and weights](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)

Choose the CPU/GPU package from the official compatibility guidance for the machine that will run the demo. This optional layout runtime is separate from the Xiaomi-OCR-0 model runtime.

## Model configuration and troubleshooting

If your inference runtime uses a custom served model name, set `XIAOMI_OCR_MODEL`
to the exact ID returned by its `/v1/models` endpoint. `XIAOMI_OCR_LOCAL_URL`
defaults to `http://127.0.0.1:8000/v1`. Both variables must be set in the terminal
that launches the demo. The connection indicator checks the configured model via
`GET /api/health`; it does not imply that optional layout dependencies are installed.

If output is truncated, increase **Maximum output tokens** when the model context
has room; the error includes the budget used for that request. Input images and
the prompt also consume context. If the request exceeds the server's context limit,
reduce the output budget or input size. The Demo's maximum setting of 16384 is an
input validation limit, not a guarantee that every request can generate that much.

The UI includes explicit run/upload actions, rendered/raw/JSON views, copy and
result downloads, and a mobile layout. Samples are the repository's real receipt,
dense-equations and Qianziwen images. The handwritten-formula example is excluded.
Live failures remain errors; the demo does not silently replace them with cached results.

`GET /api/runtime` provides the shared read-only layout/OCR device inspection.
It distinguishes requested layout placement from the loaded predictor's resolved
device and labels OCR server metadata or process observations with their source.
Unavailable devices remain unknown. This endpoint uses the shared cache; explicit MCP
`runtime_status` requests also use the cache unless `refresh: true` is requested. For stage progress and device diagnostics,
see [Runtime placement and progress](../INSTALL.md#runtime-placement-and-progress).

## Regression checks

The preview regression cases can be checked with:

```bash
node --test tests/test_preview.cjs
```

These cases exercise the bundled KaTeX renderer. They do not establish runtime
compatibility; device placement and MCP progress require validation with real
PaddleX and OCR services.

## Compact task status

The status strip keeps a constant height throughout a task. It shows the current
phase, elapsed time, processed pages (or image regions), and both devices. A single
bar uses those counts; Page-image inference has no invented percentage. Details
open in a bounded, scrollable overlay with current parallel stages, per-page counts,
device evidence and errors. The shared monitor generates the summary for both
Demo and MCP; no extra polling is added.

PDF pages rendered as a uniformly light image are counted as successful blank pages
and skip layout/OCR in both modes. Page markers and previews are retained. The check
runs before resizing and requires every pixel in each color channel to be identical;
faint marks, sparse text and nonuniform scans still take the OCR path. Empty model
responses on nonblank pages remain errors. The summary reports blank and failed page
counts separately; retained partial output is marked `partial`.
