# Local installation

Xiaomi-OCR-0 runs on the user's machine. Choose a local inference runtime:
SGLang or vLLM with the `SeerRay-Lab/Xiaomi-OCR-0` Hugging Face checkpoint,
or Ollama with the experimental `longwayxu/xiaomi-ocr-0:bf16` community package.
The MCP service connects only to the loopback listener on that machine. There is no
hosted OCR service in this workflow.

Python 3.10 or newer is required. The installer uses a compatible `python3` when available
and otherwise checks `python3.13` through `python3.10`; set `PYTHON_BIN` to choose one explicitly.

## 1. Check for existing local components

Before installing anything, check whether a compatible SGLang/vLLM/Ollama server, model checkpoint,
MCP environment, or agent configuration already exists. Reuse compatible installations.
The setup does not choose a runtime on the user's behalf. Do not install SGLang/vLLM/Ollama or
download model/layout weights without the user's choice and authorization.

## 2. Start the model locally

Choose **one** of the following runtime paths.

### SGLang / vLLM

If the user chooses to install a runtime, use a build with Qwen3.5 multimodal support for
their GPU and driver. Explain which environment and model files will be downloaded first.
Runtime compatibility changes over time; use the current upstream install and model guides:

- [SGLang Qwen3.5 deployment guide](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx)
- [vLLM Qwen3.5 implementation](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)

Basic local examples (change memory and parallelism options to suit the host):

```bash
python -m sglang.launch_server \
  --model-path SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --context-length 16384
```

```bash
vllm serve SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code --max-model-len 16384
```

The checkpoint is a multimodal `Qwen3_5ForConditionalGeneration` model. Confirm that
startup completes and the runtime accepts image inputs before configuring MCP. Keep the
listener bound to `127.0.0.1`.

### Ollama (experimental)

Install [Ollama](https://ollama.com/download) and keep its local service running
through the desktop app or `ollama serve`. With model-download authorization, pull
the community BF16 package, which includes the language model and vision projector
(about 1.8 GB):

```bash
ollama pull longwayxu/xiaomi-ocr-0:bf16
```

Use `http://127.0.0.1:11434/v1` as the MCP inference URL and
`longwayxu/xiaomi-ocr-0:bf16` as the model name. Keep the service on loopback.
See [OLLAMA.md](OLLAMA.md) for conversion details and experimental validation notes.

## 3. Install the MCP package

Run the command for your selected runtime from the repository root.

For **SGLang / vLLM**:

```bash
XIAOMI_OCR_LOCAL_URL=http://127.0.0.1:8000/v1 \
XIAOMI_OCR_MODEL=SeerRay-Lab/Xiaomi-OCR-0 \
bash skills/xiaomi-ocr/install.sh --agent=codex
```

For **Ollama**:

```bash
XIAOMI_OCR_BACKEND=ollama \
XIAOMI_OCR_LOCAL_URL=http://127.0.0.1:11434/v1 \
XIAOMI_OCR_MODEL=longwayxu/xiaomi-ocr-0:bf16 \
bash skills/xiaomi-ocr/install.sh --agent=codex
```

The installer saves the backend, URL and model in the MCP configuration. Pass the
URL and model explicitly; `XIAOMI_OCR_BACKEND` selects the transport, not those values.

After the user authorizes this step, the installer creates a private Python environment,
installs the packages in `skills/xiaomi-ocr/requirements.txt`, and writes `mcp-config.json`. With `--agent=codex` it also registers the
MCP server while preserving other entries. For other agents, omit `--agent=codex`; merge this entry into their
own MCP configuration, then restart the agent. If the local model runs on another port, set
`XIAOMI_OCR_LOCAL_URL` before running the installer, for example
`http://127.0.0.1:8001/v1`.

## 4. Region mode (included in complete agent-assisted setup)

A complete agent-assisted installation includes region dependencies and a real region-mode
check when the platform supports them and the user's setup authorization covers the downloads.
Users can choose page-only setup. `install.sh` itself installs only the base MCP dependencies;
the agent selects and installs the platform-specific layout runtime as the next step.

For manual installation, use the official
[PaddlePaddle installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html)
and [PaddleX installation guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html).
After installing a compatible PaddlePaddle wheel, run this from the repository root with
the MCP environment's Python:

```bash
skills/xiaomi-ocr/.venv/bin/python -m pip install -r requirements-paddlex.txt
```

For a standalone Demo, run the same pip command with the Demo environment's Python instead.
Use `XIAOMI_OCR_LAYOUT_DEVICE=cpu` for CPU layout; verify a real region request after installation.

Whole-page mode, KIE, and VQA need only the dependencies installed by `install.sh`. For
printed-document region parsing, with the user's authorization install PaddlePaddle and
PaddleX/PP-DocLayoutV3 using versions and device wheels appropriate for the host. Install the
PaddleX extras that PP-DocLayoutV3 needs (`requirements-paddlex.txt` includes `paddlex[cv]` and
`shapely`); a plain `paddlex` install is not sufficient and fails at layout time. This may
also download layout weights. The MCP process loads the layout model,
detects/crops regions, runs OCR requests concurrently, and restores reading order. No
separate layout service is started or configured.

Use region mode for regular printed documents, especially multi-page PDFs where region OCR
can run concurrently. Use page mode for scene text, handwriting, calligraphy, and historical
or irregular layouts: an incorrect region split can lose context and compound errors.

## MCP tasks

- `ocr_image` and `ocr_pdf`: document parsing in `page` or `region` mode.
- `kie_image`: key-value extraction as JSON; optionally supply desired field names.
- `vqa_image`: answer a question about the document image.
- `runtime_status`: inspect layout/OCR devices and active calls without loading models.

Image/PDF preprocessing, prompts, layout handling, concurrency, and output formatting stay
inside the MCP tools. Model prompts request OTSL; the tools convert tables into HTML within Markdown.
Set `XIAOMI_OCR_LAYOUT_MODEL_DIR` to reuse PaddleX-format layout weights and
`XIAOMI_OCR_LAYOUT_DEVICE` to select the layout device. These variables belong in the MCP environment
(or the Demo process environment). See [`skills/xiaomi-ocr/SKILL.md`](skills/xiaomi-ocr/SKILL.md) for the
agent workflow and [`skills/xiaomi-ocr/scripts/otsl_convert.py`](skills/xiaomi-ocr/scripts/otsl_convert.py)
for optional offline conversion; it is not a required agent step.

### Runtime placement and progress

Call `runtime_status` to inspect **both** PP-DocLayoutV3 and Xiaomi-OCR-0. The layout
entry separates `requested_device` (`auto` when unset) from the loaded predictor's
resolved `device`, including CPU fallback. PaddleX may select a GPU automatically;
set `XIAOMI_OCR_LAYOUT_DEVICE=cpu` in the MCP environment and restart MCP if CPU layout
is preferred. An unloaded layout model reports `device: unknown`; inspection does
not import Paddle or download weights.

OCR inspection first checks `/v1/models` to identify the configured model's backend,
then reads its device metadata. When unavailable, a local NVIDIA fallback matches
the listener/worker PIDs obtained with `lsof`/`ps` to `nvidia-smi` GPU residency.
Results are cached for 30 seconds across OCR calls; explicit `runtime_status`
requests use the same cache unless `refresh: true` is requested. Missing metadata or tools produce `unknown` with the error
source. Server-local GPU indices and memory residency are labeled; residency is
not utilization or a measurement of CPU offload. Probes have short timeouts and do
not start or reconfigure inference services.

Clients can request progress on an OCR tool call using MCP request metadata:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ocr_pdf","arguments":{"pdf_path":"/absolute/path/document.pdf","mode":"region"},"_meta":{"progressToken":"ocr-1"}}}
```

Progress messages cover preparation, PDF rendering, layout loading/detection,
inference queueing, page/region completion, and assembly. While waiting,
a heartbeat every five seconds lists active stages and elapsed time. Generated
text is not logged. Page/region counts are reported when known; `progress` is a
monotonic event count, not an estimated token percentage. Display depends on the
MCP client. Without a progress token, structured
events remain available on stderr and the OCR result remains the normal text output.

`runtime_status` and `ping` remain responsive during OCR. Final tool results include
a `xiaomi_ocr` summary in `_meta`, with devices and completion counters; stage start/end
events include elapsed time. All concurrent page,
region, and tool requests share `XIAOMI_OCR_CONCURRENCY` inference slots (default 4) within each service process. The shared execution entry
admits at most two active tasks; excess tasks return an explicit busy error. Demo and
MCP use the same implementation; separately launched processes have separate limits.

The Demo displays these same snapshots, including both component devices, stages,
page counts and per-page region counts. Image progress is coalesced to at most two
intermediate updates per second; PDF status polling is once per second while active.
Idle pages do not poll devices. See [Demo monitoring](demo/README.md#use).

## Batch evaluation

The restored [pipeline commands](evaluation_pipeline/README.md) and [postprocessing modules](postprocess/README.md) share the document assembly used by MCP and the Demo. Install `requirements.txt` for batch inference.

### Output budget and existing Codex entries

`ocr_image`, `ocr_pdf`, and `kie_image` accept a per-call `max_tokens` argument.
The default is 4096. Page mode applies it to each page; Region mode applies it to
every region type. Each PDF page/region receives its own budget. KIE applies it to
one extraction; `vqa_image` uses a fixed 1024-token budget for concise answers.

For example, call `ocr_pdf` with these arguments to allow up to 8192 tokens per region:

```json
{"pdf_path": "/absolute/path/document.pdf", "mode": "region", "max_tokens": 8192}
```

To change the MCP default, set `XIAOMI_OCR_MAX_TOKENS` in the MCP server's configured
environment and restart that server. An explicit tool argument takes precedence.
The Demo uses its own **Settings → Maximum output tokens** value, initially 4096.

Increase the budget for truncated output only when image/prompt tokens plus the
requested output fit the model context. For context-overflow errors, reduce the
output budget or input size. Raising the limit allows longer output; it does not
force the model to generate that many tokens or change the inference concurrency.

Re-registering a previous HTTP `xiaomi-ocr` entry replaces its transport with stdio;
other MCP servers, custom environment variables, and tool preferences are preserved.

`runtime_status({})` returns a compact process-local overview. Use `request_id` from
a progress notification or the final `Runtime:` content block to query one task
without hardware probes. The last eight completed task summaries are retained in
memory (overview lists three); unknown/evicted IDs return `found: false`. Use
`verbose: true` for active per-page/stage details. `progress` is an event sequence,
not percent complete. The first result content block remains OCR output; the separate
runtime block is status metadata for the agent, not extracted document content.
