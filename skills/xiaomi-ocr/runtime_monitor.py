"""MCP stage progress and cached, read-only runtime inspection."""
import copy
import http.client
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from collections import deque

WRITE_LOCK = threading.RLock()
RPC_OUTPUT = sys.stdout
_CALLS_LOCK = threading.Lock()
_CALLS = {}
_RECENT_CALLS = deque(maxlen=8)
_RUNTIME_CACHE = {}
_PROBE_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()
_GPU_CACHE = {}


def reserve_protocol_stdout():
    sys.stdout = sys.stderr


def write_rpc(message):
    with WRITE_LOCK:
        print(json.dumps(message, ensure_ascii=False), file=RPC_OUTPUT, flush=True)


def active_calls():
    with _CALLS_LOCK:
        calls = list(_CALLS.values())
    return [call.snapshot() for call in calls]


def recent_calls():
    with _CALLS_LOCK:
        return copy.deepcopy(list(_RECENT_CALLS))


def compact_devices(components):
    keys = ("device", "display_device", "gpu_indices", "loaded", "requested_device", "backend", "source", "cache_age_ms", "note")
    return {name: {key: info[key] for key in keys if key in info} for name, info in components.items()}


def device_display(info):
    """Use host nvidia-smi indices, never a process-local CUDA ordinal, for display."""
    if info.get("loaded") is False:
        return "not_loaded"
    kind = str(info.get("device", "unknown")).lower().split(":")[0]
    if kind == "cpu":
        return "CPU"
    if kind in {"gpu", "cuda", "cpu+gpu"}:
        label = "CPU + GPU" if kind == "cpu+gpu" else "GPU"
        indices = info.get("gpu_indices", [])
        return label + (" " + ", ".join(map(str, indices)) if indices else "")
    return "unknown"


def summarize(state):
    """The same bounded task view for Demo, status queries and MCP notifications."""
    stages, counts = state["active_stages"], state["progress_counts"]
    current = next((s for s in stages if s["stage"] == "inference"), stages[0] if stages else {})
    pages = counts.get("pipeline/recognized_pages")
    rendered = counts.get("pipeline/rendered_pages")
    regions = [v for k, v in counts.items() if k.startswith("ocr/regions")]
    summary = {k: state[k] for k in ("request_id", "tool", "status", "elapsed_ms", "error", "warning")}
    summary.update(phase=current.get("stage", "between_stages") if state["status"] == "running" else state["status"],
                   component=current.get("component"), active_pages=sorted({s["page"] for s in stages if "page" in s}),
                   last_activity_ms=state["last_activity_ms"], devices=compact_devices(state["components"]))
    if rendered:
        summary["rendered_pages"] = rendered
        summary["pages"] = {"completed": pages["completed"] if pages else 0, "total": rendered["total"],
                            "failed": state["counters"].get("failed_pages", 0), "blank": state["counters"].get("blank_pages", 0)}
    if regions:
        summary["regions"] = {"completed": sum(c["completed"] for c in regions), "total": sum(c["total"] for c in regions),
                              "total_is_final": not rendered or len(regions) == rendered["total"] or state["status"] in {"completed", "partial"}}
    return summary


def status_message(summary):
    parts = [summary["request_id"], summary["status"], summary["phase"], f"{summary['elapsed_ms'] / 1000:.1f}s"]
    for name in ("pages", "regions"):
        count = summary.get(name)
        if count:
            total = str(count["total"]) if count.get("total_is_final", True) else f"{count['total']} known"
            parts.append(f"{name} {count['completed']}/{total}")
            if name == "pages":
                parts.extend(f"{key} {count[key]}" for key in ("failed", "blank") if count.get(key))
    if summary["active_pages"]:
        parts.append("active pages " + ",".join(map(str, summary["active_pages"])))
    parts.extend(f"{name}={info.get('display_device', 'unknown')}" for name, info in summary["devices"].items())
    if summary["error"] or summary["warning"]:
        parts.append(summary["error"] or summary["warning"])
    return " · ".join(parts)


class CallMonitor:
    """Thread-safe stage events, completion counts, and five-second heartbeats."""
    def __init__(self, tool, notify=None, observe=None, cancelled=None):
        self.id, self.tool, self.notify = uuid.uuid4().hex[:12], tool, notify
        self.observe, self.cancelled = observe, cancelled
        self.started = time.monotonic()
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.sequence, self.status = 0, "running"
        self.components, self.active, self.counters, self.progress_counts = {}, {}, {}, {}
        self.last_event = {}
        self.last_activity = self.started
        self.error, self.warning = None, None

    def check_cancelled(self):
        if self.cancelled and self.cancelled():
            raise RuntimeError("Request cancelled")

    def __enter__(self):
        with _CALLS_LOCK:
            _CALLS[self.id] = self
        self.emit("pipeline", "request", "Request started")
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        self.status = "cancelled" if self.cancelled and self.cancelled() else "failed" if exc_type else "partial" if self.warning else "completed"
        self.error = str(exc) if exc_type else None
        self.emit("pipeline", "request", "Request " + self.status, status=self.status)
        summary = self.snapshot()["summary"]
        summary["finished_at"] = time.time()
        with _CALLS_LOCK:
            _CALLS.pop(self.id, None)
            _RECENT_CALLS.appendleft(summary)

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            state = {"request_id": self.id, "tool": self.tool, "status": self.status,
                    "progress": self.sequence, "elapsed_ms": int((now - self.started) * 1000),
                    "last_activity_ms": int((now - self.last_activity) * 1000), "error": self.error, "warning": self.warning,
                    "components": copy.deepcopy(self.components), "counters": dict(self.counters),
                    "last_event": dict(self.last_event),
                    "progress_counts": copy.deepcopy(self.progress_counts),
                    "active_stages": [dict(context, elapsed_ms=int((now - start) * 1000))
                                      for start, context in self.active.values()]}
            state["summary"] = summarize(state)
            return state

    def emit(self, component, stage, message, **details):
        with self.lock:
            self.sequence += 1
            if stage != "heartbeat":
                self.last_activity = time.monotonic()
            if "completed" in details:
                key = f"{component}/{stage}"
                if stage == "regions" and "page" in details:
                    key += f"/page={details['page']}"
                self.progress_counts[key] = {k: details[k] for k in ("completed", "total")}
            event = dict(event="ocr_progress", request_id=self.id, tool=self.tool,
                         progress=self.sequence, component=component, stage=stage, message=message,
                         elapsed_ms=int((time.monotonic() - self.started) * 1000), **details)
            self.last_event = {k: event[k] for k in ("component", "stage", "message", "status", "page", "region") if k in event}
            with WRITE_LOCK:
                print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
            state = self.snapshot() if self.notify or self.observe else None
            if self.notify:
                self.notify(self.sequence, status_message(state["summary"]))
            if self.observe:
                self.observe(state)

    def component(self, name, info):
        with self.lock:
            if self.components.get(name) != info:
                self.components[name] = info
                self.emit(name, "runtime", name + " runtime: " + json.dumps(info, ensure_ascii=False), runtime=info)

    @contextmanager
    def stage(self, component, stage, **context):
        self.check_cancelled()
        key, start = uuid.uuid4().hex, time.monotonic()
        label = f"{component}/{stage} " + " ".join(f"{k}={v}" for k, v in context.items())
        with self.lock:
            self.active[key] = (start, dict(component=component, stage=stage, **context))
            self.emit(component, stage, label + " started", status="running", **context)
        status = "completed"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            with self.lock:
                self.active.pop(key)
                ms = int((time.monotonic() - start) * 1000)
                self.emit(component, stage, f"{label} {status} ({ms} ms)",
                          status=status, duration_ms=ms, **context)

    def count(self, name, total, **context):
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + 1
            done = self.counters[name]
            self.emit("pipeline", name, f"{name}: {done}/{total}", completed=done, total=total, **context)

    def _heartbeat(self):
        while not self.stop.wait(5):
            with self.lock:
                state = self.snapshot()
                labels = [f"{s['component']}/{s['stage']} ({s['elapsed_ms']} ms)" for s in state["active_stages"]]
                self.emit("pipeline", "heartbeat", "Still running: " + (", ".join(labels) or "between stages"),
                          active_stages=state["active_stages"], progress_counts=state["progress_counts"])


def gpu_placement(pids, refresh=False):
    """Cache process residency for 30s. UUIDs are only used internally to join indices."""
    key = frozenset(pids)
    if not key:
        return {"gpu_indices": [], "note": "No matching process found; physical GPU index remains unknown"}
    with _GPU_LOCK:
        now = time.monotonic()
        for old_key in list(_GPU_CACHE):
            if now - _GPU_CACHE[old_key][0] >= 30:
                del _GPU_CACHE[old_key]
        if key in _GPU_CACHE and not refresh:
            return copy.deepcopy(_GPU_CACHE[key][1])
        try:
            def query(fields):
                output = subprocess.check_output(["nvidia-smi", fields, "--format=csv,noheader,nounits"],
                                                 text=True, stderr=subprocess.DEVNULL, timeout=1)
                return [[part.strip() for part in row.split(",")] for row in output.splitlines()]
            indices = {uuid: int(index) for index, uuid in query("--query-gpu=index,uuid")}
            rows = query("--query-compute-apps=pid,gpu_uuid,used_gpu_memory")
            matched = [{"pid": int(pid), "gpu_index": indices[uuid], "used_memory_mib": memory}
                       for pid, uuid, memory in rows if int(pid) in key and uuid in indices]
            info = {"gpu_indices": sorted({p["gpu_index"] for p in matched}), "gpu_processes": matched}
            if not matched:
                info["note"] = "No matching GPU residency observed; physical GPU index remains unknown"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            info = {"gpu_indices": [], "note": "GPU process inspection unavailable: " + str(exc)[:200]}
        _GPU_CACHE[key] = (time.monotonic(), info)
        return copy.deepcopy(info)


def _server_gpu_placement(base, refresh=False):
    """Observe only GPU processes belonging to the listener and its workers."""
    if sys.platform == "darwin":
        return {}
    address = urllib.parse.urlsplit(base)
    def command(args):
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=1)
    try:
        pids = set(map(int, command(["lsof", "-nP", "-a", f"-iTCP@{address.hostname}:{address.port or 80}",
                                     "-sTCP:LISTEN", "-t"]).split()))
        tree = [tuple(map(int, line.split())) for line in command(["ps", "-e", "-o", "pid=", "-o", "ppid="]).splitlines()]
        children = pids
        while children:
            children = {pid for pid, parent in tree if parent in children} - pids
            pids.update(children)
        return gpu_placement(pids, refresh=refresh)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"note": "GPU process inspection unavailable: " + str(exc)[:200]}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe_ocr_runtime(base, model, refresh=False):
    """Cache placement for 30s; explicit runtime_status requests can refresh it."""
    # A cache miss shared by simultaneous Demo/MCP calls performs one probe.
    with _PROBE_LOCK:
        return _probe_ocr_runtime(base, model, refresh)


def _probe_ocr_runtime(base, model, refresh):
    key, now = (base, model), time.monotonic()
    with _CALLS_LOCK:
        cached = _RUNTIME_CACHE.get(key)
        if cached and not refresh and now - cached[0] < 30:
            return dict(copy.deepcopy(cached[1]), cache_age_ms=int((now - cached[0]) * 1000))
    root = base.rstrip("/").removesuffix("/v1")
    info = {"model": model, "endpoint": base, "backend": "unknown", "device": "unknown", "source": "unavailable"}
    errors = {}
    opener = urllib.request.build_opener(_NoRedirect)
    def read(path):
        try:
            with opener.open(root + path, timeout=.75) as response:
                data = json.load(response)
                if not isinstance(data, dict):
                    raise ValueError("Runtime metadata must be a JSON object")
                return data
        except (OSError, ValueError, http.client.HTTPException) as exc:
            errors[path] = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else str(exc)[:200]
            return {}
    models = read("/v1/models")
    if "/v1/models" in errors:
        info.update(note="Model list unavailable", errors=errors)
    else:
        entry = next((m for m in models.get("data", []) if m.get("id") == model), {})
        owner = entry.get("owned_by", "").lower()
        paths = {"vllm": ["/server_info?config_format=json"], "sglang": ["/server_info", "/get_server_info"]}
        if owner in paths:
            info["backend"] = owner
        for path in paths.get(owner, ["/server_info?config_format=json", "/get_server_info", "/api/ps"]):
            data = read(path)
            if "vllm_config" in data:
                device = data["vllm_config"].get("device_config", {})
                info.update(backend="vllm", device=device.get("device") or device.get("device_type") or "unknown", source="server /server_info")
            elif "server_args" in data or ("model_path" in data and "device" in data):
                args = data.get("server_args", data)
                info.update(backend="sglang", device=args.get("device") or "unknown", base_gpu_id=args.get("base_gpu_id"),
                            tp_size=args.get("tp_size"), source="server " + path)
            elif path == "/api/ps" and "models" in data:
                names = {model, model + ":latest"}
                loaded = next((m for m in data["models"] if m.get("name") in names or m.get("model") in names), None)
                info.update(backend="ollama", loaded=loaded is not None, source="server /api/ps")
                if loaded:
                    size, vram = loaded.get("size"), loaded.get("size_vram")
                    if isinstance(size, int) and size > 0 and isinstance(vram, int) and vram >= 0:
                        info.update(device="cpu" if vram == 0 else "gpu" if vram >= size else "cpu+gpu",
                                    size_bytes=size, size_vram_bytes=vram)
                else:
                    info["note"] = "Configured model is not loaded"
            if info["source"] != "unavailable":
                break
        kind = str(info["device"]).lower().split(":")[0]
        if kind in {"gpu", "cuda", "cpu+gpu", "unknown"} and info.get("loaded") is not False:
            placement = _server_gpu_placement(base, refresh=refresh)
            info.update(placement)
            if placement.get("gpu_indices") and kind == "unknown":
                info.update(device="gpu", source="listener/worker PIDs + nvidia-smi")
        if errors:
            info["errors"] = errors
    info["display_device"] = device_display(info)
    with _CALLS_LOCK:
        _RUNTIME_CACHE[key] = (time.monotonic(), copy.deepcopy(info))
    return dict(info, cache_age_ms=0)
