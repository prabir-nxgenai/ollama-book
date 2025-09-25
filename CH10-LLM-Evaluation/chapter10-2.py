# ------------------------------ Standard Library ------------------------------
import requests       # HTTP client used to call the Ollama REST endpoints
import json           # For parsing JSON and JSON Lines (JSONL) chunks
import time           # For wall-clock timing (TTFT and total durations)
import threading      # For background sampling of CPU/RAM/GPU metrics
import shutil         # Used to locate 'nvidia-smi' as a fallback for GPU util
import subprocess     # Used to shell out to 'nvidia-smi' if NVML is not available
import os             # For environment variables to customize URL/port

# --------------------------- Optional Dependencies ----------------------------
# These imports are optional. If they fail, the script continues and simply
# omits the corresponding metrics. This keeps the demo portable.
try:
    import psutil     # Process/system metrics (CPU/RAM)
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    import pynvml     # NVIDIA's NVML for GPU metrics
    pynvml.nvmlInit()  # Initialize NVML early; if this fails, we'll mark unavailable
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False


# ======================= Configuration (edit these) =======================

# Base URL for Ollama "generate" API. You can override with env var OLLAMA_API_URL,
# e.g., "http://<remote-host>:11434/api/generate" if tunneling or remote usage.
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

# Port number we expect Ollama to be listening on locally; used for PID discovery.
# Change if you configured Ollama on a non-standard port.
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

# The question we want to ask the model. Keep it short if you want to minimize prompt tokens.
prompt = "What is the capital of France?"

# Model tag to use. Must be available locally (`ollama pull llama3.1`).
# You can specify quantization/size variants, e.g., "llama3.1:8b-instruct".
MODEL_NAME = "llama3.1"

# System message to shape the assistant behavior in the *answer* call.
assistant_system = "You are a helpful, accurate assistant. Answer concisely."

# System message for the evaluator behavior in the *evaluation* call.
evaluator_system = (
    "You are an evaluator. Given the user question and the AI answer, "
    "rate the response on the following: factual accuracy, relevance, completeness, "
    "coherence, helpfulness, harmfulness."
)

# Instructions to force strict JSON output from the evaluator.
# NOTE: Models may still deviate; we keep a fallback to print raw text.
evaluation_instructions = (
    "Evaluate the AI answer against the user question.\n"
    "Return ONLY a strict JSON object with EXACTLY these keys:\n"
    "factual_accuracy, relevance, completeness, coherence, helpfulness, harmfulness, comment\n\n"
    "Scoring:\n"
    "- Use integers from 0-10 for all six scores (0=worst, 10=best).\n"
    "- 'comment' should be one short sentence.\n"
    "No extra text. No markdown. JSON only."
)


# ======================= Small helper utilities =======================

def ns_to_s(ns: int) -> float:
    """
    Convert nanoseconds (as returned by Ollama) to seconds (float).
    Ollama timing fields are reported in nanoseconds; humans prefer seconds.

    Args:
        ns: Integer number of nanoseconds (can be 0 or missing)
    Returns:
        float seconds (0.0 if ns is falsy)
    """
    return ns / 1_000_000_000 if ns else 0.0


def fmt(x, digits=3):
    """
    Format numeric values with a fixed number of decimal places.

    Args:
        x: Any numeric-like value
        digits: Number of decimals to show
    Returns:
        str formatted with given precision, or str(x) if conversion fails
    """
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def bytes_to_gb(b: int) -> float:
    """
    Convert bytes to gigabytes for readability.

    Args:
        b: integer number of bytes
    Returns:
        float GB value
    """
    return b / (1024**3)


def find_ollama_pids(port: int = 11434):
    """
    Best-effort discovery of local Ollama server PIDs by:
      1) Looking for processes named 'ollama' (or with 'ollama' in cmdline)
      2) Checking if any of those are listening on the target port

    Returns:
        List[int]: Sorted list of PIDs. May be empty if psutil is missing,
                   insufficient permissions, or Ollama is not discoverable.
    """
    if not HAS_PSUTIL:
        return []

    pids = set()

    # Pass 1: Prefer processes named 'ollama' that are actually listening on our port.
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmd:
                # connections(kind="inet") may require elevated perms on some OSes.
                try:
                    for conn in proc.connections(kind="inet"):
                        if conn.laddr and conn.laddr.port == port:
                            pids.add(proc.pid)
                            break
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    # If we can't inspect connections, we'll try a broader fallback
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Pass 2: If Pass 1 found nothing, accept any 'ollama' process (less precise).
    if not pids:
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if "ollama" in (proc.info.get("name") or "").lower():
                    pids.add(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    return sorted(pids)


class ResourceMonitor:
    """
    Background sampler that periodically records:
      - System-wide CPU% and RAM usage
      - Per-process (Ollama PIDs) CPU% and RSS (resident memory)
      - Overall GPU utilization (avg across GPUs)
      - Per-process GPU memory used by Ollama PIDs (requires NVML)

    OPERATION
    ---------
    • start(): spawns a daemon thread that samples every `interval` seconds.
    • stop():  signals thread to stop and joins it.
    • summarize(): returns averages/peaks for all collected series.

    DEPENDENCIES
    ------------
    • psutil: required for CPU/RAM (system + per-process)
    • pynvml: required for GPU metrics (overall + per-process GPU memory)
    • If NVML is unavailable but nvidia-smi exists, we query overall GPU util via CLI.
    """
    def __init__(self, pids=None, interval: float = 0.1):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.pids = pids or []

        # System-wide samples
        self.sys_cpu = []       # [%] instantaneous CPU utilization
        self.sys_ram_used = []  # [bytes] used RAM
        self.sys_ram_pct = []   # [%] RAM utilization

        # Per-process (summed across discovered Ollama PIDs)
        self.proc_cpu = []      # [%] sum CPU across target PIDs
        self.proc_rss = []      # [bytes] sum RSS across target PIDs

        # GPU overall and per-process
        self.gpu_util = []      # [%] average across GPUs per sample
        self.gpu_proc_mem = []  # [bytes] per-process GPU memory used by target PIDs

        # Prime psutil CPU meters so first cpu_percent calls are meaningful.
        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None)
            for pid in self.pids:
                try:
                    psutil.Process(pid).cpu_percent(interval=None)
                except Exception:
                    pass

    def start(self):
        """Kick off the background sampler thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the sampler to stop and wait (briefly) for the thread to exit."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _sample_gpu_overall(self):
        """
        Sample overall GPU utilization:
          • Preferred: NVML (pynvml) — average of per-device utilization
          • Fallback: `nvidia-smi` CLI (if present) — coarse but often sufficient
        Returns:
            float | None : utilization percentage or None if unavailable
        """
        util = None
        if HAS_PYNVML:
            try:
                count = pynvml.nvmlDeviceGetCount()
                utils = []
                for i in range(count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    utils.append(float(u))
                if utils:
                    util = sum(utils) / len(utils)
            except Exception:
                pass
        else:
            # Fallback to CLI (works when NVIDIA driver & tools are installed)
            if shutil.which("nvidia-smi"):
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        stderr=subprocess.DEVNULL, timeout=1.0
                    ).decode().strip()
                    if out:
                        vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
                        if vals:
                            util = sum(vals) / len(vals)
                except Exception:
                    pass
        return util

    def _sample_gpu_proc_mem(self, target_pids):
        """
        Sample per-process GPU memory (bytes) used by *target_pids* across devices.
        Requires NVML. If any errors occur or pids empty, returns None.

        NVML quirks:
          • On some drivers, per-process memory may report NVML_VALUE_NOT_AVAILABLE.
          • We try both v2 and legacy getters and both compute/graphics listings.
        """
        if not HAS_PYNVML or not target_pids:
            return None
        total = 0
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                for getter in (
                    getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
                    getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses_v2", None),
                    getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
                    getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
                ):
                    if not getter:
                        continue
                    try:
                        procs = getter(h)
                        for pr in procs or []:
                            pid = getattr(pr, "pid", None)
                            used = getattr(pr, "usedGpuMemory", None)
                            if pid in target_pids and used not in (None, getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)):
                                total += int(used)
                    except Exception:
                        # Ignore device-specific errors and continue scanning
                        continue
        except Exception:
            return None
        return total

    def _run(self):
        """Sampler loop: records all metrics every `interval` seconds until stopped."""
        while not self._stop.is_set():
            # ---- System-wide CPU/RAM ----
            if HAS_PSUTIL:
                self.sys_cpu.append(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                self.sys_ram_used.append(vm.used)
                self.sys_ram_pct.append(vm.percent)

                # ---- Per-process (Ollama) CPU and RSS ----
                agg_cpu = 0.0
                agg_rss = 0
                for pid in list(self.pids):
                    try:
                        p = psutil.Process(pid)
                        agg_cpu += p.cpu_percent(interval=None)  # includes all cores normalized
                        mem = p.memory_info()
                        agg_rss += mem.rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                if self.pids:
                    self.proc_cpu.append(agg_cpu)
                    self.proc_rss.append(agg_rss)

            # ---- GPU overall utilization ----
            util = self._sample_gpu_overall()
            if util is not None:
                self.gpu_util.append(util)

            # ---- Per-process GPU memory (Ollama PIDs) ----
            gpumem = self._sample_gpu_proc_mem(self.pids)
            if gpumem is not None:
                self.gpu_proc_mem.append(gpumem)

            # Wait before next sample to control overhead
            time.sleep(self.interval)

    def summarize(self):
        """
        Compute summary statistics (averages and peaks) for all collected metrics.

        Returns:
            dict[str, float]: keys include:
              - system_cpu_avg_pct / system_cpu_peak_pct
              - system_ram_avg_gb / system_ram_peak_gb / system_ram_avg_pct / system_ram_peak_pct
              - ollama_cpu_avg_pct / ollama_cpu_peak_pct
              - ollama_ram_avg_gb / ollama_ram_peak_gb
              - gpu_overall_util_avg_pct / gpu_overall_util_peak_pct
              - ollama_gpu_mem_avg_gb / ollama_gpu_mem_peak_gb
            Only includes keys for series that were actually collected.
        """
        out = {}
        if self.sys_cpu:
            out["system_cpu_avg_pct"] = sum(self.sys_cpu)/len(self.sys_cpu)
            out["system_cpu_peak_pct"] = max(self.sys_cpu)
        if self.sys_ram_used:
            out["system_ram_avg_gb"] = bytes_to_gb(sum(self.sys_ram_used)/len(self.sys_ram_used))
            out["system_ram_peak_gb"] = bytes_to_gb(max(self.sys_ram_used))
            out["system_ram_avg_pct"] = sum(self.sys_ram_pct)/len(self.sys_ram_pct)
            out["system_ram_peak_pct"] = max(self.sys_ram_pct)

        if self.proc_cpu:
            out["ollama_cpu_avg_pct"] = sum(self.proc_cpu)/len(self.proc_cpu)
            out["ollama_cpu_peak_pct"] = max(self.proc_cpu)
        if self.proc_rss:
            out["ollama_ram_avg_gb"] = bytes_to_gb(sum(self.proc_rss)/len(self.proc_rss))
            out["ollama_ram_peak_gb"] = bytes_to_gb(max(self.proc_rss))

        if self.gpu_util:
            out["gpu_overall_util_avg_pct"] = sum(self.gpu_util)/len(self.gpu_util)
            out["gpu_overall_util_peak_pct"] = max(self.gpu_util)
        if self.gpu_proc_mem:
            out["ollama_gpu_mem_avg_gb"] = bytes_to_gb(sum(self.gpu_proc_mem)/len(self.gpu_proc_mem))
            out["ollama_gpu_mem_peak_gb"] = bytes_to_gb(max(self.gpu_proc_mem))

        return out


# ======================= 1) STREAMING ANSWER (true TTFT) =======================

# Build the payload for the answer request.
# NOTE: "stream": True is essential for *true* TTFT. Non-streaming would only
#       tell you end-to-end time, not the time until first token arrives.
answer_payload = {
    "model": MODEL_NAME,        # local model to use (must be pulled/available)
    "prompt": prompt,           # user's question
    "system": assistant_system, # behavior control for the assistant
    "stream": True              # stream JSONL chunks back to the client
}

# Attempt to discover Ollama server PIDs so we can attribute per-process usage.
# If not found (permissions or naming differences), metrics fall back to system-wide only.
ollama_pids = find_ollama_pids(port=OLLAMA_PORT)

# Start monitoring *before* we fire the HTTP request so the whole window is included.
monitor = ResourceMonitor(pids=ollama_pids, interval=0.1)
monitor.start()

# Record POST timestamp to anchor TTFT measurement.
post_ts = time.time()

# Issue the streaming POST request to /api/generate.
# `stream=True` lets us iterate over the response as the server emits JSONL lines.
with requests.post(OLLAMA_API_URL, json=answer_payload, stream=True) as resp:
    if resp.status_code != 200:
        monitor.stop()
        print("Error (answer):", resp.status_code, resp.text)
        raise SystemExit(1)

    # Accumulate streamed text chunks here and join at the end.
    response_chunks = []

    # Timestamp when we receive the *first* tokenized text (first "response" field).
    first_token_ts = None

    # The final JSON line (with "done": true) contains timing and token counts.
    load_duration_ns = 0
    prompt_eval_duration_ns = 0
    eval_duration_ns = 0
    total_duration_ns = 0
    prompt_eval_count = 0   # number of prompt tokens the model processed
    eval_count = 0          # number of tokens generated in the response

    # Iterate line-by-line (each line is a standalone JSON object).
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            # Some servers send keep-alive newlines; ignore empty lines.
            continue

        # Parse each JSONL chunk. If one chunk is malformed, skip (should be rare).
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Non-final chunks typically have a small "response" string (piece of text).
        piece = data.get("response", "")
        if piece and first_token_ts is None:
            # First time we see generated text => mark TTFT arrival moment.
            first_token_ts = time.time()
        if piece:
            response_chunks.append(piece)

        # The terminal line includes done=true and timing/token counters.
        if data.get("done"):
            load_duration_ns         = int(data.get("load_duration", 0))          # model load (ns)
            prompt_eval_duration_ns  = int(data.get("prompt_eval_duration", 0))   # prompt eval (ns)
            eval_duration_ns         = int(data.get("eval_duration", 0))          # generation (ns)
            total_duration_ns        = int(data.get("total_duration", 0))         # full server-side (ns)
            prompt_eval_count        = int(data.get("prompt_eval_count", 0))      # #prompt tokens
            eval_count               = int(data.get("eval_count", 0))             # #output tokens
            break  # Exit the stream loop once terminal object received

# Record a wall-clock timestamp at the very end of the stream for total wall time.
wall_end = time.time()

# Stop the resource sampler; the answer call is complete.
monitor.stop()

# Combine all text fragments into the final answer (strip trailing whitespace).
ai_answer = "".join(response_chunks).strip()

# ------------------ Throughput & Latency Calculations ------------------

# Convert nanos to seconds for readability.
prompt_eval_s = ns_to_s(prompt_eval_duration_ns)
response_eval_s = ns_to_s(eval_duration_ns)
total_s = ns_to_s(total_duration_ns)

# Compute throughputs defensively (avoid divide-by-zero).
prompt_tps = (prompt_eval_count / prompt_eval_s) if prompt_eval_s > 0 else 0.0
response_tps = (eval_count / response_eval_s) if response_eval_s > 0 else 0.0
total_tps = ((prompt_eval_count + eval_count) / total_s) if total_s > 0 else 0.0

# True TTFT: network + server + queuing up to the first streamed token you see.
ttft_s = (first_token_ts - post_ts) if first_token_ts else None

# Wall time as observed by the client (from POST to end-of-stream).
wall_time_s = wall_end - post_ts

# Model compute time as reported by the server (usually ~ total_s).
model_compute_s = total_s


# ------------------ Present Answer and Metrics ------------------

print("\n=== ANSWER ===")
print(ai_answer)

print("\n=== METRICS (Answer call) ===")
print(f"Prompt tokens: {prompt_eval_count}")
print(f"Response tokens: {eval_count}")
print(f"Prompt throughput (t/s): {fmt(prompt_tps)}")
print(f"Response throughput (t/s): {fmt(response_tps)}")
print(f"Total throughput (t/s): {fmt(total_tps)}")
print(f"True TTFT (s): {fmt(ttft_s) if ttft_s is not None else 'N/A'}")
print(f"Wall time full answer (s): {fmt(wall_time_s)}")
print(f"Model compute time (s) reported by server: {fmt(model_compute_s)}")

# Print summarized resource usage over the streaming window.
sys_metrics = monitor.summarize()
if sys_metrics:
    print("\n--- System-wide & Ollama per-process resource usage (during answer) ---")
    # System CPU%
    if "system_cpu_avg_pct" in sys_metrics:
        print(f"System CPU avg %:  {fmt(sys_metrics['system_cpu_avg_pct'])}")
        print(f"System CPU peak %: {fmt(sys_metrics['system_cpu_peak_pct'])}")

    # System RAM (GB and %)
    if "system_ram_avg_gb" in sys_metrics:
        print(f"System RAM avg used:  {fmt(sys_metrics['system_ram_avg_gb'])} GB")
        print(f"System RAM peak used: {fmt(sys_metrics['system_ram_peak_gb'])} GB")
    if "system_ram_avg_pct" in sys_metrics:
        print(f"System RAM avg %:     {fmt(sys_metrics['system_ram_avg_pct'])}")
        print(f"System RAM peak %:    {fmt(sys_metrics['system_ram_peak_pct'])}")

    # Ollama per-process CPU/RAM (sum across PIDs)
    if "ollama_cpu_avg_pct" in sys_metrics:
        print(f"Ollama CPU avg %:     {fmt(sys_metrics['ollama_cpu_avg_pct'])}")
        print(f"Ollama CPU peak %:    {fmt(sys_metrics['ollama_cpu_peak_pct'])}")
    if "ollama_ram_avg_gb" in sys_metrics:
        print(f"Ollama RAM avg used:  {fmt(sys_metrics['ollama_ram_avg_gb'])} GB")
        print(f"Ollama RAM peak used: {fmt(sys_metrics['ollama_ram_peak_gb'])} GB")

    # GPU overall and Ollama per-process GPU memory
    if "gpu_overall_util_avg_pct" in sys_metrics:
        print(f"GPU overall util avg %:  {fmt(sys_metrics['gpu_overall_util_avg_pct'])}")
        print(f"GPU overall util peak %: {fmt(sys_metrics['gpu_overall_util_peak_pct'])}")
    if "ollama_gpu_mem_avg_gb" in sys_metrics:
        print(f"Ollama GPU mem avg:      {fmt(sys_metrics['ollama_gpu_mem_avg_gb'])} GB")
        print(f"Ollama GPU mem peak:     {fmt(sys_metrics['ollama_gpu_mem_peak_gb'])} GB")


# ======================= 2) Non-streaming evaluator call =======================
# Next, we ask the model (or a different one) to evaluate the answer against
# the original question and return STRICT JSON (six 0–10 scores + one comment).

evaluation_prompt = f"""USER QUESTION:
{prompt}

AI ANSWER:
{ai_answer}

{evaluation_instructions}
"""

# Non-streaming is simpler for evaluation since we just need a single JSON blob.
eval_payload = {
    "model": MODEL_NAME,         # can be same or a faster/smaller judge model
    "prompt": evaluation_prompt, # includes question, answer, and strict-JSON rules
    "system": evaluator_system,  # steers the model into evaluator role
    "stream": False
}

# Fire the evaluator request and handle basic HTTP errors.
eval_resp = requests.post(OLLAMA_API_URL, json=eval_payload)
if eval_resp.status_code != 200:
    print("\nError (evaluation):", eval_resp.status_code, eval_resp.text)
    raise SystemExit(1)

# Extract the model's string output (which *should* be a JSON object).
eval_text = (eval_resp.json().get("response") or "").strip()

print("\n=== EVALUATION ===")
try:
    # Attempt to parse strict JSON. If the model obeyed instructions, this succeeds.
    evaluation = json.loads(eval_text)
    print(json.dumps(evaluation, indent=2))
except json.JSONDecodeError:
    # Fallback: show raw text so you can inspect deviations and refine prompts.
    print("(non-JSON evaluator output)")
    print(eval_text)

