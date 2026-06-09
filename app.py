"""
Web UI for the Storyline → Video → Joint Angles Pipeline
---------------------------------------------------------
Run:  uvicorn app:app --reload   (or: python app.py)
Open: http://localhost:8000
"""

import asyncio
import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from pipeline import run_pipeline, OUTPUT_DIR
from rl_train import train_ppo_async, PPO_DEFAULTS

RL_OUTPUT_DIR = OUTPUT_DIR / "rl"

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Motion Pipeline")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# ─── SSE event queue per session (simple single-user approach) ────────────────
_event_queue: queue.Queue = queue.Queue()
_pipeline_state = {
    "running": False,
    "done":    False,
    "error":   None,
    "result":  None,
    "stages":  {i: {"status": "pending", "messages": []} for i in range(1, 6)},
}
_state_lock = threading.Lock()

STAGE_NAMES = {
    1: "Generate Video Prompt",
    2: "Generate Video (NVIDIA Cosmos)",
    3: "Extract Joint Angles",
    4: "Smooth & Filter Angles",
    5: "Build Stick Figure + Trajectory",
    6: "Behavior Cloning (RL Training)",
}


def _reset_state():
    global _pipeline_state
    with _state_lock:
        _pipeline_state = {
            "running": False,
            "done":    False,
            "error":   None,
            "result":  None,
            "stages":  {i: {"status": "pending", "messages": []} for i in range(1, 7)},
        }


def _progress_cb(stage: int, msg: str, data: dict):
    with _state_lock:
        stages = _pipeline_state["stages"]
        if stage not in stages:
            stages[stage] = {"status": "pending", "messages": []}
        st = stages[stage]
        if st["status"] == "pending":
            st["status"] = "running"
        # Mark previous stages done
        for s in list(stages.keys()):
            if s < stage and stages[s]["status"] == "running":
                stages[s]["status"] = "done"
        # Don't spam token-by-token messages into history
        if not data.get("token"):
            st["messages"].append(msg)

    event = {"stage": stage, "msg": msg, "data": data}
    _event_queue.put(event)


def _run_in_thread(storyline: str, existing_video: Optional[str]):
    try:
        with _state_lock:
            _pipeline_state["running"] = True
        result = run_pipeline(storyline, cb=_progress_cb, use_existing_video=existing_video)
        with _state_lock:
            _pipeline_state["running"] = False
            _pipeline_state["done"]    = True
            _pipeline_state["result"]  = result
            for s in range(1, 6):
                if _pipeline_state["stages"][s]["status"] == "running":
                    _pipeline_state["stages"][s]["status"] = "done"
        _event_queue.put({"stage": 0, "msg": "DONE", "data": result})
    except Exception as e:
        with _state_lock:
            _pipeline_state["running"] = False
            _pipeline_state["error"]   = str(e)
        _event_queue.put({"stage": 0, "msg": "ERROR", "data": {"error": str(e)}})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/run")
async def api_run(request: Request):
    body = await request.json()
    storyline = (body.get("storyline") or "").strip()
    existing_video = body.get("existing_video") or None

    if not storyline:
        return JSONResponse({"error": "storyline is required"}, status_code=400)

    with _state_lock:
        if _pipeline_state["running"]:
            return JSONResponse({"error": "Pipeline already running"}, status_code=409)

    _reset_state()
    # Drain leftover events
    while not _event_queue.empty():
        try:
            _event_queue.get_nowait()
        except queue.Empty:
            break

    t = threading.Thread(target=_run_in_thread, args=(storyline, existing_video), daemon=True)
    t.start()
    return JSONResponse({"status": "started"})


@app.get("/api/state")
async def api_state():
    with _state_lock:
        return JSONResponse(_pipeline_state)


@app.get("/api/stream")
async def api_stream(request: Request):
    """Server-Sent Events stream — emits pipeline progress in real time."""
    async def event_generator():
        # Send current state first
        with _state_lock:
            snap = json.dumps(_pipeline_state)
        yield f"data: {json.dumps({'type':'state','payload': json.loads(snap)})}\n\n"

        while True:
            if await request.is_disconnected():
                break
            try:
                ev = _event_queue.get(timeout=0.5)
            except queue.Empty:
                yield ": ping\n\n"
                continue

            yield f"data: {json.dumps({'type':'event','payload': ev})}\n\n"

            if ev.get("msg") in ("DONE", "ERROR"):
                break

    return StreamingResponse(event_generator(),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    dest = UPLOADS_DIR / f"upload_{int(time.time())}{suffix}"
    content = await file.read()
    dest.write_bytes(content)
    return JSONResponse({"path": str(dest), "filename": file.filename})


@app.get("/api/download/video")
async def download_video():
    p = OUTPUT_DIR / "stick_figure.mp4"
    if not p.exists():
        return JSONResponse({"error": "Not generated yet"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4", filename="stick_figure.mp4")


@app.get("/api/download/trajectory")
async def download_trajectory():
    p = OUTPUT_DIR / "trajectory.json"
    if not p.exists():
        return JSONResponse({"error": "Not generated yet"}, status_code=404)
    return FileResponse(str(p), media_type="application/json", filename="trajectory.json")


# ─── Frontend ─────────────────────────────────────────────────────────────────

_PLACEHOLDER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motion Pipeline</title>
<style>
  :root {
    --bg: #0e0e14; --surface: #16161f; --border: #2a2a38;
    --accent: #5b8cff; --accent2: #a78bfa;
    --text: #e0e0f0; --muted: #6b6b8a; --success: #4ade80;
    --warn: #facc15; --error: #f87171; --running: #60a5fa;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
         min-height: 100vh; display: flex; flex-direction: column; }
  header { padding: 20px 32px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 14px; }
  header .logo { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px;
                 background: linear-gradient(135deg, var(--accent), var(--accent2));
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header .sub { color: var(--muted); font-size: 0.85rem; }
  main { flex: 1; display: grid; grid-template-columns: 380px 1fr; gap: 0; }

  /* ── Left panel ── */
  .left-panel { border-right: 1px solid var(--border); padding: 28px 24px;
                display: flex; flex-direction: column; gap: 20px; overflow-y: auto; }
  label { font-size: 0.78rem; font-weight: 600; color: var(--muted); text-transform: uppercase;
          letter-spacing: 0.06em; display: block; margin-bottom: 6px; }
  textarea { width: 100%; min-height: 110px; background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; color: var(--text); font-size: 0.92rem; padding: 12px;
             resize: vertical; outline: none; transition: border-color .2s; font-family: inherit; }
  textarea:focus { border-color: var(--accent); }
  input[type=text] { width: 100%; background: var(--surface); border: 1px solid var(--border);
                     border-radius: 8px; color: var(--text); font-size: 0.88rem; padding: 10px 12px;
                     outline: none; transition: border-color .2s; }
  input[type=text]:focus { border-color: var(--accent); }
  .hint { font-size: 0.75rem; color: var(--muted); margin-top: 4px; }
  .btn { width: 100%; padding: 13px; border-radius: 8px; border: none; cursor: pointer;
         font-size: 0.95rem; font-weight: 600; transition: all .2s; }
  .btn-primary { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; }
  .btn-primary:hover:not(:disabled) { opacity: .88; transform: translateY(-1px); }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; transform: none; }

  /* ── Stage tracker ── */
  .stages { display: flex; flex-direction: column; gap: 10px; }
  .stage { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
           padding: 14px 16px; transition: border-color .3s, background .3s; }
  .stage.running { border-color: var(--running); background: #1a2233; }
  .stage.done    { border-color: var(--success); }
  .stage.error   { border-color: var(--error); }
  .stage-header  { display: flex; align-items: center; gap: 10px; }
  .stage-icon    { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
                   font-size: 0.75rem; font-weight: 700; flex-shrink: 0; }
  .pending  .stage-icon { background: var(--border); color: var(--muted); }
  .running  .stage-icon { background: var(--running); color: #fff; animation: pulse 1.2s infinite; }
  .done     .stage-icon { background: var(--success); color: #0e0e14; }
  .error    .stage-icon { background: var(--error); color: #fff; }
  @keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(96,165,250,.5)} 50%{box-shadow:0 0 0 6px rgba(96,165,250,0)} }
  .stage-name { font-size: 0.88rem; font-weight: 600; flex: 1; }
  .stage-badge { font-size: 0.68rem; padding: 2px 8px; border-radius: 20px; font-weight: 600;
                 text-transform: uppercase; letter-spacing: .04em; }
  .pending .stage-badge  { background: var(--border); color: var(--muted); }
  .running .stage-badge  { background: rgba(96,165,250,.2); color: var(--running); }
  .done    .stage-badge  { background: rgba(74,222,128,.15); color: var(--success); }
  .error   .stage-badge  { background: rgba(248,113,113,.15); color: var(--error); }
  .stage-msgs { margin-top: 8px; font-size: 0.76rem; color: var(--muted);
                max-height: 80px; overflow-y: auto; display: flex; flex-direction: column; gap: 3px; }
  .stage-msgs span { line-height: 1.4; }

  /* ── Right panel ── */
  .right-panel { padding: 28px 32px; display: flex; flex-direction: column; gap: 24px;
                 overflow-y: auto; }
  .section-title { font-size: 0.78rem; font-weight: 700; color: var(--muted);
                   text-transform: uppercase; letter-spacing: .07em; margin-bottom: 12px; }

  /* Prompt preview */
  .prompt-box { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
                padding: 16px; font-size: 0.85rem; line-height: 1.6; color: var(--text);
                min-height: 80px; white-space: pre-wrap; word-break: break-word; }
  .prompt-box.streaming::after { content: '▋'; animation: blink .7s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  /* Angle chart area */
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px,1fr)); gap: 12px; }
  .angle-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
                padding: 14px 16px; }
  .angle-card .jname { font-size: 0.78rem; font-weight: 700; color: var(--accent2); margin-bottom: 8px; }
  .angle-row { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted);
               margin-bottom: 3px; }
  .angle-row span:last-child { color: var(--text); font-weight: 600; }

  /* Downloads */
  .download-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .dl-btn { padding: 10px 20px; border-radius: 8px; border: 1px solid var(--accent);
            background: transparent; color: var(--accent); font-size: 0.85rem; font-weight: 600;
            cursor: pointer; transition: all .2s; text-decoration: none; display: inline-flex;
            align-items: center; gap: 6px; }
  .dl-btn:hover { background: rgba(91,140,255,.1); }
  .dl-btn:disabled, .dl-btn.disabled { opacity: .35; pointer-events: none; }

  /* Video embed */
  video { width: 100%; max-width: 480px; border-radius: 12px;
          border: 1px solid var(--border); background: #000; }

  /* Log */
  .log { background: #0a0a10; border: 1px solid var(--border); border-radius: 10px;
         padding: 14px; font-family: 'Consolas', monospace; font-size: 0.75rem;
         color: #aaa; max-height: 200px; overflow-y: auto; white-space: pre-wrap;
         word-break: break-all; line-height: 1.5; }

  .empty-state { color: var(--muted); font-size: 0.88rem; text-align: center;
                 padding: 40px 0; opacity: .6; }
  .error-banner { background: rgba(248,113,113,.1); border: 1px solid var(--error);
                  border-radius: 8px; padding: 12px 16px; color: var(--error); font-size: 0.85rem; }
  .success-banner { background: rgba(74,222,128,.08); border: 1px solid var(--success);
                    border-radius: 8px; padding: 12px 16px; color: var(--success); font-size: 0.85rem; }
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">&#9889; Motion Pipeline</div>
    <div class="sub">Storyline → Video → Joint Angles → Trajectory</div>
  </div>
</header>
<main>

<!-- Left panel: Input + Stage tracker -->
<div class="left-panel">
  <div>
    <label for="storyline">Storyline</label>
    <textarea id="storyline" placeholder="e.g. A humanoid robot walks into a factory, waves hello, and lifts a box onto a shelf."></textarea>
  </div>
  <div>
    <label for="existing-video">Use existing video (optional)</label>
    <input type="text" id="existing-video" placeholder="Path to .mp4 — skips Cosmos generation">
    <div class="hint">Leave blank to generate via NVIDIA Cosmos.</div>
  </div>
  <button class="btn btn-primary" id="run-btn" onclick="startPipeline()">&#9654; Run Pipeline</button>

  <div class="stages" id="stages-container"></div>
</div>

<!-- Right panel: Live output -->
<div class="right-panel">

  <div id="banner-area"></div>

  <div>
    <div class="section-title">&#128196; Generated Video Prompt</div>
    <div class="prompt-box" id="prompt-box"><span class="empty-state">Prompt will appear here…</span></div>
  </div>

  <div id="video-section" style="display:none">
    <div class="section-title">&#127909; Stick Figure Video</div>
    <video id="stick-video" controls></video>
  </div>

  <div id="angles-section" style="display:none">
    <div class="section-title">&#128260; Joint Angle Statistics</div>
    <div class="chart-grid" id="angle-grid"></div>
  </div>

  <div>
    <div class="section-title">&#11015; Downloads</div>
    <div class="download-row">
      <a id="dl-video" class="dl-btn disabled" href="/api/download/video" download>
        &#127909; Stick Video (.mp4)
      </a>
      <a id="dl-traj" class="dl-btn disabled" href="/api/download/trajectory" download>
        &#128196; Trajectory (.json)
      </a>
    </div>
  </div>

  <div>
    <div class="section-title">&#128220; Live Log</div>
    <div class="log" id="log">Waiting for pipeline to start…\n</div>
  </div>

</div>
</main>

<script>
const STAGE_NAMES = {
  1: "Generate Video Prompt",
  2: "Generate Video (NVIDIA Cosmos)",
  3: "Extract Joint Angles",
  4: "Smooth & Filter Angles",
  5: "Build Stick Figure + Trajectory",
};

let evtSource = null;
let promptStreaming = false;
let promptBuffer = "";

function renderStages(stages) {
  const container = document.getElementById("stages-container");
  container.innerHTML = "";
  for (let s = 1; s <= 5; s++) {
    const info = stages[s] || { status: "pending", messages: [] };
    const st = info.status;
    const icon = st === "done" ? "✓" : st === "error" ? "✕" : s;
    const badge = st.charAt(0).toUpperCase() + st.slice(1);
    const msgs = (info.messages || []).slice(-3).map(function(m) {
      return "<span>" + escHtml(m) + "</span>";
    }).join("");
    const msgsHtml = msgs ? '<div class="stage-msgs">' + msgs + "</div>" : "";
    container.innerHTML +=
      '<div class="stage ' + st + '">' +
        '<div class="stage-header">' +
          '<div class="stage-icon">' + icon + "</div>" +
          '<div class="stage-name">' + STAGE_NAMES[s] + "</div>" +
          '<div class="stage-badge">' + badge + "</div>" +
        "</div>" +
        msgsHtml +
      "</div>";
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function appendLog(text) {
  const el = document.getElementById("log");
  el.textContent += text + "\n";
  el.scrollTop = el.scrollHeight;
}

function setBanner(type, msg) {
  const el = document.getElementById("banner-area");
  if (!type) { el.innerHTML = ""; return; }
  el.innerHTML = '<div class="' + type + '-banner">' + escHtml(msg) + "</div>";
}

function handleEvent(ev) {
  const { stage, msg, data } = ev;

  // Token streaming for prompt
  if (stage === 1 && data && data.token) {
    const box = document.getElementById("prompt-box");
    if (!promptStreaming) {
      promptBuffer = "";
      promptStreaming = true;
      box.innerHTML = "";
      box.classList.add("streaming");
    }
    promptBuffer += data.token;
    box.textContent = promptBuffer;
    return;
  }

  if (stage === 1 && msg === "Video prompt ready." && data && data.prompt) {
    promptStreaming = false;
    const box = document.getElementById("prompt-box");
    box.classList.remove("streaming");
    box.textContent = data.prompt;
    promptBuffer = data.prompt;
  }

  if (msg === "DONE") {
    setBanner("success", "✓ Pipeline complete! All stages finished.");
    document.getElementById("run-btn").disabled = false;
    showResults(data);
    return;
  }

  if (msg === "ERROR") {
    setBanner("error", "✕ Pipeline error: " + (data.error || "unknown"));
    document.getElementById("run-btn").disabled = false;
    return;
  }

  if (!data || !data.token) {
    appendLog("[Stage " + stage + "] " + msg);
  }
}

function showResults(result) {
  // Angle stats
  const stats = result.angle_stats || {};
  const grid = document.getElementById("angle-grid");
  grid.innerHTML = "";
  for (const [jname, s] of Object.entries(stats)) {
    grid.innerHTML +=
      '<div class="angle-card">' +
        '<div class="jname">' + escHtml(jname) + "</div>" +
        '<div class="angle-row"><span>Mean</span><span>' + s.mean.toFixed(1) + "\xB0</span></div>" +
        '<div class="angle-row"><span>Std</span><span>'  + s.std.toFixed(1)  + "\xB0</span></div>" +
        '<div class="angle-row"><span>Min</span><span>'  + s.min.toFixed(1)  + "\xB0</span></div>" +
        '<div class="angle-row"><span>Max</span><span>'  + s.max.toFixed(1)  + "\xB0</span></div>" +
      "</div>";
  }
  if (Object.keys(stats).length > 0) {
    document.getElementById("angles-section").style.display = "block";
  }

  // Video
  const vs = document.getElementById("video-section");
  const vid = document.getElementById("stick-video");
  vs.style.display = "block";
  vid.src = "/output/stick_figure.mp4?t=" + Date.now();
  vid.load();

  // Downloads
  document.getElementById("dl-video").classList.remove("disabled");
  document.getElementById("dl-traj").classList.remove("disabled");
}

async function startPipeline() {
  const storyline = document.getElementById("storyline").value.trim();
  if (!storyline) { alert("Please enter a storyline."); return; }

  const existingVideo = document.getElementById("existing-video").value.trim();

  // Reset UI
  setBanner(null);
  document.getElementById("run-btn").disabled = true;
  document.getElementById("log").textContent = "";
  document.getElementById("prompt-box").innerHTML = '<span class="empty-state">Generating…</span>';
  document.getElementById("video-section").style.display = "none";
  document.getElementById("angles-section").style.display = "none";
  document.getElementById("dl-video").classList.add("disabled");
  document.getElementById("dl-traj").classList.add("disabled");
  promptStreaming = false;
  promptBuffer = "";

  renderStages({ 1:{status:"pending",messages:[]}, 2:{status:"pending",messages:[]},
                 3:{status:"pending",messages:[]}, 4:{status:"pending",messages:[]},
                 5:{status:"pending",messages:[]} });

  // Close existing SSE
  if (evtSource) { evtSource.close(); evtSource = null; }

  // Start pipeline
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ storyline, existing_video: existingVideo || null }),
  });
  const rdata = await resp.json();
  if (rdata.error) {
    setBanner("error", rdata.error);
    document.getElementById("run-btn").disabled = false;
    return;
  }

  appendLog("Pipeline started…");

  // Open SSE stream
  evtSource = new EventSource("/api/stream");
  evtSource.onmessage = (e) => {
    const payload = JSON.parse(e.data);
    if (payload.type === "state") {
      renderStages(payload.payload.stages);
    } else if (payload.type === "event") {
      handleEvent(payload.payload);
      // Re-fetch state to keep stage tracker fresh
      fetch("/api/state").then(r => r.json()).then(s => renderStages(s.stages));
    }
  };
  evtSource.onerror = () => {
    appendLog("[SSE] Connection lost.");
    if (evtSource) { evtSource.close(); evtSource = null; }
  };
}
</script>
</body>
</html>
"""


_rl_queue: Optional[queue.Queue] = None
_rl_thread: Optional[threading.Thread] = None


def _rl_queue_to_sse_bridge():
    """Drain the RL event queue into the main SSE queue."""
    global _rl_queue
    while _rl_queue is not None:
        try:
            ev = _rl_queue.get(timeout=0.5)
            # Update pipeline state for stage 6
            with _state_lock:
                stages = _pipeline_state["stages"]
                if 6 not in stages:
                    stages[6] = {"status": "running", "messages": []}
                st = stages[6]
                if st["status"] == "pending":
                    st["status"] = "running"
                if ev["msg"] not in ("DONE", "ERROR") and not ev["data"].get("reward_history"):
                    st["messages"].append(ev["msg"])
                    st["messages"] = st["messages"][-5:]
                if ev["msg"] == "DONE" or ev["data"].get("type") == "done":
                    st["status"] = "done"
                    _pipeline_state["running"] = False
                elif ev["msg"] == "ERROR":
                    st["status"] = "error"
                    _pipeline_state["running"] = False

            _event_queue.put(ev)

            if ev["msg"] in ("ERROR",) or ev["data"].get("type") == "done":
                break
        except queue.Empty:
            # Check if thread finished
            if _rl_thread is not None and not _rl_thread.is_alive():
                break


@app.post("/api/train")
async def api_train(request: Request):
    """Start PPO training on the G1 motion tracking environment."""
    global _rl_queue, _rl_thread

    traj_path = OUTPUT_DIR / "trajectory.json"
    if not traj_path.exists():
        return JSONResponse(
            {"error": "No trajectory.json found — run the pipeline first."},
            status_code=400,
        )

    with _state_lock:
        if _pipeline_state["running"]:
            return JSONResponse({"error": "A job is already running."}, status_code=409)
        _pipeline_state["running"] = True
        _pipeline_state["stages"][6] = {"status": "pending", "messages": []}

    try:
        body = await request.json()
    except Exception:
        body = {}

    cfg = {
        "total_timesteps": int(body.get("total_timesteps", PPO_DEFAULTS["total_timesteps"])),
        "n_envs":          int(body.get("n_envs",          PPO_DEFAULTS["n_envs"])),
        "learning_rate":   float(body.get("lr",            PPO_DEFAULTS["learning_rate"])),
        "n_steps":         int(body.get("n_steps",         PPO_DEFAULTS["n_steps"])),
        "batch_size":      int(body.get("batch_size",      PPO_DEFAULTS["batch_size"])),
        "device":          body.get("device",              PPO_DEFAULTS["device"]),
    }

    # Drain leftover events
    while not _event_queue.empty():
        try:
            _event_queue.get_nowait()
        except queue.Empty:
            break

    RL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _rl_queue, _rl_thread = train_ppo_async(traj_path, RL_OUTPUT_DIR, cfg=cfg)

    # Bridge thread: forwards RL events → SSE queue
    bridge = threading.Thread(target=_rl_queue_to_sse_bridge, daemon=True)
    bridge.start()

    return JSONResponse({"status": "training_started", "cfg": cfg})


@app.get("/api/train/status")
async def api_train_status():
    with _state_lock:
        return JSONResponse({
            "running": _pipeline_state["running"],
            "stage6":  _pipeline_state["stages"].get(6, {"status": "pending"}),
        })


@app.get("/api/download/policy")
async def download_policy():
    for name in ["ppo_final.zip", "policy_best.pt"]:
        p = RL_OUTPUT_DIR / name
        if p.exists():
            return FileResponse(str(p), filename=p.name)
    return JSONResponse({"error": "No policy trained yet."}, status_code=404)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse({}, status_code=204)


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"), media_type="text/html")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
