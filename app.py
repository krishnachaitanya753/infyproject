"""
Web UI for the Storyline → Video → Joint Angles Pipeline
---------------------------------------------------------
Run:  uvicorn app:app --reload   (or: python app.py)
Open: http://localhost:8000
"""

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

STAGE_NAMES = {
    1: "Generate Video Prompt",
    2: "Generate Video",
    3: "Extract Joint Angles",
    4: "Smooth & Filter Angles",
    5: "Build Stick Figure + Trajectory",
    6: "PPO Motion Tracking",
}
PIPELINE_STAGES = 5                  # stages run by run_pipeline (stage 6 = RL, separate)
TOTAL_STAGES    = len(STAGE_NAMES)   # single source of truth for stage count


def _blank_stages():
    return {i: {"status": "pending", "messages": []} for i in range(1, TOTAL_STAGES + 1)}


def _fresh_state():
    return {"running": False, "done": False, "error": None,
            "result": None, "stages": _blank_stages()}


# ─── SSE event queue (single shared queue → single-user / single-tab only) ────
_event_queue: queue.Queue = queue.Queue()
_pipeline_state = _fresh_state()
_state_lock = threading.Lock()


def _reset_state():
    global _pipeline_state
    with _state_lock:
        _pipeline_state = _fresh_state()


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
            for s in range(1, PIPELINE_STAGES + 1):
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
    p = RL_OUTPUT_DIR / "ppo_final.zip"
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
