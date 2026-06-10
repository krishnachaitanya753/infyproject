"""
Storyline → Video → Joint Angles → Smoothed Trajectory Pipeline
-----------------------------------------------------------------
Stages:
  1. generate_prompt  : storyline → video-ready prompt (NVIDIA LLM)
  2. generate_video   : prompt    → mp4 (NVIDIA Cosmos / Wan video-gen)
  3. extract_angles   : video     → raw joint angles  (MediaPipe)
  4. smooth_angles    : raw       → cleaned angles (outlier + Savitzky-Golay)
  5. build_output     : render stick-figure video + save trajectory JSON

Each stage emits progress via an optional callback:
    progress_cb(stage: int, message: str, data: dict | None)
"""

import os
import cv2
import json
import time
import base64
import urllib.request
import numpy as np
import imageio.v2 as imageio
import requests as http_requests
from pathlib import Path
from typing import Callable, Optional
from openai import OpenAI
from dotenv import load_dotenv
from scipy.signal import savgol_filter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
FAL_API_KEY     = os.getenv("FAL_API_KEY", "")
RUNWAY_API_KEY  = os.getenv("RUNWAY_API_KEY", "")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")

# VIDEO_PROVIDER: "veo" | "hf" (free) | "fal" | "runway"
VIDEO_PROVIDER  = os.getenv("VIDEO_PROVIDER", "veo")

SCRIPT_MODEL    = "minimaxai/minimax-m2.7"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# fal.ai model options (pick one via FAL_MODEL env or default to kling)
FAL_MODEL       = os.getenv("FAL_MODEL", "fal-ai/kling-video/v1.6/pro/text-to-video")

# Runway Gen-3 Alpha Turbo
RUNWAY_MODEL    = "gen3a_turbo"

# G1 robot reference image for Veo (shown to the model so it knows the robot's appearance)
G1_REFERENCE_IMAGE = os.getenv("G1_REFERENCE_IMAGE",
                                str(Path(__file__).parent / "static" / "g1_reference.png"))

# HuggingFace Space options (free, no key needed):
#   "Lightricks/LTX-Video"   — fastest (~30s)
#   "THUDM/CogVideoX5B-Space" — best quality (~3-5 min)
#   "Wan-AI/Wan2.1-T2V-14B-Gradio" — great motion (~5 min)
HF_SPACE = os.getenv("HF_SPACE", "Lightricks/LTX-Video")

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = Path("pose_landmarker_heavy.task")

OUTPUT_DIR = Path("pipeline_output")

# Base prompt optimised for robot retargeting (from reference project):
# full body visible, tight clothing, static camera, realistic indoor motion
VEO_BASE_PROMPT = """Full-body shot of a single adult humanoid subject, with the entire body visible from head to feet at all times.

The subject is wearing tight-fitting motion capture style clothing: a short-sleeve shirt and slim athletic pants.
No coat, no jacket, no robe, no cloak, no skirt, no loose clothing, no accessories.

Static camera, eye-level, neutral perspective.
The subject remains fully inside the frame throughout the entire video.

The scene takes place in a realistic indoor room environment.
The room has clearly visible walls, floor, and corners.
The boundary between the floor and the walls is clearly visible.
The floor plane is clearly defined and fully visible.
The background is NOT a seamless white backdrop, NOT a studio cyclorama, and NOT an infinite background.

The room resembles a simple laboratory, motion analysis room, or empty interior space.
Surfaces are plain but spatially well-defined.

Even, neutral indoor lighting with no dramatic shadows or highlights.
No cinematic effects.

Motion is biomechanically accurate and physically realistic.
Natural human joint limits, correct center-of-mass movement, realistic balance, gravity, inertia, and ground contact.
No exaggerated motion, no stylized animation.

No camera movement, no cuts, no slow motion, no motion blur.
"""

SCRIPT_SYSTEM = """You are a video prompt engineer specialising in humanoid robot motion videos.
Given a one-line storyline, output ONLY a concise action sequence description (max 80 words) in this format:

Action sequence:
The subject <action step 1>.
<action step 2>.
...

Describe only the physical movements step by step. No scene details, no camera directions — those are handled separately.
Output ONLY the action sequence, no explanation."""

# ─── Skeleton / angle definitions (from stickfigure.py) ──────────────────────

LM = {
    "nose":0,"left_eye_inner":1,"left_eye":2,"left_eye_outer":3,
    "right_eye_inner":4,"right_eye":5,"right_eye_outer":6,
    "left_ear":7,"right_ear":8,"mouth_left":9,"mouth_right":10,
    "left_shoulder":11,"right_shoulder":12,
    "left_elbow":13,"right_elbow":14,
    "left_wrist":15,"right_wrist":16,
    "left_pinky":17,"right_pinky":18,
    "left_index":19,"right_index":20,
    "left_thumb":21,"right_thumb":22,
    "left_hip":23,"right_hip":24,
    "left_knee":25,"right_knee":26,
    "left_ankle":27,"right_ankle":28,
    "left_heel":29,"right_heel":30,
    "left_foot":31,"right_foot":32,
}

SEGMENTS = [
    ("nose","left_eye",(180,200,255),2),("nose","right_eye",(180,200,255),2),
    ("left_shoulder","right_shoulder",(120,220,120),3),
    ("left_shoulder","left_hip",(120,220,120),3),
    ("right_shoulder","right_hip",(120,220,120),3),
    ("left_hip","right_hip",(120,220,120),3),
    ("left_shoulder","left_elbow",(80,160,255),3),
    ("left_elbow","left_wrist",(60,120,220),2),
    ("left_wrist","left_index",(40,90,180),1),
    ("right_shoulder","right_elbow",(255,160,80),3),
    ("right_elbow","right_wrist",(220,120,60),2),
    ("right_wrist","right_index",(180,90,40),1),
    ("left_hip","left_knee",(80,220,220),3),
    ("left_knee","left_ankle",(60,180,180),2),
    ("left_ankle","left_foot",(40,140,140),1),
    ("left_ankle","left_heel",(40,140,140),1),
    ("right_hip","right_knee",(220,80,220),3),
    ("right_knee","right_ankle",(180,60,180),2),
    ("right_ankle","right_foot",(140,40,140),1),
    ("right_ankle","right_heel",(140,40,140),1),
]

ANGLE_JOINTS = {
    "L shoulder": (LM["left_shoulder"],  LM["left_elbow"],     LM["left_hip"],       (-52,-10)),
    "R shoulder": (LM["right_shoulder"], LM["right_elbow"],    LM["right_hip"],      ( 10,-10)),
    "L elbow":    (LM["left_elbow"],     LM["left_shoulder"],  LM["left_wrist"],     (-52,  0)),
    "R elbow":    (LM["right_elbow"],    LM["right_shoulder"], LM["right_wrist"],    ( 10,  0)),
    "L hip":      (LM["left_hip"],       LM["left_shoulder"],  LM["left_knee"],      (-52,  0)),
    "R hip":      (LM["right_hip"],      LM["right_shoulder"], LM["right_knee"],     ( 10,  0)),
    "L knee":     (LM["left_knee"],      LM["left_hip"],       LM["left_ankle"],     (-52,  0)),
    "R knee":     (LM["right_knee"],     LM["right_hip"],      LM["right_ankle"],    ( 10,  0)),
}
ANGLE_NAMES = list(ANGLE_JOINTS.keys())

BG_COLOR  = (18, 18, 24)
CANVAS_W  = 720
CANVAS_H  = 960
MARGIN    = 0.10
FONT      = cv2.FONT_HERSHEY_SIMPLEX

# Stage-4 smoothing: clip each joint's angle to ±this many degrees around its
# median before filtering, to reject MediaPipe spikes.
OUTLIER_CLIP_DEG = 40.0
# G1 has no direct ankle-pitch readout from 2D pose; approximate it as a fixed
# fraction of knee flexion so the foot stays roughly flat as the knee bends.
ANKLE_KNEE_COUPLING = -0.5

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _emit(cb: Optional[Callable], stage: int, msg: str, data: dict = None):
    if cb:
        cb(stage, msg, data or {})


def _ensure_model():
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _make_landmarker():
    """Create a configured PoseLandmarker (single source of options)."""
    _ensure_model()
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


class _LM:
    """Lightweight landmark with the attributes the renderer/angle code needs."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, d):
        self.x, self.y, self.z = d["x"], d["y"], d["z"]
        self.visibility = d["visibility"]


def _restore_landmarks(frame_dict):
    """Rebuild landmark objects from the data already stored in stage 3."""
    lms = frame_dict.get("landmarks")
    return [_LM(d) for d in lms] if lms else None


def _angle3pt(a, v, b):
    va, vb = a - v, b - v
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))))


def _compute_angles(lms):
    pts = np.array([[p.x, p.y, p.z] for p in lms])
    angles = {}
    for label, (v_i, a_i, b_i, _) in ANGLE_JOINTS.items():
        ang = _angle3pt(pts[a_i], pts[v_i], pts[b_i])
        angles[label] = ang if ang is not None else 0.0
    return angles


def _angles_to_q(a: dict) -> dict:
    """Map the 8 extracted MediaPipe angles (deg) → 10 G1 joint targets (rad)."""
    q = {
        "left_knee":            float(np.deg2rad(a.get("L knee", 0))),
        "right_knee":           float(np.deg2rad(a.get("R knee", 0))),
        "left_elbow":           float(np.deg2rad(a.get("L elbow", 0))),
        "right_elbow":          float(np.deg2rad(a.get("R elbow", 0))),
        "left_hip_pitch":       float(np.deg2rad(-(180 - a.get("L hip", 180)))),
        "right_hip_pitch":      float(np.deg2rad(-(180 - a.get("R hip", 180)))),
        "left_shoulder_pitch":  float(np.deg2rad(a.get("L shoulder", 90) - 90)),
        "right_shoulder_pitch": float(np.deg2rad(a.get("R shoulder", 90) - 90)),
    }
    q["left_ankle_pitch"]  = ANKLE_KNEE_COUPLING * q["left_knee"]
    q["right_ankle_pitch"] = ANKLE_KNEE_COUPLING * q["right_knee"]
    return q


def _clip_scale(all_frames, w, h, margin=MARGIN):
    """One fixed scale for the whole clip so the figure doesn't 'breathe'.

    Uses the median per-frame body extent (robust to a few bad frames) so the
    rendered skeleton keeps a constant size instead of pulsing with each frame's
    own bounding box.
    """
    xrs, yrs = [], []
    for f in all_frames:
        lms = f.get("landmarks")
        if not lms:
            continue
        xs = [d["x"] for d in lms]
        ys = [d["y"] for d in lms]
        xrs.append(max(xs) - min(xs))
        yrs.append(max(ys) - min(ys))
    if not yrs:
        return 1.0
    xr = float(np.median(xrs)) or 1.0
    yr = float(np.median(yrs)) or 1.0
    uw = w * (1 - 2 * margin)
    uh = h * (1 - 2 * margin)
    return min(uw / xr, uh / yr)


def _landmarks_to_pixels(lms, w, h, scale=None, margin=MARGIN):
    xs = np.array([l.x for l in lms])
    ys = np.array([l.y for l in lms])

    if scale is None:
        # Legacy per-frame fit (kept for any standalone callers).
        xr = xs.max() - xs.min() or 1.0
        yr = ys.max() - ys.min() or 1.0
        scale = min(w * (1 - 2 * margin) / xr, h * (1 - 2 * margin) / yr)
        px = (xs - xs.min()) * scale + (w - xr * scale) / 2
        py = (ys - ys.min()) * scale + (h - yr * scale) / 2 + h * 0.03
        return np.stack([px, py], axis=1)

    # Fixed scale + hip-anchored centring → stable size and position.
    hx = (xs[LM["left_hip"]] + xs[LM["right_hip"]]) / 2
    hy = (ys[LM["left_hip"]] + ys[LM["right_hip"]]) / 2
    px = (xs - hx) * scale + w / 2
    py = (ys - hy) * scale + h * 0.55
    return np.stack([px, py], axis=1)


def _draw_arc(canvas, vp, ap, bp, ang, radius=22, color=(220,220,220)):
    va = (ap - vp).astype(float)
    na = np.linalg.norm(va)
    if na < 1:
        return
    start = float(np.degrees(np.arctan2(-va[1], va[0])))
    sweep = float(np.clip(ang, 5, 175))
    cv2.ellipse(canvas, tuple(vp.astype(int)), (radius, radius),
                0, -start, -start + sweep, color, 1, cv2.LINE_AA)


def _render_frame(lms, frame_idx, fps, smoothed_angles=None, scale=None,
                  canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)
    for x in range(0, canvas_w, 60):
        cv2.line(canvas, (x,0), (x,canvas_h), (28,28,36), 1)
    for y in range(0, canvas_h, 60):
        cv2.line(canvas, (0,y), (canvas_w,y), (28,28,36), 1)

    if lms is None:
        cv2.putText(canvas, "No pose detected", (canvas_w//2 - 100, canvas_h//2),
                    FONT, 0.8, (100,100,100), 1, cv2.LINE_AA)
        return canvas

    pts = _landmarks_to_pixels(lms, canvas_w, canvas_h, scale=scale)
    vis = np.array([l.visibility if hasattr(l,"visibility") and l.visibility is not None
                    else 0.0 for l in lms])

    for (an, bn, color, thick) in SEGMENTS:
        ai, bi = LM[an], LM[bn]
        if vis[ai] < 0.3 or vis[bi] < 0.3:
            continue
        p1, p2 = tuple(pts[ai].astype(int)), tuple(pts[bi].astype(int))
        cv2.line(canvas, p1, p2, tuple(int(c*0.3) for c in color), thick+4, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        if vis[i] < 0.3:
            continue
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(canvas, (cx,cy), 7, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx,cy), 5, (220,220,255), -1, cv2.LINE_AA)

    if smoothed_angles:
        for label, (v_i, a_i, b_i, off) in ANGLE_JOINTS.items():
            if vis[v_i] < 0.4 or vis[a_i] < 0.4 or vis[b_i] < 0.4:
                continue
            ang = smoothed_angles.get(label)
            if ang is None:
                continue
            color = (80,160,255) if label.startswith("L") else (255,160,80)
            _draw_arc(canvas, pts[v_i], pts[a_i], pts[b_i], ang, color=color)
            tx = int(pts[v_i][0]) + off[0]
            ty = int(pts[v_i][1]) + off[1]
            txt = f"{label}: {ang:.1f}"
            (tw, th), _ = cv2.getTextSize(txt, FONT, 0.38, 1)
            cv2.rectangle(canvas, (tx-3, ty-th-2), (tx+tw+3, ty+3), (10,10,18), -1)
            cv2.putText(canvas, txt, (tx,ty), FONT, 0.38, color, 1, cv2.LINE_AA)

    if vis[LM["nose"]] > 0.3:
        hx, hy = int(pts[LM["nose"]][0]), int(pts[LM["nose"]][1])
        head_r = max(18, int(abs(pts[LM["left_eye"]][0]-pts[LM["right_eye"]][0])*2.2))
        cv2.circle(canvas, (hx,hy), head_r+3, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (hx,hy), head_r, (100,140,200), 2, cv2.LINE_AA)

    ts = frame_idx / fps
    cv2.putText(canvas, f"Frame {frame_idx:04d}  |  {ts:6.2f}s",
                (14, canvas_h-12), FONT, 0.4, (80,80,100), 1, cv2.LINE_AA)
    return canvas


# ─── Stage 1: Storyline → Video Prompt ───────────────────────────────────────

def stage1_generate_prompt(storyline: str, cb: Optional[Callable] = None) -> str:
    _emit(cb, 1, "Connecting to NVIDIA LLM to generate action sequence...")
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    stream = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": SCRIPT_SYSTEM},
            {"role": "user", "content": f"Storyline: {storyline}"},
        ],
        temperature=0.9,
        top_p=0.95,
        max_tokens=256,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        c = chunk.choices[0].delta.content
        if c:
            chunks.append(c)
            _emit(cb, 1, "streaming", {"token": c})

    action_sequence = "".join(chunks).strip()
    # Combine with BASE_PROMPT for optimal pose extraction
    full_prompt = f"{VEO_BASE_PROMPT}\n\n{action_sequence}"
    _emit(cb, 1, "Video prompt ready.", {"prompt": full_prompt, "action": action_sequence})
    return full_prompt


# ─── Stage 2 helpers: individual providers ────────────────────────────────────

def _download_video(url: str, out_path: Path):
    r = http_requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)


VEO_MODEL     = os.getenv("VEO_MODEL", "veo-3.0-generate-001")
VEO_DURATION  = int(os.getenv("VEO_DURATION", "8"))   # 4, 6, or 8
VEO_API_BASE  = "https://generativelanguage.googleapis.com/v1beta"

def _stage2_veo(prompt: str, out_path: Path, cb: Optional[Callable]) -> Path:
    """Google Veo via direct REST API (no SDK — avoids all SDK quirks)."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GOOGLE_API_KEY,
    }
    is_veo2 = "veo-2" in VEO_MODEL
    valid_durations = [5, 6, 8] if is_veo2 else [4, 6, 8]
    duration = VEO_DURATION if VEO_DURATION in valid_durations else valid_durations[-1]

    body = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "aspectRatio": "16:9",
            "durationSeconds": duration,
            "sampleCount": 1,
            "personGeneration": "allow_all",
        },
    }

    _emit(cb, 2, f"Submitting to Veo ({VEO_MODEL}, {duration}s)...")
    resp = http_requests.post(
        f"{VEO_API_BASE}/models/{VEO_MODEL}:predictLongRunning",
        headers=headers, json=body, timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Veo submit failed [{resp.status_code}]: {resp.text[:500]}")

    operation_name = resp.json().get("name")
    if not operation_name:
        raise RuntimeError(f"No operation name in response: {resp.json()}")

    _emit(cb, 2, f"Job started: {operation_name.split('/')[-1]}. Polling...")

    poll_url = f"{VEO_API_BASE}/{operation_name}"
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        if elapsed > 900:
            raise TimeoutError("Veo timed out after 15 minutes.")
        time.sleep(10)
        poll = http_requests.get(poll_url, headers=headers, timeout=30)
        if poll.status_code != 200:
            continue
        status = poll.json()
        if not status.get("done"):
            _emit(cb, 2, f"Generating... ({elapsed}s)")
            continue

        if "error" in status:
            raise RuntimeError(f"Veo generation failed: {status['error']}")

        _emit(cb, 2, f"Done in {elapsed}s. Saving video...")
        response = status.get("response", {})

        # Try all known response shapes
        video_data = None
        if response.get("generateVideoResponse", {}).get("generatedSamples"):
            video_data = response["generateVideoResponse"]["generatedSamples"][0].get("video", {})
        elif response.get("generatedVideos"):
            video_data = response["generatedVideos"][0].get("video", {})
        elif response.get("videos"):
            video_data = response["videos"][0]

        if not video_data:
            # Dump response for debugging
            (out_path.parent / "veo_response.json").write_text(
                json.dumps(status, indent=2))
            raise RuntimeError("Unknown Veo response format. Saved veo_response.json for inspection.")

        if "bytesBase64Encoded" in video_data:
            out_path.write_bytes(base64.b64decode(video_data["bytesBase64Encoded"]))
        elif "uri" in video_data:
            _download_video(video_data["uri"], out_path)
        elif "gcsUri" in video_data:
            raise RuntimeError(f"GCS download not supported. URI: {video_data['gcsUri']}")
        else:
            raise RuntimeError(f"Unknown video format: {list(video_data.keys())}")

        _emit(cb, 2, f"Video saved -> {out_path}", {"video_path": str(out_path)})
        return out_path


def _stage2_hf(prompt: str, out_path: Path, cb: Optional[Callable]) -> Path:
    """Free video gen via Hugging Face public Spaces (no API key needed)."""
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        raise RuntimeError("Run: pip install gradio-client")

    space = HF_SPACE
    _emit(cb, 2, f"Connecting to HuggingFace Space: {space} (free, no key needed)...")

    client = Client(space)

    # Each space has a slightly different API — try common patterns
    if "LTX-Video" in space:
        _emit(cb, 2, "Generating with LTX-Video (~30-60s)...")
        result = client.predict(
            prompt=prompt,
            negative_prompt="blurry, low quality, static, no motion",
            num_inference_steps=30,
            guidance_scale=3.0,
            seed=-1,
            api_name="/generate",
        )

    elif "CogVideoX" in space:
        _emit(cb, 2, "Generating with CogVideoX-5B (~3-5 min)...")
        result = client.predict(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=6.0,
            seed=42,
            api_name="/generate_video",
        )

    elif "Wan" in space:
        _emit(cb, 2, "Generating with Wan 2.1 (~3-5 min)...")
        result = client.predict(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            num_frames=81,
            guidance_scale=5.0,
            api_name="/generate",
        )

    else:
        _emit(cb, 2, f"Generating with {space}...")
        result = client.predict(prompt=prompt, api_name="/generate")

    # result is usually a file path or dict with a video path
    video_file = result
    if isinstance(result, dict):
        video_file = result.get("video") or result.get("output") or list(result.values())[0]
    if isinstance(result, (list, tuple)):
        video_file = result[0]

    if not video_file or not Path(str(video_file)).exists():
        raise RuntimeError(f"HF Space returned unexpected result: {result}")

    import shutil
    shutil.copy2(str(video_file), str(out_path))
    _emit(cb, 2, f"Video saved -> {out_path}", {"video_path": str(out_path)})
    return out_path


def _stage2_fal(prompt: str, out_path: Path, cb: Optional[Callable]) -> Path:
    """fal.ai queue API — works with Kling 1.6, Hunyuan, LTX, etc."""
    _emit(cb, 2, f"Submitting to fal.ai ({FAL_MODEL})...")
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }
    # Submit
    submit = http_requests.post(
        f"https://queue.fal.run/{FAL_MODEL}",
        headers=headers,
        json={"prompt": prompt, "duration": "5", "aspect_ratio": "16:9"},
        timeout=30,
    )
    submit.raise_for_status()
    request_id = submit.json()["request_id"]
    _emit(cb, 2, f"Queued (id={request_id}). Waiting...")

    # Poll
    status_url = f"https://queue.fal.run/{FAL_MODEL}/requests/{request_id}/status"
    result_url = f"https://queue.fal.run/{FAL_MODEL}/requests/{request_id}"
    for attempt in range(240):
        time.sleep(5)
        poll = http_requests.get(status_url, headers=headers, timeout=15)
        if poll.status_code != 200:
            continue
        status = poll.json().get("status", "")
        elapsed = attempt * 5
        _emit(cb, 2, f"Status: {status} ({elapsed}s)", {"status": status})
        if status == "COMPLETED":
            result = http_requests.get(result_url, headers=headers, timeout=15).json()
            video_url = (result.get("video") or {}).get("url") or \
                        (result.get("videos") or [{}])[0].get("url", "")
            if not video_url:
                raise RuntimeError(f"No video URL in fal result: {result}")
            _emit(cb, 2, "Downloading video...")
            _download_video(video_url, out_path)
            return out_path
        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal.ai job failed: {poll.json()}")
    raise TimeoutError("fal.ai video generation timed out after 20 minutes.")


def _stage2_runway(prompt: str, out_path: Path, cb: Optional[Callable]) -> Path:
    """RunwayML Gen-3 Alpha Turbo via their REST API."""
    _emit(cb, 2, f"Submitting to RunwayML ({RUNWAY_MODEL})...")
    headers = {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "Content-Type": "application/json",
        "X-Runway-Version": "2024-11-06",
    }
    submit = http_requests.post(
        "https://api.dev.runwayml.com/v1/text_to_video",
        headers=headers,
        json={"model": RUNWAY_MODEL, "promptText": prompt, "duration": 5,
              "ratio": "1280:720"},
        timeout=30,
    )
    submit.raise_for_status()
    task_id = submit.json()["id"]
    _emit(cb, 2, f"Task submitted (id={task_id}). Polling...")

    for attempt in range(240):
        time.sleep(5)
        poll = http_requests.get(
            f"https://api.dev.runwayml.com/v1/tasks/{task_id}",
            headers=headers, timeout=15,
        )
        if poll.status_code != 200:
            continue
        pdata  = poll.json()
        status = pdata.get("status", "").lower()
        _emit(cb, 2, f"Status: {status} ({attempt*5}s)")
        if status == "succeeded":
            video_url = (pdata.get("output") or [""])[0]
            if not video_url:
                raise RuntimeError(f"No output URL: {pdata}")
            _emit(cb, 2, "Downloading video...")
            _download_video(video_url, out_path)
            return out_path
        if status in ("failed", "error"):
            raise RuntimeError(f"Runway job failed: {pdata}")
    raise TimeoutError("RunwayML video generation timed out.")


# ─── Stage 2: Prompt → Video ─────────────────────────────────────────────────

def stage2_generate_video(prompt: str, out_path: Path, cb: Optional[Callable] = None) -> Path:
    provider = VIDEO_PROVIDER.lower()
    _emit(cb, 2, f"Video provider: {provider}", {"provider": provider})

    if provider == "veo":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not set in .env")
        return _stage2_veo(prompt, out_path, cb)
    elif provider == "hf":
        return _stage2_hf(prompt, out_path, cb)
    elif provider == "fal":
        if not FAL_API_KEY:
            raise RuntimeError("FAL_API_KEY not set in .env")
        return _stage2_fal(prompt, out_path, cb)
    elif provider == "runway":
        if not RUNWAY_API_KEY:
            raise RuntimeError("RUNWAY_API_KEY not set in .env")
        return _stage2_runway(prompt, out_path, cb)
    else:
        raise ValueError(f"Unknown VIDEO_PROVIDER '{provider}'. Use: veo | hf | fal | runway")


# ─── Stage 3: Video → Raw Joint Angles ───────────────────────────────────────

def stage3_extract_angles(video_path: Path, cb: Optional[Callable] = None) -> list[dict]:
    _emit(cb, 3, "Loading MediaPipe pose model…")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _emit(cb, 3, f"Opened video: {total} frames @ {fps:.1f} fps", {"total_frames": total})

    all_frames = []
    with _make_landmarker() as lmk:
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = int(fidx / fps * 1000)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = lmk.detect_for_video(mp_img, ts_ms)
            lms = result.pose_landmarks[0] if result.pose_landmarks else None

            fd = {"frame": fidx, "fps": fps, "has_pose": lms is not None, "angles": {}, "landmarks": []}
            if lms is not None:
                fd["angles"] = _compute_angles(lms)
                fd["landmarks"] = [
                    {"x": float(l.x), "y": float(l.y), "z": float(l.z),
                     "visibility": float(l.visibility)} for l in lms
                ]
            all_frames.append(fd)

            if fidx % 30 == 0:
                pct = int(fidx / max(total,1) * 100)
                _emit(cb, 3, f"Extracting… {fidx}/{total} frames", {"progress": pct, "frame": fidx})
            fidx += 1

    cap.release()
    detected = sum(1 for f in all_frames if f["has_pose"])
    _emit(cb, 3, f"Extraction complete: {detected}/{len(all_frames)} frames with pose.",
          {"detected": detected, "total": len(all_frames)})
    return all_frames


# ─── Stage 4: Smooth & Filter Angles ─────────────────────────────────────────

def stage4_smooth_angles(all_frames: list[dict], cb: Optional[Callable] = None) -> list[dict]:
    _emit(cb, 4, "Applying outlier removal and Savitzky-Golay smoothing…")

    pose_indices = [i for i, f in enumerate(all_frames) if f["has_pose"]]
    pose_frames  = [all_frames[i] for i in pose_indices]
    n = len(pose_frames)

    if n < 5:
        _emit(cb, 4, "Too few pose frames to smooth — returning raw angles.")
        return all_frames

    window   = min(11, n if n % 2 == 1 else n - 1)
    polyord  = min(3, window - 1)

    for name in ANGLE_NAMES:
        vals = np.array([f["angles"].get(name, 0.0) for f in pose_frames], dtype=float)
        median = np.median(vals)
        vals   = np.clip(vals, median - OUTLIER_CLIP_DEG, median + OUTLIER_CLIP_DEG)
        vals   = savgol_filter(vals, window_length=window, polyorder=polyord)
        for i, v in enumerate(vals):
            pose_frames[i]["angles"][name] = float(v)

    for idx, pi in enumerate(pose_indices):
        all_frames[pi]["angles"] = pose_frames[idx]["angles"]

    _emit(cb, 4, f"Smoothing done (window={window}, poly={polyord}).",
          {"window": window, "polyorder": polyord, "pose_frames": n})
    return all_frames


# ─── Stage 5: Render Stick Figure + Trajectory ───────────────────────────────

def stage5_build_output(all_frames: list[dict], video_path: Path,
                        output_dir: Path, cb: Optional[Callable] = None) -> dict:
    _emit(cb, 5, "Rendering stick-figure video and building trajectory JSON…")
    output_dir.mkdir(parents=True, exist_ok=True)

    stick_path = output_dir / "stick_figure.mp4"
    traj_path  = output_dir / "trajectory.json"

    sample = next((f for f in all_frames if f.get("fps")), None)
    fps    = sample["fps"] if sample else 30.0
    total  = len(all_frames)

    smoothed_by_frame = {f["frame"]: f["angles"] for f in all_frames if f["has_pose"]}
    scale = _clip_scale(all_frames, CANVAS_W, CANVAS_H)   # fixed size for whole clip

    # Render directly from the landmarks stored in stage 3 — no second MediaPipe
    # pass, no video re-decode. Write H.264 + yuv420p so the <video> tag can play it.
    writer = imageio.get_writer(
        str(stick_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    for f in all_frames:
        lms    = _restore_landmarks(f)
        sa     = smoothed_by_frame.get(f["frame"]) if lms is not None else None
        canvas = _render_frame(lms, f["frame"], fps, smoothed_angles=sa, scale=scale)
        writer.append_data(canvas[:, :, ::-1])   # BGR (OpenCV) → RGB (imageio)
        if f["frame"] % 30 == 0:
            pct = int(f["frame"] / max(total, 1) * 100)
            _emit(cb, 5, f"Rendering… {f['frame']}/{total}", {"progress": pct})
    writer.close()

    # Build trajectory JSON
    pose_frames = [f for f in all_frames if f["has_pose"]]
    trajectory  = []
    for i, f in enumerate(pose_frames):
        q     = _angles_to_q(f["angles"])
        # fps is stored per-entry: g1_env.load_reference reads it from data[0]
        # (without it the RL env assumes 30fps and replays the motion too fast)
        entry = {"frame": f["frame"], "fps": fps, "q_ref": q}

        if i < len(pose_frames) - 1:
            nf        = pose_frames[i + 1]
            nq        = _angles_to_q(nf["angles"])
            frame_gap = nf["frame"] - f["frame"]
            dt        = frame_gap / fps if frame_gap > 0 else 1.0 / fps
            entry["qvel_ref"] = {j: (nq[j] - q[j]) / dt for j in q}
        else:
            entry["qvel_ref"] = {j: 0.0 for j in q}

        trajectory.append(entry)

    traj_path.write_text(json.dumps(trajectory, indent=2))

    # Per-joint stats — gather each joint's values once, then reduce
    angle_stats = {}
    for name in ANGLE_NAMES:
        arr = np.array([f["angles"][name] for f in pose_frames if name in f["angles"]])
        if arr.size:
            angle_stats[name] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "min":  float(arr.min()),  "max": float(arr.max()),
            }

    summary = {
        "stick_video":  str(stick_path),
        "trajectory":   str(traj_path),
        "total_frames": total,
        "pose_frames":  len(pose_frames),
        "joints":       ANGLE_NAMES,
        "angle_stats":  angle_stats,
    }

    _emit(cb, 5, "Pipeline complete!", summary)
    return summary


# ─── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(storyline: str, cb: Optional[Callable] = None,
                 use_existing_video: Optional[str] = None) -> dict:
    """
    Run all 5 stages end-to-end.
    If use_existing_video is provided (path to an mp4), stage 2 is skipped.
    Returns the summary dict from stage 5.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1
    video_prompt = stage1_generate_prompt(storyline, cb)

    # Stage 2
    if use_existing_video:
        video_path = Path(use_existing_video)
        _emit(cb, 2, f"Skipping video generation — using: {video_path}",
              {"video_path": str(video_path)})
    else:
        video_path = OUTPUT_DIR / "generated_video.mp4"
        stage2_generate_video(video_prompt, video_path, cb)

    # Stage 3
    all_frames = stage3_extract_angles(video_path, cb)

    # Stage 4
    all_frames = stage4_smooth_angles(all_frames, cb)

    # Stage 5
    summary = stage5_build_output(all_frames, video_path, OUTPUT_DIR, cb)
    summary["video_prompt"] = video_prompt
    summary["storyline"]    = storyline

    result_path = OUTPUT_DIR / "pipeline_result.json"
    result_path.write_text(json.dumps(summary, indent=2))
    return summary



# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    storyline = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Storyline: ").strip()

    def cli_cb(stage, msg, data):
        tag = f"[Stage {stage}]"
        if data.get("token"):
            print(data["token"], end="", flush=True)
        else:
            print(f"{tag} {msg}")

    result = run_pipeline(storyline, cb=cli_cb)
    print("\n─── Pipeline Result ───")
    print(json.dumps(result, indent=2))
