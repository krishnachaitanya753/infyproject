"""
Humanoid Robot Joint Angle Estimator
-------------------------------------
Compatible with MediaPipe >= 0.10 (new Tasks API).

Install dependencies:
    pip install mediapipe opencv-python numpy requests

Usage:
    python estimate_joint_angles.py --video robot.mp4 --output angles.json

The script auto-downloads the pose_landmarker model on first run.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
import urllib.request
from pathlib import Path

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as lm_module


# ─── Model download ───────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = Path("pose_landmarker_heavy.task")

def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading pose model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Done.")


# ─── Landmark index map (same 33-point body as the old API) ──────────────────
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow":    13, "right_elbow":    14,
    "left_wrist":    15, "right_wrist":    16,
    "left_index":    19, "right_index":    20,
    "left_hip":      23, "right_hip":      24,
    "left_knee":     25, "right_knee":     26,
    "left_ankle":    27, "right_ankle":    28,
    "left_foot":     31, "right_foot":     32,
}


# ─── Joint definitions (proximal, vertex, distal) ────────────────────────────
JOINTS = {
    "left_shoulder":  ("left_elbow",    "left_shoulder",  "left_hip"),
    "right_shoulder": ("right_elbow",   "right_shoulder", "right_hip"),
    "left_elbow":     ("left_shoulder", "left_elbow",     "left_wrist"),
    "right_elbow":    ("right_shoulder","right_elbow",    "right_wrist"),
    "left_wrist":     ("left_elbow",    "left_wrist",     "left_index"),
    "right_wrist":    ("right_elbow",   "right_wrist",    "right_index"),
    "left_hip":       ("left_shoulder", "left_hip",       "left_knee"),
    "right_hip":      ("right_shoulder","right_hip",      "right_knee"),
    "left_knee":      ("left_hip",      "left_knee",      "left_ankle"),
    "right_knee":     ("right_hip",     "right_knee",     "right_ankle"),
    "left_ankle":     ("left_knee",     "left_ankle",     "left_foot"),
    "right_ankle":    ("right_knee",    "right_ankle",    "right_foot"),
}


# ─── Math helpers ─────────────────────────────────────────────────────────────

def lm_to_vec(lm):
    return np.array([lm.x, lm.y, lm.z])


def angle_between(a, vertex, b):
    va = a - vertex
    vb = b - vertex
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return None
    cos_t = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def compute_angles(landmarks, min_vis=0.5):
    """landmarks: list of NormalizedLandmark from the Tasks API."""
    angles = {}
    for jname, (a_key, v_key, b_key) in JOINTS.items():
        a_i, v_i, b_i = LM[a_key], LM[v_key], LM[b_key]
        a_lm, v_lm, b_lm = landmarks[a_i], landmarks[v_i], landmarks[b_i]

        # visibility check (Tasks API exposes .visibility as Optional[float])
        vis = getattr(v_lm, "visibility", None)
        if vis is not None and vis < min_vis:
            angles[jname] = None
            continue

        deg = angle_between(lm_to_vec(a_lm), lm_to_vec(v_lm), lm_to_vec(b_lm))
        angles[jname] = round(deg, 2) if deg is not None else None

    return angles


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_video(video_path, output_path, skip_frames=1,
                  min_visibility=0.5, draw_debug=False):

    ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path}  {w}×{h}  {fps:.1f} fps  {total} frames")

    debug_writer = None
    if draw_debug:
        dbg = str(Path(output_path).with_stem(Path(output_path).stem + "_debug").with_suffix(".mp4"))
        debug_writer = cv2.VideoWriter(dbg, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Debug video → {dbg}")

    # Build the Tasks-API landmarker (VIDEO mode for temporal smoothing)
    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    results_list = []

    with mp_vision.PoseLandmarker.create_from_options(opts) as landmarker:
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            timestamp_ms = int(frame_idx / fps * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection = landmarker.detect_for_video(mp_image, timestamp_ms)

            entry = {
                "frame":         frame_idx,
                "timestamp_s":   round(frame_idx / fps, 4),
                "pose_detected": False,
                "joints":        {},
            }

            if detection.pose_landmarks:
                lms = detection.pose_landmarks[0]   # first (only) pose
                entry["pose_detected"] = True
                entry["joints"] = compute_angles(lms, min_vis=min_visibility)

                if draw_debug and debug_writer:
                    # Draw skeleton manually (drawing_utils removed in 0.10)
                    CONNECTIONS = [
                        (11,12),(11,13),(13,15),(12,14),(14,16),
                        (11,23),(12,24),(23,24),(23,25),(24,26),
                        (25,27),(26,28),(27,31),(28,32),
                    ]
                    for i, j in CONNECTIONS:
                        lm_a = lms[i]; lm_b = lms[j]
                        x1,y1 = int(lm_a.x*w), int(lm_a.y*h)
                        x2,y2 = int(lm_b.x*w), int(lm_b.y*h)
                        cv2.line(frame, (x1,y1), (x2,y2), (0,220,100), 2)
                    for lm in lms:
                        cx,cy = int(lm.x*w), int(lm.y*h)
                        cv2.circle(frame, (cx,cy), 4, (255,120,0), -1)

            results_list.append(entry)
            if debug_writer:
                debug_writer.write(frame)

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{total}", end="\r")

            frame_idx += 1

    cap.release()
    if debug_writer:
        debug_writer.release()

    # ── Summary stats ─────────────────────────────────────────────────────────
    detected = [f for f in results_list if f["pose_detected"]]
    stats = {}
    for jname in JOINTS:
        vals = [f["joints"].get(jname) for f in detected
                if f["joints"].get(jname) is not None]
        if vals:
            stats[jname] = {
                "mean": round(float(np.mean(vals)), 2),
                "std":  round(float(np.std(vals)),  2),
                "min":  round(float(np.min(vals)),  2),
                "max":  round(float(np.max(vals)),  2),
            }

    output = {
        "metadata": {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total,
            "processed_frames": len(results_list),
            "pose_detected_frames": len(detected),
            "joint_names": list(JOINTS.keys()),
            "angle_stats": stats,
        },
        "frames": results_list,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {output_path}")
    print(f"Pose detected in {len(detected)}/{len(results_list)} frames")
    return output


# ─── Optional CSV export ──────────────────────────────────────────────────────

def export_csv(json_path):
    import csv
    with open(json_path) as f:
        data = json.load(f)
    jnames   = data["metadata"]["joint_names"]
    csv_path = json_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_s", "pose_detected"] + jnames)
        for fr in data["frames"]:
            row = [fr["frame"], fr["timestamp_s"], fr["pose_detected"]]
            row += [fr["joints"].get(jn, "") for jn in jnames]
            writer.writerow(row)
    print(f"CSV → {csv_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Humanoid joint angle estimator (MediaPipe >= 0.10)")
    parser.add_argument("--video",   required=True)
    parser.add_argument("--output",  default="angles.json")
    parser.add_argument("--skip",    type=int,   default=1,   help="Process every N-th frame")
    parser.add_argument("--min-vis", type=float, default=0.5, help="Min landmark visibility 0-1")
    parser.add_argument("--debug",   action="store_true",     help="Write annotated debug video")
    parser.add_argument("--csv",     action="store_true",     help="Also export CSV")
    args = parser.parse_args()

    process_video(args.video, args.output, args.skip, args.min_vis, args.debug)
    if args.csv:
        export_csv(args.output)