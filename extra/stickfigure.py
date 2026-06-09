"""
Stick Figure Video Generator — PPO-Ready Pipeline
---------------------------------------------------
Re-runs MediaPipe on the original video, extracts all 33 landmarks,
computes & smooths joint angles, generates G1 reference trajectory,
and draws a clean animated stick figure for every frame.

Pipeline:
  Video -> MediaPipe -> Store 33 landmarks -> Compute angles
  -> Outlier removal -> Temporal smoothing -> G1 joint conversion
  -> Save trajectory.json -> Draw cleaned stick figure -> Output video

Install:
    pip install mediapipe opencv-python numpy scipy

Usage:
    python stickfigure.py --video robot.mp4 --output clean_stick.mp4
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import urllib.request
import json
from pathlib import Path
from scipy.signal import savgol_filter


# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = Path("pose_landmarker_heavy.task")

def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading model -> {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Done.")


# ─── Skeleton definition ──────────────────────────────────────────────────────
# MediaPipe 33-point indices
LM = {
    "nose":0,"left_eye_inner":1,"left_eye":2,"left_eye_outer":3,
    "right_eye_inner":4,"right_eye":5,"right_eye_outer":6,
    "left_ear":7,"right_ear":8,
    "mouth_left":9,"mouth_right":10,
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

# Segments grouped by body part → (index_a, index_b, color_BGR, thickness)
SEGMENTS = [
    # Head
    ("nose",            "left_eye",       (180,200,255), 2),
    ("nose",            "right_eye",      (180,200,255), 2),
    # Torso
    ("left_shoulder",   "right_shoulder", (120,220,120), 3),
    ("left_shoulder",   "left_hip",       (120,220,120), 3),
    ("right_shoulder",  "right_hip",      (120,220,120), 3),
    ("left_hip",        "right_hip",      (120,220,120), 3),
    # Left arm
    ("left_shoulder",   "left_elbow",     ( 80,160,255), 3),
    ("left_elbow",      "left_wrist",     ( 60,120,220), 2),
    ("left_wrist",      "left_index",     ( 40, 90,180), 1),
    # Right arm
    ("right_shoulder",  "right_elbow",    (255,160, 80), 3),
    ("right_elbow",     "right_wrist",    (220,120, 60), 2),
    ("right_wrist",     "right_index",    (180, 90, 40), 1),
    # Left leg
    ("left_hip",        "left_knee",      ( 80,220,220), 3),
    ("left_knee",       "left_ankle",     ( 60,180,180), 2),
    ("left_ankle",      "left_foot",      ( 40,140,140), 1),
    ("left_ankle",      "left_heel",      ( 40,140,140), 1),
    # Right leg
    ("right_hip",       "right_knee",     (220, 80,220), 3),
    ("right_knee",      "right_ankle",    (180, 60,180), 2),
    ("right_ankle",     "right_foot",     (140, 40,140), 1),
    ("right_ankle",     "right_heel",     (140, 40,140), 1),
]

# Joint dots to highlight and which angles to label there
ANGLE_JOINTS = {
    # joint_name: (vertex_idx, proximal_idx, distal_idx, label_offset_xy)
    "L shoulder": (LM["left_shoulder"],  LM["left_elbow"],     LM["left_hip"],       (-52, -10)),
    "R shoulder": (LM["right_shoulder"], LM["right_elbow"],    LM["right_hip"],      ( 10, -10)),
    "L elbow":    (LM["left_elbow"],     LM["left_shoulder"],  LM["left_wrist"],     (-52,   0)),
    "R elbow":    (LM["right_elbow"],    LM["right_shoulder"], LM["right_wrist"],    ( 10,   0)),
    "L hip":      (LM["left_hip"],       LM["left_shoulder"],  LM["left_knee"],      (-52,   0)),
    "R hip":      (LM["right_hip"],      LM["right_shoulder"], LM["right_knee"],     ( 10,   0)),
    "L knee":     (LM["left_knee"],      LM["left_hip"],       LM["left_ankle"],     (-52,   0)),
    "R knee":     (LM["right_knee"],     LM["right_hip"],      LM["right_ankle"],    ( 10,   0)),
}

JOINT_DOT_RADIUS = 5

# Angle names for smoothing (must match ANGLE_JOINTS keys)
ANGLE_NAMES = [
    "L shoulder", "R shoulder",
    "L elbow",    "R elbow",
    "L hip",      "R hip",
    "L knee",     "R knee",
]

# ─── Canvas settings ──────────────────────────────────────────────────────────
BG_COLOR      = (18, 18, 24)          # near-black dark blue
CANVAS_W      = 720
CANVAS_H      = 960
MARGIN        = 0.10                  # 10% margin on each side
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ─── Math ─────────────────────────────────────────────────────────────────────

def angle3pt(a, v, b):
    va = a - v;  vb = b - v
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return None
    cos_t = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def compute_angles(lms):
    """Compute all 8 joint angles from 3D landmark coordinates."""
    pts = np.array([[p.x, p.y, p.z] for p in lms])
    angles = {}
    for label, (v_i, a_i, b_i, _) in ANGLE_JOINTS.items():
        ang = angle3pt(pts[a_i], pts[v_i], pts[b_i])
        angles[label] = ang if ang is not None else 0.0
    return angles


def angles_to_g1(frame):
    """Convert cleaned angles dict to G1 robot joint radians (10 DOFs)."""
    a = frame["angles"]
    q = {}

    q["left_knee"]  = np.deg2rad(a["L knee"])
    q["right_knee"] = np.deg2rad(a["R knee"])

    q["left_elbow"]  = np.deg2rad(a["L elbow"])
    q["right_elbow"] = np.deg2rad(a["R elbow"])

    q["left_hip_pitch"]  = np.deg2rad(-(180 - a["L hip"]))
    q["right_hip_pitch"] = np.deg2rad(-(180 - a["R hip"]))

    q["left_shoulder_pitch"]  = np.deg2rad(a["L shoulder"] - 90)
    q["right_shoulder_pitch"] = np.deg2rad(a["R shoulder"] - 90)

    q["left_ankle_pitch"]  = -0.5 * q["left_knee"]
    q["right_ankle_pitch"] = -0.5 * q["right_knee"]

    return q


def draw_arc(canvas, vertex_px, a_px, b_px, angle_deg, radius=22, color=(220,220,220)):
    """Draw a small arc at vertex showing the joint angle."""
    va = (a_px - vertex_px).astype(float)
    vb = (b_px - vertex_px).astype(float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1 or nb < 1:
        return
    start_deg = float(np.degrees(np.arctan2(-va[1], va[0])))
    sweep      = float(np.clip(angle_deg, 5, 175))
    cv2.ellipse(canvas,
                tuple(vertex_px.astype(int)),
                (radius, radius),
                0,
                -start_deg,
                -start_deg + sweep,
                color, 1, cv2.LINE_AA)


def landmarks_to_pixels(lms, w, h, margin=MARGIN):
    """
    Convert normalized landmarks to canvas pixel coords.
    Re-scales so the figure fills the canvas minus margin,
    centering it horizontally.
    """
    xs = np.array([l.x for l in lms])
    ys = np.array([l.y for l in lms])

    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    if x_range < 1e-4: x_range = 1.0
    if y_range < 1e-4: y_range = 1.0

    usable_w = w * (1 - 2 * margin)
    usable_h = h * (1 - 2 * margin)

    scale = min(usable_w / x_range, usable_h / y_range)
    px = (xs - xs.min()) * scale + (w - x_range * scale) / 2
    py = (ys - ys.min()) * scale + (h - y_range * scale) / 2 + h * 0.03

    return np.stack([px, py], axis=1)   # shape (33, 2)


# ─── Frame renderer ───────────────────────────────────────────────────────────

def render_frame(lms, frame_idx, fps, canvas_w=CANVAS_W, canvas_h=CANVAS_H,
                 show_angles=True, show_grid=True, smoothed_angles=None):

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    # Subtle grid
    if show_grid:
        for x in range(0, canvas_w, 60):
            cv2.line(canvas, (x, 0), (x, canvas_h), (28, 28, 36), 1)
        for y in range(0, canvas_h, 60):
            cv2.line(canvas, (0, y), (canvas_w, y), (28, 28, 36), 1)

    if lms is None:
        cv2.putText(canvas, "No pose detected", (canvas_w//2 - 100, canvas_h//2),
                    FONT, 0.8, (100,100,100), 1, cv2.LINE_AA)
        return canvas

    pts = landmarks_to_pixels(lms, canvas_w, canvas_h)
    vis = np.array([l.visibility if hasattr(l, "visibility") and l.visibility is not None
                    else 0.0 for l in lms])

    # ── Draw segments ──────────────────────────────────────────────────────
    for (a_name, b_name, color, thick) in SEGMENTS:
        ai, bi = LM[a_name], LM[b_name]
        if vis[ai] < 0.3 or vis[bi] < 0.3:
            continue
        p1 = tuple(pts[ai].astype(int))
        p2 = tuple(pts[bi].astype(int))
        cv2.line(canvas, p1, p2,
                 tuple(int(c * 0.3) for c in color), thick + 4, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)

    # ── Draw joint dots ────────────────────────────────────────────────────
    for i, pt in enumerate(pts):
        if vis[i] < 0.3:
            continue
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(canvas, (cx, cy), JOINT_DOT_RADIUS + 2, (0,0,0),   -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), JOINT_DOT_RADIUS,     (220,220,255), -1, cv2.LINE_AA)

    # ── Angle labels + arcs ────────────────────────────────────────────────
    if show_angles:
        for label, (v_i, a_i, b_i, off) in ANGLE_JOINTS.items():
            if vis[v_i] < 0.4 or vis[a_i] < 0.4 or vis[b_i] < 0.4:
                continue
            v_pt = pts[v_i]
            a_pt = pts[a_i]
            b_pt = pts[b_i]

            # FIX: Only use smoothed angles; do NOT fall back to 2D pixel-derived
            # angles which are geometrically inconsistent with the 3D-computed ones.
            if smoothed_angles and label in smoothed_angles:
                ang = smoothed_angles[label]
            else:
                # Compute from 3D landmarks (same method as Pass 1) to stay consistent
                ang = None

            if ang is None:
                continue

            color = (80,160,255) if label.startswith("L") else (255,160,80)

            draw_arc(canvas, v_pt, a_pt, b_pt, ang, radius=20, color=color)

            tx = int(v_pt[0]) + off[0]
            ty = int(v_pt[1]) + off[1]
            txt = f"{label}: {ang:.1f}"
            (tw, th), _ = cv2.getTextSize(txt, FONT, 0.38, 1)
            cv2.rectangle(canvas, (tx-3, ty-th-2), (tx+tw+3, ty+3),
                          (10,10,18), -1)
            cv2.putText(canvas, txt, (tx, ty), FONT, 0.38, color, 1, cv2.LINE_AA)

    # ── Head circle ────────────────────────────────────────────────────────
    if vis[LM["nose"]] > 0.3:
        hx, hy = int(pts[LM["nose"]][0]), int(pts[LM["nose"]][1])
        head_r = max(18, int(abs(pts[LM["left_eye"]][0] - pts[LM["right_eye"]][0]) * 2.2))
        cv2.circle(canvas, (hx, hy), head_r + 3, (0,0,0),       -1, cv2.LINE_AA)
        cv2.circle(canvas, (hx, hy), head_r,     (100,140,200),  2, cv2.LINE_AA)

    # ── Timestamp ──────────────────────────────────────────────────────────
    ts  = frame_idx / fps
    hud = f"Frame {frame_idx:04d}  |  {ts:6.2f}s"
    cv2.putText(canvas, hud, (14, canvas_h - 12),
                FONT, 0.4, (80, 80, 100), 1, cv2.LINE_AA)

    return canvas


# ─── Post-processing ──────────────────────────────────────────────────────────

def smooth_and_clean_angles(all_frames):
    """
    Apply outlier removal (median ± 40°) and Savitzky-Golay smoothing.

    FIX: polyorder is clamped to window_length - 1 to prevent ValueError
    when the clip is very short. Returns False if smoothing is skipped.
    """
    n = len(all_frames)
    if n < 5:
        print("  Warning: too few frames for smoothing, skipping.")
        return False

    # Adaptive window: must be odd and <= n
    window = min(11, n if n % 2 == 1 else n - 1)
    if window < 5:
        print("  Warning: window too small for smoothing, skipping.")
        return False

    # FIX: clamp polyorder so it never >= window_length
    polyorder = min(3, window - 1)

    for name in ANGLE_NAMES:
        vals = np.array([f["angles"][name] for f in all_frames], dtype=float)

        # ── Outlier removal: clip to median ± 40° ──
        median = np.median(vals)
        vals = np.clip(vals, median - 40, median + 40)

        # ── Temporal smoothing: Savitzky-Golay ──
        vals = savgol_filter(vals, window_length=window, polyorder=polyorder)

        for i, v in enumerate(vals):
            all_frames[i]["angles"][name] = float(v)

    return True


def generate_trajectory(pose_frames, out_fps):
    """
    Generate G1 reference trajectory with joint velocities.

    FIX: velocity is computed using actual frame index differences so that
    non-uniform gaps (e.g. when skip_frames > 1 combined with missed poses)
    produce correct dt per-step instead of assuming a fixed dt throughout.
    """
    trajectory = []
    for frame in pose_frames:
        q_ref = angles_to_g1(frame)
        trajectory.append({
            "frame": frame["frame"],
            "q_ref": q_ref,
        })

    # Compute velocities via finite differences with per-step dt
    for i in range(len(trajectory) - 1):
        curr_entry = trajectory[i]
        next_entry = trajectory[i + 1]

        # FIX: use actual frame gap, not assumed fixed dt
        frame_gap = next_entry["frame"] - curr_entry["frame"]
        dt = frame_gap / out_fps if frame_gap > 0 else 1.0 / out_fps

        curr_q = curr_entry["q_ref"]
        next_q = next_entry["q_ref"]
        qvel = {joint: (next_q[joint] - curr_q[joint]) / dt for joint in curr_q}
        trajectory[i]["qvel_ref"] = qvel

    # Last frame: zero velocity
    if trajectory:
        trajectory[-1]["qvel_ref"] = {j: 0.0 for j in trajectory[-1]["q_ref"]}

    return trajectory


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def generate_stick_video(video_path, output_path, trajectory_path,
                         canvas_w=CANVAS_W, canvas_h=CANVAS_H,
                         show_angles=True, show_grid=True,
                         skip_frames=1):

    ensure_model()

    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps / skip_frames if skip_frames > 1 else fps

    print(f"Input : {video_path}  ({total} frames @ {fps:.1f} fps)")
    print(f"Output: {output_path}  ({canvas_w}x{canvas_h} @ {out_fps:.1f} fps)")
    print(f"Trajectory: {trajectory_path}")

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 1: Collect all landmarks and angles
    # ═══════════════════════════════════════════════════════════════════════
    print("\n-- Pass 1: Extracting landmarks & angles --")
    all_frames = []

    with mp_vision.PoseLandmarker.create_from_options(opts) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            ts_ms = int(frame_idx / fps * 1000)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            lms = result.pose_landmarks[0] if result.pose_landmarks else None

            frame_data = {
                "frame": frame_idx,
                "landmarks": {},
                "angles": {},
                "has_pose": lms is not None,
            }

            if lms is not None:
                for idx, lm in enumerate(lms):
                    frame_data["landmarks"][idx] = {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility),
                    }
                frame_data["angles"] = compute_angles(lms)

            all_frames.append(frame_data)

            if frame_idx % 30 == 0:
                print(f"  {frame_idx}/{total}", end="\r")

            frame_idx += 1

    cap.release()

    # Separate pose frames for smoothing
    pose_indices = [i for i, f in enumerate(all_frames) if f["has_pose"]]
    pose_frames  = [all_frames[i] for i in pose_indices]

    print(f"\n  Extracted {len(all_frames)} frames, {len(pose_frames)} with pose")

    # ═══════════════════════════════════════════════════════════════════════
    # POST-PROCESSING: Outlier removal + Temporal smoothing
    # ═══════════════════════════════════════════════════════════════════════
    print("\n-- Post-processing: outlier removal & smoothing --")
    smoothing_applied = smooth_and_clean_angles(pose_frames)
    if not smoothing_applied:
        print("  Warning: smoothing skipped — trajectory angles are raw.")

    # Write smoothed values back into all_frames
    for idx, pi in enumerate(pose_indices):
        all_frames[pi]["angles"] = pose_frames[idx]["angles"]

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE G1 TRAJECTORY
    # ═══════════════════════════════════════════════════════════════════════
    print("-- Generating G1 reference trajectory --")

    # FIX: pass out_fps so per-step dt calculation can use it as base rate
    trajectory = generate_trajectory(pose_frames, out_fps)

    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"  Saved -> {trajectory_path}  ({len(trajectory)} frames)")

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2: Render cleaned stick figure video
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n-- Pass 2: Rendering cleaned stick figure -> {output_path} --")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (canvas_w, canvas_h),
    )

    # Re-open video for Pass 2
    cap2 = cv2.VideoCapture(video_path)

    # FIX: Build a lookup from frame_idx -> smoothed angles to decouple
    # the render loop from a fragile parallel counter.
    smoothed_angles_by_frame = {
        f["frame"]: f["angles"]
        for f in all_frames if f["has_pose"]
    }

    with mp_vision.PoseLandmarker.create_from_options(opts) as landmarker:
        frame_idx = 0

        while True:
            ret, frame = cap2.read()
            if not ret:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            ts_ms = int(frame_idx / fps * 1000)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            lms = result.pose_landmarks[0] if result.pose_landmarks else None

            # FIX: look up by frame index, not by a parallel counter
            smoothed_angles = smoothed_angles_by_frame.get(frame_idx)

            # Only pass smoothed_angles to render if we actually have a pose
            # for this frame; avoids rendering stale angles on a no-pose frame.
            if lms is None:
                smoothed_angles = None

            canvas = render_frame(
                lms, frame_idx, fps,
                canvas_w=canvas_w, canvas_h=canvas_h,
                show_angles=show_angles,
                show_grid=show_grid,
                smoothed_angles=smoothed_angles,
            )
            writer.write(canvas)

            if frame_idx % 30 == 0:
                print(f"  {frame_idx}/{total}", end="\r")

            frame_idx += 1

    cap2.release()
    writer.release()
    print(f"\nDone -> {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stick figure video + PPO reference trajectory from robot motion"
    )
    parser.add_argument("--video",        required=True)
    parser.add_argument("--output",       default="clean_stick.mp4")
    parser.add_argument("--trajectory",   default="g1_reference_motion.json",
                        help="Output path for G1 PPO reference trajectory")
    parser.add_argument("--width",        type=int, default=720,  help="Canvas width  (default 720)")
    parser.add_argument("--height",       type=int, default=960,  help="Canvas height (default 960)")
    parser.add_argument("--skip",         type=int, default=1,    help="Process every N-th frame")
    parser.add_argument("--no-angles",    action="store_true",    help="Hide joint angle labels")
    parser.add_argument("--no-grid",      action="store_true",    help="Hide background grid")
    args = parser.parse_args()

    generate_stick_video(
        video_path=args.video,
        output_path=args.output,
        trajectory_path=args.trajectory,
        canvas_w=args.width,
        canvas_h=args.height,
        show_angles=not args.no_angles,
        show_grid=not args.no_grid,
        skip_frames=args.skip,
    )