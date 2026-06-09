"""
G1 PPO Environment — Split-signal motion tracking
--------------------------------------------------
Architecture:

    Observation (102-dim)
      ├── q_current    (29)  where the robot is right now
      ├── qdot_current (29)  how fast joints are moving
      ├── q_ref_10     (10)  reference angles from video  ← 10 extracted joints
      ├── q_base       (29)  base policy target (balance)
      ├── phase         (1)  position in clip [0,1]
      ├── pelvis_h      (1)  pelvis height
      └── pelvis_vel    (3)  pelvis linear velocity

    Action (29-dim) — PPO controls all 29 DOF
      Tracked 10 joints  → reward pushes them toward q_ref_10 (motion from video)
      Untracked 19 joints→ reward pushes them toward q_base   (balance/standing)

    PD controller converts 29 joint targets → torques → MuJoCo physics

The 10 reference angles are explicit in the observation so the PPO
can directly see what each tracked joint should be doing this frame.
The base policy provides the 19 balance targets — default is the G1
nominal standing pose; swap in Unitree's checkpoint via UNITREE_POLICY_PATH.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE  = Path(__file__).parent
G1_XML = _HERE / "robot_assets/unitree_g1/g1_mocap_29dof.xml"

# ─── Joint layout (must match XML actuator order) ────────────────────────────

JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",         "waist_roll_joint",        "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",   "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",  "right_wrist_pitch_joint","right_wrist_yaw_joint",
]
NUM_DOF = len(JOINT_NAMES)   # 29

# MediaPipe joint key → index in JOINT_NAMES
TRAJ_TO_IDX: dict[str, int] = {
    "left_hip_pitch":       JOINT_NAMES.index("left_hip_pitch_joint"),
    "right_hip_pitch":      JOINT_NAMES.index("right_hip_pitch_joint"),
    "left_knee":            JOINT_NAMES.index("left_knee_joint"),
    "right_knee":           JOINT_NAMES.index("right_knee_joint"),
    "left_ankle_pitch":     JOINT_NAMES.index("left_ankle_pitch_joint"),
    "right_ankle_pitch":    JOINT_NAMES.index("right_ankle_pitch_joint"),
    "left_shoulder_pitch":  JOINT_NAMES.index("left_shoulder_pitch_joint"),
    "right_shoulder_pitch": JOINT_NAMES.index("right_shoulder_pitch_joint"),
    "left_elbow":           JOINT_NAMES.index("left_elbow_joint"),
    "right_elbow":          JOINT_NAMES.index("right_elbow_joint"),
}
TRACKED_IDXS: list[int] = sorted(TRAJ_TO_IDX.values())
UNTRACKED_IDXS: list[int] = [i for i in range(NUM_DOF) if i not in TRACKED_IDXS]

# ─── G1 nominal standing pose ─────────────────────────────────────────────────
# Joint angles [rad] for a stable standing posture.
# Slightly bent knees/ankles for stability; everything else neutral.

_STAND: dict[str, float] = {
    "left_hip_pitch_joint":   -0.10,
    "left_knee_joint":         0.30,
    "left_ankle_pitch_joint": -0.20,
    "right_hip_pitch_joint":  -0.10,
    "right_knee_joint":        0.30,
    "right_ankle_pitch_joint":-0.20,
}
G1_STAND_POSE = np.array(
    [_STAND.get(j, 0.0) for j in JOINT_NAMES], dtype=np.float64
)

# ─── PD gains ─────────────────────────────────────────────────────────────────

KP = np.array([
    150,  60,  60,    # left  hip p/r/y
    150,  60,  30,    # left  knee, ankle p/r
    150,  60,  60,    # right hip p/r/y
    150,  60,  30,    # right knee, ankle p/r
     80,  40,  40,    # waist y/r/p
     60,  40,  30,    # left  shoulder p/r/y
     40,  15,  15, 15,# left  elbow + wrist r/p/y
     60,  40,  30,    # right shoulder p/r/y
     40,  15,  15, 15,# right elbow + wrist r/p/y
], dtype=np.float64)

KD = np.array([
    5.0, 2.0, 2.0,
    5.0, 2.0, 1.0,
    5.0, 2.0, 2.0,
    5.0, 2.0, 1.0,
    3.0, 2.0, 2.0,
    2.0, 1.5, 1.0,
    1.5, 0.5, 0.5, 0.5,
    2.0, 1.5, 1.0,
    1.5, 0.5, 0.5, 0.5,
], dtype=np.float64)

# Torque limits [Nm] — Unitree G1 spec
TORQUE_LIMITS = np.array([
     88,  50,  50,    # left  hip p/r/y
    139,  50,  50,    # left  knee, ankle p/r
     88,  50,  50,    # right hip p/r/y
    139,  50,  50,    # right knee, ankle p/r
     88,  50,  50,    # waist y/r/p
     25,  25,  25,    # left  shoulder p/r/y
     25,  10,  10, 10,# left  elbow + wrist r/p/y
     25,  25,  25,    # right shoulder p/r/y
     25,  10,  10, 10,# right elbow + wrist r/p/y
], dtype=np.float64)


# ─── Reward weights ────────────────────────────────────────────────────────────

W_POSE   = 0.65   # track reference angles on 10 joints
W_VEL    = 0.10   # track reference velocities
W_ALIVE  = 0.15   # alive bonus per step
W_ENERGY = 0.10   # penalise torque

MIN_HEIGHT = 0.45   # pelvis z [m] — below this = fell
SIM_DT     = 0.0005 # inner MuJoCo timestep  (2000 Hz) — finer dt prevents NaN instability
CTRL_HZ    = 50     # policy control frequency


# ─── Base Policy ─────────────────────────────────────────────────────────────

class BasePolicy:
    """
    Unitree G1 base locomotion/standing controller.

    Default behaviour: returns the nominal standing pose for all joints.
    This keeps the robot balanced while the residual PPO learns the motion.

    To use Unitree's real policy, provide the path to the checkpoint (.pt)
    trained in Isaac Gym / IsaacLab. The checkpoint must export a callable
    that accepts the standard proprioceptive observation and returns joint
    position targets (29-dim, rad).

    Args:
        checkpoint_path: path to a .pt file with Unitree's policy weights.
                         If None (default) the standing pose is used.
    """

    def __init__(self, checkpoint_path: Optional[str | Path] = None):
        self._net = None
        self._use_stand = True

        if checkpoint_path is not None:
            self._load(Path(checkpoint_path))

    def _load(self, path: Path):
        try:
            import torch
            data = torch.load(str(path), map_location="cpu")
            # Accept common checkpoint formats:
            #   {"model": state_dict, ...}  or bare state_dict
            if isinstance(data, dict) and "model" in data:
                state = data["model"]
            else:
                state = data

            # Try to infer architecture from checkpoint keys and build net
            # This is a best-effort loader for Unitree Isaac-Gym style policies
            # (obs_dim → [512, 256, 128] → 29)
            import torch.nn as nn
            obs_dim = 48  # standard Unitree G1 obs (ang_vel + gravity + cmd + q + qdot + prev_a)
            net = nn.Sequential(
                nn.Linear(obs_dim, 512), nn.ELU(),
                nn.Linear(512, 256),     nn.ELU(),
                nn.Linear(256, 128),     nn.ELU(),
                nn.Linear(128, NUM_DOF),
            )
            try:
                net.load_state_dict(state, strict=False)
                net.eval()
                self._net = net
                self._use_stand = False
                print(f"[BasePolicy] Loaded Unitree policy from {path}")
            except Exception as e:
                print(f"[BasePolicy] Could not load checkpoint ({e}), using standing pose.")
        except Exception as e:
            print(f"[BasePolicy] Checkpoint load failed ({e}), using standing pose.")

    def __call__(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Return base target joint positions (29,) in radians.

        obs_vec: full environment observation (93-dim). Sliced to
                 proprioceptive subset when a real Unitree policy is loaded.
        """
        if self._net is not None:
            import torch
            # Extract proprioceptive obs (first 48 dims or trim/pad)
            prop = obs_vec[:48] if len(obs_vec) >= 48 else np.pad(obs_vec, (0, 48 - len(obs_vec)))
            with torch.no_grad():
                action = self._net(torch.from_numpy(prop.astype(np.float32)))
            return action.numpy().astype(np.float64)

        # Default: nominal standing pose
        return G1_STAND_POSE.copy()


# ─── Reference motion loader ─────────────────────────────────────────────────

def load_reference(traj_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load trajectory.json → (q_ref T×29, qvel_ref T×29, fps)."""
    data = json.loads(Path(traj_path).read_text())
    T    = len(data)
    q_ref    = np.zeros((T, NUM_DOF), dtype=np.float32)
    qvel_ref = np.zeros((T, NUM_DOF), dtype=np.float32)
    for t, frame in enumerate(data):
        q    = frame.get("q_ref",    {})
        qdot = frame.get("qvel_ref", {})
        for key, idx in TRAJ_TO_IDX.items():
            q_ref[t,    idx] = float(q.get(key,    0.0))
            qvel_ref[t, idx] = float(qdot.get(key, 0.0))
    fps = float(data[0].get("fps", 30.0)) if data else 30.0
    return q_ref, qvel_ref, fps


# ─── Environment ─────────────────────────────────────────────────────────────

class G1MotionEnv(gym.Env):
    """
    Observation (102-dim):
        q_current    (29)  current joint angles
        qdot_current (29)  current joint velocities
        q_ref_10     (10)  reference angles for tracked joints (from video)
        q_base       (29)  base policy target (balance/standing)
        phase         (1)  position in clip [0,1], cycles
        pelvis_h      (1)  pelvis z height [m]
        pelvis_vel    (3)  pelvis linear velocity [m/s]
        = 102 total

    Action (29-dim):
        Target joint angles for all 29 DOF, normalised to [-1, 1].
        Mapped within joint limits. PD controller applies torques.
        PPO learns:
          - tracked 10: match q_ref_10  (motion from video)
          - untracked 19: match q_base  (balance, via reward)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        traj_path:         str | Path,
        xml_path:          str | Path = G1_XML,
        base_policy_path:  Optional[str | Path] = None,
        substeps:          int  = 10,
        early_termination: bool = True,
        render_mode:       Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.substeps    = substeps
        self.early_term  = early_termination
        self._ctrl_dt    = SIM_DT * substeps   # seconds per policy step

        # Check for Unitree policy from env var if not passed directly
        if base_policy_path is None:
            base_policy_path = os.environ.get("UNITREE_POLICY_PATH")

        self.base_policy = BasePolicy(base_policy_path)

        # ── MuJoCo ───────────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = SIM_DT   # apply our timestep — XML default is ignored
        self.data  = mujoco.MjData(self.model)

        # Scale actuator gear: ctrl=±1 → ±TORQUE_LIMITS[i] Nm
        for i, jname in enumerate(JOINT_NAMES):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            if aid >= 0:
                self.model.actuator_gear[aid, 0]  = TORQUE_LIMITS[i]
                self.model.actuator_ctrlrange[aid] = np.array([-1.0, 1.0])

        # qpos/qvel offsets: freejoint = 7 qpos, 6 qvel
        self._qpos0 = 7
        self._qvel0 = 6

        # Joint limits
        self.q_lo = np.zeros(NUM_DOF)
        self.q_hi = np.zeros(NUM_DOF)
        for i, jname in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self.q_lo[i] = self.model.jnt_range[jid, 0]
            self.q_hi[i] = self.model.jnt_range[jid, 1]

        # ── Reference motion ─────────────────────────────────────────────
        self.q_ref, self.qvel_ref, self.fps = load_reference(traj_path)
        self.T_ref    = len(self.q_ref)
        self._ref_dt  = 1.0 / self.fps
        self._clip_len = self.T_ref * self._ref_dt   # seconds

        # ── Spaces ───────────────────────────────────────────────────────
        # obs: q(29) + qdot(29) + q_ref_10(10) + q_base(29) + phase(1) + h(1) + pvel(3) = 102
        obs_dim = NUM_DOF + NUM_DOF + len(TRACKED_IDXS) + NUM_DOF + 1 + 1 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Full 29-DOF action — normalised to [-1,1], mapped within joint limits
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_DOF,), dtype=np.float32
        )
        # Cache mid/scale for action denormalisation
        self.q_mid   = 0.5 * (self.q_lo + self.q_hi)
        self.q_scale = 0.5 * (self.q_hi - self.q_lo)

        self._ref_time   = 0.0
        self._step_count = 0
        self._renderer   = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _q(self)    -> np.ndarray:
        return self.data.qpos[self._qpos0: self._qpos0 + NUM_DOF].copy()

    def _qdot(self) -> np.ndarray:
        return self.data.qvel[self._qvel0: self._qvel0 + NUM_DOF].copy()

    def _pelvis_z(self)  -> float:
        return float(self.data.qpos[2])

    def _pelvis_vel(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()

    def _ref_at(self, t_sec: float) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolated reference at time t (cycles through clip)."""
        t_norm = (t_sec % self._clip_len) / self._clip_len
        idx_f  = t_norm * (self.T_ref - 1)
        lo     = int(idx_f)
        hi     = min(lo + 1, self.T_ref - 1)
        alpha  = idx_f - lo
        q_r  = (1 - alpha) * self.q_ref[lo]    + alpha * self.q_ref[hi]
        qd_r = (1 - alpha) * self.qvel_ref[lo] + alpha * self.qvel_ref[hi]
        return q_r.astype(np.float64), qd_r.astype(np.float64)

    def _denorm(self, a: np.ndarray) -> np.ndarray:
        """[-1,1] action → actual joint angles within limits."""
        return self.q_mid + np.clip(a, -1.0, 1.0) * self.q_scale

    def _get_obs(self, q_base: np.ndarray) -> np.ndarray:
        q      = self._q()
        qdot   = self._qdot()
        q_r, _ = self._ref_at(self._ref_time)
        # Only the 10 tracked joint references go into obs — explicit signal from video
        q_ref_10 = q_r[TRACKED_IDXS]
        phase    = np.array([(self._ref_time % self._clip_len) / self._clip_len])
        height   = np.array([self._pelvis_z()])
        pvel     = self._pelvis_vel()
        # Layout: q(29) | qdot(29) | q_ref_10(10) | q_base(29) | phase(1) | h(1) | pvel(3)
        return np.concatenate([q, qdot, q_ref_10, q_base, phase, height, pvel]).astype(np.float32)

    def _reward(
        self,
        q:       np.ndarray,
        qdot:    np.ndarray,
        q_r:     np.ndarray,
        qdot_r:  np.ndarray,
        q_base:  np.ndarray,
        torques: np.ndarray,
    ) -> tuple[float, dict]:

        # Tracked joints: compare against motion reference
        ti   = TRACKED_IDXS
        pose_err = float(np.sum((q[ti] - q_r[ti]) ** 2))
        vel_err  = float(np.sum((qdot[ti] - qdot_r[ti]) ** 2))

        # Untracked joints: compare against base policy (balance target)
        uti  = UNTRACKED_IDXS
        bal_err = float(np.sum((q[uti] - q_base[uti]) ** 2))

        r_pose  = float(np.exp(-2.0  * pose_err))
        r_vel   = float(np.exp(-0.1  * vel_err))
        r_bal   = float(np.exp(-1.0  * bal_err))   # stay close to base on untracked
        r_alive = 1.0
        r_energy = -1e-4 * float(np.sum(torques ** 2))

        # Blend: tracked joints dominate, balance stabilises untracked joints
        total = (W_POSE   * r_pose  +
                 W_VEL    * r_vel   +
                 W_ALIVE  * r_alive * r_bal +   # alive only if balanced too
                 W_ENERGY * r_energy)

        info = {
            "r_pose":        round(float(r_pose),  4),
            "r_vel":         round(float(r_vel),   4),
            "r_balance":     round(float(r_bal),   4),
            "r_energy":      round(float(r_energy),4),
            "pose_err_deg":  round(float(np.degrees(np.sqrt(pose_err / max(1, len(ti))))), 2),
            "bal_err_deg":   round(float(np.degrees(np.sqrt(bal_err  / max(1, len(uti))))), 2),
        }
        return float(total), info

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Start at the standing pose + small random perturbation
        q0 = G1_STAND_POSE.copy()
        if seed is not None:
            rng = np.random.default_rng(seed)
            q0 += rng.normal(0, 0.02, NUM_DOF)

        self.data.qpos[self._qpos0: self._qpos0 + NUM_DOF] = q0
        mujoco.mj_forward(self.model, self.data)

        self._ref_time   = 0.0
        self._step_count = 0

        return self._get_obs(G1_STAND_POSE), {}

    def step(self, action: np.ndarray):
        # Base policy: what the balance controller wants for all 29 joints
        obs_for_base = self._get_obs(G1_STAND_POSE)
        q_base   = np.clip(self.base_policy(obs_for_base), self.q_lo, self.q_hi)

        # PPO outputs full 29-DOF target (normalised -1..1 → actual angles)
        q_target = self._denorm(action)

        # Apply PD control over substeps
        avg_torques = np.zeros(NUM_DOF)
        for _ in range(self.substeps):
            q    = self._q()
            qdot = self._qdot()
            torques = KP * (q_target - q) - KD * qdot
            torques = np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)
            self.data.ctrl[:] = torques / TORQUE_LIMITS
            mujoco.mj_step(self.model, self.data)
            avg_torques += np.abs(torques)
            # Catch NaN/Inf — reset immediately to avoid corrupting rollout
            if not np.isfinite(self.data.qpos).all():
                obs, _ = self.reset()
                return obs, 0.0, True, False, {"nan_reset": True}
        avg_torques /= self.substeps

        self._ref_time   += self._ctrl_dt
        self._step_count += 1

        q    = self._q()
        qdot = self._qdot()
        q_r, qdot_r = self._ref_at(self._ref_time)

        reward, info = self._reward(q, qdot, q_r, qdot_r, q_base, avg_torques)
        info["q_base_used"] = not self.base_policy._use_stand  # True if real Unitree policy

        fell      = self._pelvis_z() < MIN_HEIGHT
        clip_done = self._ref_time >= self._clip_len
        terminated = (fell and self.early_term) or clip_done
        truncated  = False

        info["fell"]      = bool(fell)
        info["clip_done"] = bool(clip_done)
        info["step"]      = self._step_count

        obs = self._get_obs(q_base)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
