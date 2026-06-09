# G1 PPO Training — Windows + uv Setup Guide
# ============================================
# No WSL, no conda, no pip — pure uv.
# Run every block in PowerShell (not CMD).


# ─── 0. Prerequisites ─────────────────────────────────────────────────────────
# Install uv if you haven't:
#   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Then restart PowerShell.


# ─── 1. Create project and virtual environment ────────────────────────────────

cd path\to\your\project          # folder containing g1_29dof.urdf, g1.xml,
                                 # g1_reference_motion.json, g1_ppo\

uv init g1_training              # creates g1_training\ with pyproject.toml
cd g1_training

uv venv --python 3.10            # MuJoCo + rsl_rl work best on 3.10
.venv\Scripts\activate           # activate the venv


# ─── 2. Install all dependencies ──────────────────────────────────────────────
# Do them in this order — mujoco before rsl_rl matters on Windows.

uv pip install mujoco==3.1.6
uv pip install gymnasium==0.29.1
uv pip install numpy==1.26.4
uv pip install scipy
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# ↑ If you have NO GPU, use this line instead:
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

uv pip install tensorboard
uv pip install rsl-rl


# ─── 3. Verify MuJoCo can see your XML ────────────────────────────────────────
# Quick sanity check — should print model info without error:

python -c "
import mujoco, pathlib
m = mujoco.MjModel.from_xml_path('g1.xml')
print(f'OK — {m.njnt} joints, {m.nbody} bodies')
"

# Expected output:
#   OK — 29 joints, N bodies


# ─── 4. Project layout ────────────────────────────────────────────────────────
# Your folder should look like this before training:
#
# g1_training\
# ├── g1.xml                        ← MuJoCo XML of the robot
# ├── g1_29dof.urdf                 ← original URDF (not used at runtime)
# ├── g1_reference_motion.json      ← output of stickfigure.py
# ├── meshes\                       ← STL meshes (same folder as XML expects)
# ├── g1_ppo\
# │   ├── envs\
# │   │   └── g1_env.py
# │   ├── rsl_rl_cfg\
# │   │   └── g1_ppo_cfg.py
# │   └── scripts\
# │       └── train.py              ← generated below (Step 5)
# └── logs\                         ← created automatically by runner


# ─── 5. Create train.py ───────────────────────────────────────────────────────
# Save this as g1_ppo\scripts\train.py

===============================================================
FILE: g1_ppo\scripts\train.py
===============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner

from g1_ppo.envs.g1_env import G1Env
from g1_ppo.rsl_rl_cfg.g1_ppo_cfg import G1TrainCfg, ActorCriticCfg, PPOCfg


# ── rsl_rl VecEnv shim ──────────────────────────────────────────────────────
# rsl_rl's OnPolicyRunner expects a vectorised env that returns tensors.
# This thin wrapper bridges our numpy G1Env to that interface.

class G1VecEnv:
    """Single-env torch wrapper compatible with rsl_rl OnPolicyRunner."""

    def __init__(self, env: G1Env, device: str = "cpu"):
        self.env      = env
        self.device   = device
        self.num_envs = 1
        self.num_obs  = env.num_obs
        self.num_privileged_obs = None   # no asymmetric AC for now
        self.num_actions         = env.num_actions
        self.max_episode_length  = env.episode_len
        self.episode_length_buf  = torch.zeros(1, device=device, dtype=torch.long)

    def get_observations(self):
        return self._last_obs, None   # obs, privileged_obs

    def reset(self):
        obs_np       = self.env.reset()
        self._last_obs = self._to_tensor(obs_np)
        self.episode_length_buf[:] = 0
        return self._last_obs, None

    def step(self, actions: torch.Tensor):
        act_np                       = actions.squeeze(0).cpu().numpy()
        obs_np, rew, term, trunc, info = self.env.step(act_np)

        self._last_obs               = self._to_tensor(obs_np)
        rewards                      = torch.tensor([[rew]], device=self.device, dtype=torch.float32)
        dones                        = torch.tensor([[term or trunc]], device=self.device, dtype=torch.bool)
        self.episode_length_buf     += 1

        if term or trunc:
            obs_np         = self.env.reset()
            self._last_obs = self._to_tensor(obs_np)
            self.episode_length_buf[:] = 0

        return self._last_obs, rewards, dones, info

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0).to(self.device, dtype=torch.float32)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",          default="g1.xml")
    parser.add_argument("--trajectory",   default="g1_reference_motion.json")
    parser.add_argument("--log_dir",      default="logs")
    parser.add_argument("--resume",       default="",        help="path to checkpoint .pt to resume from")
    parser.add_argument("--max_iter",     type=int, default=10_000)
    parser.add_argument("--imitation_w",  type=float, default=1.0,
                        help="Starting imitation weight (1=pure imitation, 0=pure RL)")
    parser.add_argument("--decay",        type=float, default=5e-6,
                        help="Imitation weight decay per step")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device : {args.device}")
    print(f"XML    : {args.xml}")
    print(f"Traj   : {args.trajectory}")

    # Build env
    raw_env = G1Env(
        xml_path          = args.xml,
        trajectory_path   = args.trajectory,
        imitation_weight  = args.imitation_w,
        imitation_decay   = args.decay,
        enable_domain_rand= True,
    )
    env = G1VecEnv(raw_env, device=args.device)

    # Build config
    train_cfg            = G1TrainCfg()
    train_cfg.max_iterations = args.max_iter

    # Runner — this is the rsl_rl OnPolicyRunner
    runner = OnPolicyRunner(
        env,
        train_cfg.__dict__,          # runner reads dict form
        log_dir    = args.log_dir,
        device     = args.device,
    )

    if args.resume:
        runner.load(args.resume)
        print(f"Resumed from: {args.resume}")

    runner.learn(
        num_learning_iterations = args.max_iter,
        init_at_random_ep_len   = True,
    )


if __name__ == "__main__":
    main()

===============================================================


# ─── 6. Run training ──────────────────────────────────────────────────────────

# Phase 1 — Pure imitation (learn the motion first, ~3k iterations)
python g1_ppo\scripts\train.py `
    --xml g1.xml `
    --trajectory g1_reference_motion.json `
    --imitation_w 1.0 `
    --decay 0.0 `
    --max_iter 3000 `
    --log_dir logs\phase1

# Phase 2 — Decay to stability (imitation fades, RL takes over)
# Resume from best phase 1 checkpoint (check logs\phase1\ for the .pt file)
python g1_ppo\scripts\train.py `
    --xml g1.xml `
    --trajectory g1_reference_motion.json `
    --imitation_w 1.0 `
    --decay 5e-6 `
    --max_iter 7000 `
    --resume logs\phase1\model_3000.pt `
    --log_dir logs\phase2


# ─── 7. Monitor training ──────────────────────────────────────────────────────

# In a second PowerShell window (same venv activated):
tensorboard --logdir logs

# Then open browser: http://localhost:6006
# Key metrics to watch:
#   Train/mean_reward       — should trend upward
#   Train/mean_episode_len  — longer = robot stays alive longer
#   Train/value_loss        — should decrease and stabilise
#   Train/surrogate_loss    — should decrease
#   imitation_weight        — watch it decay in phase 2 logs (printed to console)


# ─── 8. Resume / export checkpoint ───────────────────────────────────────────

# Checkpoints are saved as:  logs\phase2\model_XXXX.pt  every 100 iterations
# To export policy weights only (for deployment):

python -c "
import torch
ckpt = torch.load('logs\phase2\model_7000.pt', map_location='cpu')
torch.save(ckpt['model_state_dict'], 'g1_policy_weights.pt')
print('Exported policy weights.')
"


# ─── 9. Troubleshooting ───────────────────────────────────────────────────────

# ERROR: 'Joint X not found in MuJoCo model'
#   → Your XML joint names differ from the URDF. Run this to list XML joints:
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('g1.xml')
for i in range(m.njnt):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
"
#   Then update JOINT_NAMES in g1_env.py to match exactly.

# ERROR: DLL load failed / mujoco import error on Windows
#   → uv pip install mujoco==3.1.6  (pin the version)
#   → Make sure Visual C++ Redistributable 2019+ is installed.
#     Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

# ERROR: CUDA out of memory
#   → Add --device cpu to the train command (single env is fine on CPU)

# REWARD stuck near 0 after 500 iterations
#   → Increase --imitation_w decay to 0.0 (keep phase 1 longer)
#   → Check your trajectory JSON has valid q_ref entries (non-zero values)

# Robot falls immediately
#   → Your XML starting height may be wrong. Check g1.xml freejoint z position.
#     In g1_env.py reset(), change: self.data.qpos[2] = 0.78
#     to match actual G1 standing height in your XML.
