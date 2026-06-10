# Storyline → Video → Joint Angles → PPO Motion Tracking

An end-to-end pipeline that turns a **text storyline** into a **physically-trained Unitree G1 humanoid policy** that mimics the motion from an AI-generated reference video.

```
Storyline ──▶ Video Prompt ──▶ AI Video ──▶ Pose Extraction ──▶ Trajectory ──▶ PPO Motion Tracking
  (text)        (LLM)         (Veo/Kling)     (MediaPipe)        (10 joints)      (MuJoCo + RL)
```

Two independent halves:

| Half | What it does | Runtime | Where |
|------|--------------|---------|-------|
| **Pipeline** (stages 1–5) | storyline → stick-figure video + `trajectory.json` | minutes | any machine |
| **RL training** (stage 6) | `trajectory.json` → trained PPO policy | hours | GPU/many-core box |

The two are deliberately decoupled — one video produces one trajectory in minutes, while RL training chews on that trajectory for millions of simulation steps.

---

## How it works

### The pipeline (`pipeline.py`)

1. **Generate video prompt** — an LLM expands your storyline into a detailed, motion-rich video prompt.
2. **Generate video** — sent to a text-to-video provider (Veo by default; Kling/Runway/HF also supported).
3. **Extract joint angles** — MediaPipe Pose extracts body landmarks per frame and computes joint angles.
4. **Smooth & filter** — temporal smoothing removes jitter from the raw per-frame angles.
5. **Build output** — renders an annotated **stick-figure video** and writes **`trajectory.json`** (the reference motion).

### The RL trainer (`rl_train.py` + `g1_env.py`)

A PPO policy learns to physically execute the reference motion inside a **MuJoCo** simulation of the 29-DOF Unitree G1.

- **Observation (102-dim):** `q(29) | qdot(29) | q_ref_10(10) | q_base(29) | phase(1) | height(1) | pelvis_vel(3)`
  - `q_ref_10` is the **explicit video reference** for the 10 tracked joints this frame — the robot can *see* what it should be doing.
  - `q_base` is the balance target for the 19 untracked joints (default: nominal G1 standing pose).
- **Action (29-dim):** target joint angles for all DOFs, fed to a per-joint **PD position controller** that produces torques.
- **Reward (DeepMimic-style, strictly positive per step):**
  ```
  r = 0.50·r_pose + 0.10·r_vel + 0.20·r_balance + 0.20·r_alive − tiny_energy
  ```
  - `r_pose = exp(-2·‖q−q_ref‖²)` on the 10 tracked joints
  - `r_balance` keeps the 19 untracked joints near the base pose
  - `r_alive` is a flat survival bonus; falling triggers a one-time penalty
  - All imitation terms are in `[0,1]` and weights sum to `1.0`, so **every upright step is rewarded** — the agent's dominant strategy is to stay balanced *and* track the motion.

**10 tracked joints:** left/right of `hip_pitch`, `knee`, `ankle_pitch`, `shoulder_pitch`, `elbow`.

---

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/krishnachaitanya753/infyproject.git
cd infyproject
uv sync
```

Create a `.env` for the pipeline (not needed for RL-only use):

```env
# Pick a video provider: veo | fal | runway | hf
VIDEO_PROVIDER=veo
GOOGLE_API_KEY=your_key      # for Veo
# FAL_API_KEY=...            # for Kling via fal.ai
# RUNWAY_API_KEY=...         # for Runway
NVIDIA_API_KEY=your_key      # for the LLM prompt generator
```

> **Robot assets** (`robot_assets/unitree_g1/`) ship in the repo — the MuJoCo XML + meshes needed for simulation. No external download required.

---

## Usage

### Web UI (everything in one place)

```bash
uv run python app.py
```

Open `http://localhost:8000` (binds `0.0.0.0`, so it's reachable over Tailscale/LAN). Enter a storyline → watch the 5 stages stream live → download the stick-figure video and trajectory → hit **Train PPO Policy** to kick off RL with a live reward curve.

### Pipeline only (CLI)

```bash
uv run python pipeline.py --storyline "A person waves, then does a slow squat."
# or skip generation and use your own clip:
uv run python pipeline.py --storyline "..." --video my_clip.mp4
```

Outputs land in `pipeline_output/` (`stick_figure.mp4`, `trajectory.json`).

### RL training only (CLI)

```bash
uv run python rl_train.py \
  --trajectory pipeline_output/trajectory.json \
  --total-timesteps 10000000 \
  --n-envs 8 \
  --device cpu \
  --wandb-project g1-motion-tracking
```

| Flag | Default | Notes |
|------|---------|-------|
| `--total-timesteps` | 2,000,000 | total env steps |
| `--n-envs` | 4 | parallel MuJoCo envs (set to your CPU core count) |
| `--device` | cpu | MuJoCo is CPU-bound; GPU barely helps a small MLP |
| `--video-interval` | 1,000,000 | render a rollout MP4 every N steps (logged to wandb) |
| `--wandb-project` | g1-motion-tracking | metrics + rollout videos; `--no-wandb` to disable |

Checkpoints save to `pipeline_output/rl/` (`ppo_update000NN.zip`, `ppo_final.zip`).

**Watch `ep_len`, not just reward** — rising episode length means the robot is staying upright longer and genuinely learning the motion.

---

## Project layout

```
pipeline.py        Stages 1–5: storyline → trajectory.json
g1_env.py          MuJoCo gymnasium env (102-dim obs, 29-DOF action, motion-tracking reward)
rl_train.py        Standalone PPO trainer (SB3) with wandb + rollout-video logging
app.py             FastAPI web UI + SSE streaming for both halves
static/index.html  Frontend (stage tracker, prompt preview, reward chart)
robot_assets/      Unitree G1 MuJoCo XML + meshes
pipeline_output/   Generated videos, trajectory.json, RL checkpoints
extra/             Standalone scripts and demo clips
```

---

## Notes & tips

- **MuJoCo is CPU-bound.** Throughput scales with `--n-envs`, not GPU. Set `--n-envs` to your core count.
- **Cleanest extraction** comes from videos with one person, full body in frame, plain background, static camera, slow exaggerated motion.
- **Headless by design** — RL training never opens a renderer (rollout videos are written off-screen), so it's safe to run over a remote/Tailscale link with minimal bandwidth.
- **Swap in a real balance controller** — set `UNITREE_POLICY_PATH` to a Unitree G1 checkpoint and the env will use it for the 19 untracked joints instead of the static standing pose.
