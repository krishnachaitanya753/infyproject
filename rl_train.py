"""
Standalone PPO trainer for G1 motion tracking.
------------------------------------------------
Trains a PPO policy to physically execute a reference motion
extracted from a human video. Completely separate from the pipeline.

Usage:
    python rl_train.py --trajectory pipeline_output/trajectory.json

The trainer streams progress to a queue so the web UI can
display live reward curves without blocking.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ─── Default hyperparameters ──────────────────────────────────────────────────

PPO_DEFAULTS = dict(
    total_timesteps = 2_000_000,
    n_steps         = 2048,
    batch_size      = 256,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    learning_rate   = 3e-4,
    n_envs          = 4,
    log_interval    = 1,
    save_interval   = 100,
    video_interval  = 1_000_000,   # render a rollout video every N timesteps
    device          = "cpu",
    wandb_project   = "g1-motion-tracking",   # set to "" to disable wandb
    wandb_run_name  = "",                      # auto-generated if blank
)


# ─── Video rollout recorder ───────────────────────────────────────────────────

def record_video(model, traj_path: Path, output_path: Path, max_steps: int = 500):
    """
    Run one greedy episode with render_mode='rgb_array' and save as MP4.
    Runs in-process (fast, no subprocess needed).
    """
    try:
        import imageio
        from g1_env import G1MotionEnv

        env = G1MotionEnv(
            traj_path=traj_path,
            substeps=10,
            early_termination=True,
            render_mode="rgb_array",
        )
        obs, _ = env.reset()
        frames = []
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break
        env.close()

        if frames:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(output_path), frames, fps=50, quality=7)
            return str(output_path)
    except Exception as e:
        print(f"[Video] Could not record: {e}")
    return None


# ─── Progress callback ────────────────────────────────────────────────────────

class ProgressCallback:
    """
    SB3 callback that:
    - streams metrics via cb(stage=6, msg, data)
    - logs to WandB if enabled
    - records a video rollout every `video_interval` timesteps
    """

    def __init__(self, cb: Optional[Callable], total_timesteps: int,
                 log_interval: int, save_interval: int, video_interval: int,
                 output_dir: Path, traj_path: Path,
                 wandb_project: str, wandb_run_name: str, cfg: dict):
        self.cb               = cb
        self.total_ts         = total_timesteps
        self.log_interval     = log_interval
        self.save_interval    = save_interval
        self.video_interval   = video_interval
        self.output_dir       = output_dir
        self.traj_path        = traj_path
        self._update_count    = 0
        self._t0              = time.time()
        self._reward_history: list[float] = []
        self._last_video_ts   = 0

        # WandB setup
        self._wandb = None
        if wandb_project:
            try:
                import wandb
                run_name = wandb_run_name or f"g1-ppo-{time.strftime('%m%d-%H%M')}"
                self._wandb = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=cfg,
                    resume="allow",
                )
                print(f"[WandB] Logging to project '{wandb_project}' run '{run_name}'")
                print(f"[WandB] Dashboard: {self._wandb.url}")
            except Exception as e:
                print(f"[WandB] Init failed ({e}), continuing without WandB.")
                self._wandb = None

    def _emit(self, msg: str, data: dict = None):
        if self.cb:
            self.cb(6, msg, data or {})

    def on_rollout_end(self, model, n_calls: int, num_timesteps: int):
        self._update_count += 1
        upd = self._update_count

        if hasattr(model, 'ep_info_buffer') and len(model.ep_info_buffer) > 0:
            recent   = list(model.ep_info_buffer)[-20:]
            ep_rews  = [ep['r'] for ep in recent]
            ep_lens  = [ep['l'] for ep in recent]
        else:
            ep_rews = ep_lens = []

        mean_rew = float(np.mean(ep_rews)) if ep_rews else 0.0
        mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        self._reward_history.append(mean_rew)

        progress = round(num_timesteps / self.total_ts * 100, 1)
        elapsed  = round(time.time() - self._t0, 1)
        fps      = round(num_timesteps / max(elapsed, 1))

        # ── Console / SSE ────────────────────────────────────────────────────
        if upd % self.log_interval == 0:
            send_history = (upd % (self.log_interval * 5) == 0)
            self._emit(
                f"Update {upd} | steps {num_timesteps}/{self.total_ts} "
                f"| mean_rew {mean_rew:.3f} | ep_len {mean_len:.0f} | {elapsed:.0f}s",
                {
                    "update":      upd,
                    "timesteps":   num_timesteps,
                    "total_ts":    self.total_ts,
                    "mean_reward": round(mean_rew, 4),
                    "mean_ep_len": round(mean_len, 1),
                    "progress":    progress,
                    "fps":         fps,
                    "elapsed":     elapsed,
                    **({"reward_history": [round(r, 4) for r in self._reward_history[-200:]]}
                       if send_history else {}),
                },
            )

        # ── WandB metrics ─────────────────────────────────────────────────────
        if self._wandb is not None:
            log_dict = {
                "train/mean_reward":  mean_rew,
                "train/mean_ep_len":  mean_len,
                "train/progress_pct": progress,
                "train/fps":          fps,
                "train/elapsed_s":    elapsed,
            }
            # Add policy loss etc. if available
            if hasattr(model, 'logger') and model.logger is not None:
                for k, v in model.logger.name_to_value.items():
                    log_dict[f"sb3/{k}"] = v
            self._wandb.log(log_dict, step=num_timesteps)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if upd % self.save_interval == 0:
            ckpt = self.output_dir / f"ppo_update{upd:05d}.zip"
            model.save(str(ckpt))
            self._emit(f"Checkpoint saved: {ckpt.name}", {"checkpoint": str(ckpt)})

        # ── Video rollout ─────────────────────────────────────────────────────
        if (self.video_interval > 0 and
                num_timesteps - self._last_video_ts >= self.video_interval):
            self._last_video_ts = num_timesteps
            vid_path = self.output_dir / f"rollout_{num_timesteps:08d}.mp4"
            self._emit(f"Recording video rollout at {num_timesteps} steps...")
            saved = record_video(model, self.traj_path, vid_path)
            if saved:
                self._emit(f"Video saved: {vid_path.name}",
                           {"video": str(vid_path), "timesteps": num_timesteps})
                if self._wandb is not None:
                    try:
                        import wandb
                        self._wandb.log(
                            {"rollout_video": wandb.Video(str(vid_path), fps=50, format="mp4")},
                            step=num_timesteps,
                        )
                    except Exception as e:
                        print(f"[WandB] Video upload failed: {e}")

    def on_training_end(self, model, num_timesteps: int):
        elapsed = round(time.time() - self._t0, 1)
        path    = self.output_dir / "ppo_final.zip"
        model.save(str(path))

        # Final video
        if self.video_interval > 0:
            self._emit("Recording final video rollout...")
            vid_path = self.output_dir / "rollout_final.mp4"
            saved = record_video(model, self.traj_path, vid_path)
            if saved:
                self._emit(f"Final video saved: {vid_path.name}",
                           {"video": str(vid_path)})
                if self._wandb is not None:
                    try:
                        import wandb
                        self._wandb.log(
                            {"rollout_video": wandb.Video(str(vid_path), fps=50, format="mp4")},
                            step=num_timesteps,
                        )
                    except Exception as e:
                        print(f"[WandB] Final video upload failed: {e}")

        summary = {
            "total_timesteps":   num_timesteps,
            "training_time_s":   elapsed,
            "final_policy":      str(path),
            "reward_history":    [round(r, 4) for r in self._reward_history],
            "mean_final_reward": round(float(np.mean(self._reward_history[-10:])), 4)
                                  if self._reward_history else 0.0,
        }
        (self.output_dir / "ppo_result.json").write_text(json.dumps(summary, indent=2))

        if self._wandb is not None:
            self._wandb.summary.update({
                "final_reward":    summary["mean_final_reward"],
                "training_time_s": elapsed,
            })
            self._wandb.finish()

        self._emit(
            f"PPO training complete! {num_timesteps} steps in {elapsed}s. "
            f"Final mean reward: {summary['mean_final_reward']:.3f}",
            {**summary, "type": "done"},
        )
        return summary


# ─── SB3 callback wrapper ─────────────────────────────────────────────────────

def _make_sb3_callback(prog: ProgressCallback):
    from stable_baselines3.common.callbacks import BaseCallback

    class _CB(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._calls = 0

        def _on_rollout_end(self):
            self._calls += 1
            prog.on_rollout_end(self.model, self._calls, self.num_timesteps)

        def _on_step(self) -> bool:
            return True

    return _CB()


# ─── Training entry point ─────────────────────────────────────────────────────

def train_ppo(
    traj_path:  Path,
    output_dir: Path,
    cfg:        Optional[dict] = None,
    cb:         Optional[Callable] = None,
) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from g1_env import G1MotionEnv

    def emit(msg, data=None):
        if cb:
            cb(6, msg, data or {})

    cfg      = {**PPO_DEFAULTS, **(cfg or {})}
    total_ts = int(cfg["total_timesteps"])
    n_envs   = int(cfg["n_envs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    emit(f"Setting up G1 environment x{n_envs} envs...",
         {"cfg": {k: v for k, v in cfg.items() if not k.startswith("_")}})

    base_policy_path = cfg.get("base_policy_path") or os.environ.get("UNITREE_POLICY_PATH")
    if base_policy_path:
        emit(f"Using Unitree base policy: {base_policy_path}")
    else:
        emit("No Unitree checkpoint found — using nominal standing pose as base policy.")

    def _make_env():
        return G1MotionEnv(
            traj_path=traj_path,
            substeps=10,
            early_termination=True,
            base_policy_path=base_policy_path,
            render_mode=None,
        )

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(_make_env, n_envs=n_envs, vec_env_cls=vec_cls)

    emit(f"Building PPO policy (MlpPolicy, {cfg['device']})...")

    model = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        n_steps       = int(cfg["n_steps"]),
        batch_size    = int(cfg["batch_size"]),
        n_epochs      = int(cfg["n_epochs"]),
        gamma         = float(cfg["gamma"]),
        gae_lambda    = float(cfg["gae_lambda"]),
        clip_range    = float(cfg["clip_range"]),
        ent_coef      = float(cfg["ent_coef"]),
        vf_coef       = float(cfg["vf_coef"]),
        max_grad_norm = float(cfg["max_grad_norm"]),
        learning_rate = float(cfg["learning_rate"]),
        policy_kwargs = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
        device        = cfg["device"],
        verbose       = 0,
    )

    (output_dir / "ppo_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    emit(f"PPO ready. Starting {total_ts:,} timestep training run...",
         {"total_timesteps": total_ts, "obs_dim": 102, "act_dim": 29})

    prog   = ProgressCallback(
        cb=cb, total_timesteps=total_ts,
        log_interval=int(cfg["log_interval"]),
        save_interval=int(cfg["save_interval"]),
        video_interval=int(cfg["video_interval"]),
        output_dir=output_dir, traj_path=traj_path,
        wandb_project=cfg.get("wandb_project", ""),
        wandb_run_name=cfg.get("wandb_run_name", ""),
        cfg=cfg,
    )
    sb3_cb = _make_sb3_callback(prog)

    model.learn(total_timesteps=total_ts, callback=sb3_cb, progress_bar=False)

    result = prog.on_training_end(model, total_ts)
    vec_env.close()
    return result or {}


# ─── Async wrapper ────────────────────────────────────────────────────────────

def train_ppo_async(
    traj_path:  Path,
    output_dir: Path,
    cfg:        Optional[dict] = None,
) -> tuple[queue.Queue, threading.Thread]:
    q = queue.Queue()

    def _cb(stage, msg, data):
        q.put({"stage": stage, "msg": msg, "data": data or {}})

    def _run():
        try:
            train_ppo(traj_path, output_dir, cfg=cfg, cb=_cb)
        except Exception as e:
            q.put({"stage": 6, "msg": "ERROR", "data": {"error": str(e)}})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return q, t


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(description="PPO motion tracking for Unitree G1")
    ap.add_argument("--trajectory",      type=Path,  default=Path("pipeline_output/trajectory.json"))
    ap.add_argument("--output-dir",      type=Path,  default=Path("pipeline_output/rl"))
    ap.add_argument("--total-timesteps", type=int,   default=PPO_DEFAULTS["total_timesteps"])
    ap.add_argument("--n-envs",          type=int,   default=PPO_DEFAULTS["n_envs"])
    ap.add_argument("--n-steps",         type=int,   default=PPO_DEFAULTS["n_steps"])
    ap.add_argument("--batch-size",      type=int,   default=PPO_DEFAULTS["batch_size"])
    ap.add_argument("--lr",              type=float, default=PPO_DEFAULTS["learning_rate"])
    ap.add_argument("--device",          type=str,   default=PPO_DEFAULTS["device"])
    ap.add_argument("--video-interval",  type=int,   default=PPO_DEFAULTS["video_interval"],
                    help="Record video every N timesteps (0 = disable)")
    ap.add_argument("--wandb-project",   type=str,   default=PPO_DEFAULTS["wandb_project"],
                    help="WandB project name (empty string = disable)")
    ap.add_argument("--wandb-run-name",  type=str,   default="",
                    help="WandB run name (auto-generated if blank)")
    ap.add_argument("--no-wandb",        action="store_true",
                    help="Disable WandB logging")
    args = ap.parse_args()

    cfg = dict(
        total_timesteps = args.total_timesteps,
        n_envs          = args.n_envs,
        n_steps         = args.n_steps,
        batch_size      = args.batch_size,
        learning_rate   = args.lr,
        device          = args.device,
        video_interval  = args.video_interval,
        wandb_project   = "" if args.no_wandb else args.wandb_project,
        wandb_run_name  = args.wandb_run_name,
    )

    def _cb(stage, msg, data):
        if data.get("type") == "done":
            return
        print(f"[Stage {stage}] {msg}")

    result = train_ppo(args.trajectory, args.output_dir, cfg=cfg, cb=_cb)
    print("\n=== PPO Training Result ===")
    print(json.dumps({k: v for k, v in result.items() if k != "reward_history"}, indent=2))


if __name__ == "__main__":
    _cli()
