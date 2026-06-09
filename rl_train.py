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
    total_timesteps = 2_000_000,   # ~30 min on CPU, ~5 min on GPU
    n_steps         = 2048,        # rollout length per env
    batch_size      = 256,
    n_epochs        = 10,          # PPO update epochs
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    learning_rate   = 3e-4,
    n_envs          = 4,           # parallel envs (set 1 on low-end machines)
    log_interval    = 10,          # log every N policy updates
    save_interval   = 100,         # save checkpoint every N updates
    device          = "auto",      # "auto" | "cpu" | "cuda"
)


# ─── Progress callback ────────────────────────────────────────────────────────

class ProgressCallback:
    """
    SB3 callback that streams training metrics via the pipeline cb convention:
        cb(stage=6, message, data_dict)
    """

    def __init__(self, cb: Optional[Callable], total_timesteps: int,
                 log_interval: int, save_interval: int, output_dir: Path):
        self.cb               = cb
        self.total_ts         = total_timesteps
        self.log_interval     = log_interval
        self.save_interval    = save_interval
        self.output_dir       = output_dir
        self._update_count    = 0
        self._t0              = time.time()
        self._reward_history: list[float] = []

    def _emit(self, msg: str, data: dict = None):
        if self.cb:
            self.cb(6, msg, data or {})

    def on_rollout_end(self, model, n_calls: int, num_timesteps: int):
        """Called after each rollout (before PPO update)."""
        self._update_count += 1
        upd = self._update_count

        # Compute mean episode reward from rollout buffer
        ep_rews = []
        for info in getattr(model, '_last_episode_starts', []):
            pass  # SB3 stores episode info differently; we pull from locals

        # Safer: read from ep_info_buffer
        if hasattr(model, 'ep_info_buffer') and len(model.ep_info_buffer) > 0:
            recent = list(model.ep_info_buffer)[-20:]
            ep_rews = [ep['r'] for ep in recent]

        mean_rew  = float(np.mean(ep_rews))  if ep_rews else 0.0
        mean_len  = float(np.mean([ep['l'] for ep in (list(model.ep_info_buffer)[-20:] if hasattr(model,'ep_info_buffer') and model.ep_info_buffer else [])])) if ep_rews else 0.0
        self._reward_history.append(mean_rew)

        progress  = round(num_timesteps / self.total_ts * 100, 1)
        elapsed   = round(time.time() - self._t0, 1)
        fps       = round(num_timesteps / max(elapsed, 1))

        if upd % self.log_interval == 0:
            # Send reward_history only every 5× log_interval to reduce network traffic
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
                    # Only send full history occasionally — keeps SSE payload tiny
                    **({"reward_history": [round(r, 4) for r in self._reward_history[-200:]]}
                       if send_history else {}),
                },
            )

        if upd % self.save_interval == 0:
            ckpt = self.output_dir / f"ppo_update{upd:05d}.zip"
            model.save(str(ckpt))
            self._emit(f"Checkpoint saved: {ckpt.name}", {"checkpoint": str(ckpt)})

    def on_training_end(self, model, num_timesteps: int):
        elapsed = round(time.time() - self._t0, 1)
        path    = self.output_dir / "ppo_final.zip"
        model.save(str(path))
        summary = {
            "total_timesteps":  num_timesteps,
            "training_time_s":  elapsed,
            "final_policy":     str(path),
            "reward_history":   [round(r, 4) for r in self._reward_history],
            "mean_final_reward": round(float(np.mean(self._reward_history[-10:])), 4)
                                  if self._reward_history else 0.0,
        }
        (self.output_dir / "ppo_result.json").write_text(json.dumps(summary, indent=2))
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
    """
    Train a PPO policy on the G1 motion tracking environment.

    cb(stage, message, data) — same signature used throughout the pipeline.
    All RL events use stage=6.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.env_util import make_vec_env

    from g1_env import G1MotionEnv

    def emit(msg, data=None):
        if cb:
            cb(6, msg, data or {})

    cfg       = {**PPO_DEFAULTS, **(cfg or {})}
    total_ts  = int(cfg["total_timesteps"])
    n_envs    = int(cfg["n_envs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    emit(f"Setting up G1 environment x{n_envs} envs...",
         {"cfg": {k: v for k, v in cfg.items() if not k.startswith("_")}})

    # Build vectorised env
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
            render_mode=None,   # headless — no display, no network load
        )

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(_make_env, n_envs=n_envs, vec_env_cls=vec_cls)

    emit(f"Building PPO policy (MlpPolicy, {cfg['device']})...")

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
    )

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
        policy_kwargs = policy_kwargs,
        device        = cfg["device"],
        verbose       = 0,
    )

    (output_dir / "ppo_config.json").write_text(json.dumps(cfg, indent=2))
    emit(f"PPO ready. Starting {total_ts:,} timestep training run...",
         {"total_timesteps": total_ts, "obs_dim": 102, "act_dim": 29,
          "architecture": "split_signal_ppo"})

    prog     = ProgressCallback(cb, total_ts,
                                 int(cfg["log_interval"]),
                                 int(cfg["save_interval"]),
                                 output_dir)
    sb3_cb   = _make_sb3_callback(prog)

    model.learn(total_timesteps=total_ts, callback=sb3_cb, progress_bar=False)

    result = prog.on_training_end(model, total_ts)
    vec_env.close()
    return result or {}


# ─── Convenience: run training in background thread ───────────────────────────

def train_ppo_async(
    traj_path:  Path,
    output_dir: Path,
    cfg:        Optional[dict] = None,
) -> tuple[queue.Queue, threading.Thread]:
    """
    Start PPO training in a background thread.
    Returns (event_queue, thread).

    Events on the queue are dicts: {"stage": 6, "msg": str, "data": dict}
    A final event with data["type"] == "done" or msg == "ERROR" signals completion.
    """
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
    args = ap.parse_args()

    cfg = dict(
        total_timesteps = args.total_timesteps,
        n_envs          = args.n_envs,
        n_steps         = args.n_steps,
        batch_size      = args.batch_size,
        learning_rate   = args.lr,
        device          = args.device,
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
