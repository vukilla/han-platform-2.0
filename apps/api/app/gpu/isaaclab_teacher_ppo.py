from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class IsaacLabTeacherPPOConfig:
    # Task selector.
    #
    # Supported (Windows Isaac Sim + Isaac Lab):
    # - cargo_pickup_franka: Franka cube lift (pickup) environment. This is the current "cargo pickup" baseline.
    # - franka_cube_lift: alias for cargo_pickup_franka (kept for backwards compatibility).
    task: str = "cargo_pickup_franka"
    device: str = "cuda:0"
    num_envs: int = 64
    seed: int = 0

    rollout_steps: int = 128
    updates: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    epochs: int = 4


def _normalize_task(task: str) -> str:
    t = (task or "").strip().lower()
    if t in {"franka_cube_lift", "cargo_pickup_franka"}:
        return "cargo_pickup_franka"
    return t


def _mlp_actor_critic(obs_dim: int, act_dim: int, hidden: int = 256):
    # Import torch lazily so Mac containers do not require it.
    import torch
    import torch.nn as nn

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, act_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        def forward(self, obs):
            mu = self.actor(obs)
            v = self.critic(obs).squeeze(-1)
            return mu, v

    return ActorCritic()


def _compute_gae(rewards, dones, values, next_value, gamma: float, lam: float):
    """Vectorized GAE for (T, N) tensors."""
    import torch

    T, N = rewards.shape
    advantages = torch.zeros((T, N), device=rewards.device, dtype=rewards.dtype)
    last_adv = torch.zeros((N,), device=rewards.device, dtype=rewards.dtype)
    last_val = next_value
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_val * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
        last_val = values[t]
    returns = advantages + values
    return advantages, returns


def train_teacher_ppo(
    cfg: IsaacLabTeacherPPOConfig,
    *,
    out_dir: str | Path,
    log_cb: callable | None = None,
) -> dict[str, Any]:
    """Train a PPO teacher on an Isaac Lab task and export a checkpoint.

    This is intentionally a small, deterministic training loop to produce a *real* checkpoint artifact.
    It is not expected to reach high performance in the default settings.
    """
    import json
    import os
    import sys
    import time
    import traceback

    import torch

    task = _normalize_task(cfg.task)

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_path = out_path / "result.json"

    def write_result(payload: dict) -> None:
        # Write results early and often because Isaac Sim shutdown can terminate the
        # whole process before callers (e.g. a CLI wrapper) can persist outcomes.
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Seed the result file so the parent process can at least read *something* even
    # if Kit terminates unexpectedly during shutdown.
    try:
        write_result({"ok": False, "stage": "starting", "task": task, "config": cfg.__dict__})
    except Exception:
        pass

    # Make runs more reproducible.
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    torch.manual_seed(cfg.seed)

    # IMPORTANT (Windows / Isaac Sim 5.x):
    # `omni.*` modules (including `omni.timeline`) are only importable after Kit is initialized.
    # Importing Isaac Lab modules too early can fail with `ModuleNotFoundError: omni.timeline`.
    #
    # When launched from a Celery worker, stdout/stderr can be wrapped by Celery's LoggingProxy
    # which lacks `.fileno()`. Isaac Sim's SimulationApp expects real file handles.
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    try:
        if not hasattr(sys.stdout, "fileno"):
            sys.stdout = sys.__stdout__ if hasattr(sys.__stdout__, "fileno") else open(os.devnull, "w")
        if not hasattr(sys.stderr, "fileno"):
            sys.stderr = sys.__stderr__ if hasattr(sys.__stderr__, "fileno") else open(os.devnull, "w")
    except Exception:
        # Best-effort; if this fails we still try to launch SimulationApp.
        pass

    # IMPORTANT:
    # On Windows, we run the GPU Celery worker *inside* Isaac Sim's Python (Kit).
    # In that case, Kit is already initialized. Creating/closing a new SimulationApp
    # will often shut down the entire process, killing the Celery worker.
    simulation_app = None
    try:
        from omni.kit.app import get_app  # type: ignore

        if get_app() is not None:
            log("[isaacsim] detected existing Kit app; skipping SimulationApp init/close")
        else:
            raise ImportError("Kit app not initialized")
    except Exception:
        log(f"[isaacsim] launching headless SimulationApp device={cfg.device}")
        from isaacsim import SimulationApp

        # NOTE:
        # Isaac Sim defaults to `fast_shutdown=true` in some experiences, and `SimulationApp.close()`
        # can terminate the entire process before Python unwinds (which breaks subprocess callers).
        # Disabling fast shutdown makes shutdown more predictable.
        simulation_app = SimulationApp({"headless": True, "fast_shutdown": False})

    try:
        # Import Isaac Lab only after SimulationApp is live.
        from isaaclab.envs import ManagerBasedRLEnv

        # Env config
        if task == "cargo_pickup_franka":
            from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
                FrankaCubeLiftEnvCfg,
            )

            env_cfg = FrankaCubeLiftEnvCfg()
            env_cfg.scene.num_envs = int(cfg.num_envs)
            env_cfg.sim.device = cfg.device
            env = ManagerBasedRLEnv(cfg=env_cfg)
        else:
            raise ValueError(
                f"Unsupported isaaclab teacher PPO task: {cfg.task!r}. "
                "Supported: cargo_pickup_franka, franka_cube_lift."
            )
        try:
            obs_dict = env.reset()
            obs = obs_dict["policy"]
        except Exception as exc:
            raise RuntimeError(f"Failed to reset Isaac Lab env: {exc}") from exc

        act_dim = int(env.action_manager.action.shape[-1])
        obs_dim = int(obs.shape[-1])
        log(f"[isaaclab] task={task} obs_dim={obs_dim} act_dim={act_dim} num_envs={int(cfg.num_envs)}")

        model = _mlp_actor_critic(obs_dim, act_dim).to(env.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

        # Rollout buffers (allocated per update)
        rollout_T = int(cfg.rollout_steps)
        clip_eps = float(cfg.clip_range)
        gamma = float(cfg.gamma)
        lam = float(cfg.gae_lambda)
        epochs = int(cfg.epochs)

        start_time = time.time()
        total_env_steps = 0
        last_mean_reward = 0.0

        for update in range(int(cfg.updates)):
            obs_buf = torch.zeros((rollout_T, cfg.num_envs, obs_dim), device=env.device)
            act_buf = torch.zeros((rollout_T, cfg.num_envs, act_dim), device=env.device)
            logp_buf = torch.zeros((rollout_T, cfg.num_envs), device=env.device)
            rew_buf = torch.zeros((rollout_T, cfg.num_envs), device=env.device)
            done_buf = torch.zeros((rollout_T, cfg.num_envs), device=env.device)
            val_buf = torch.zeros((rollout_T, cfg.num_envs), device=env.device)

            for t in range(rollout_T):
                with torch.no_grad():
                    mu, v = model(obs)
                    std = model.log_std.exp().expand_as(mu)
                    dist = torch.distributions.Normal(mu, std)
                    act = dist.sample()
                    act = torch.clamp(act, -1.0, 1.0)
                    logp = dist.log_prob(act).sum(-1)

                obs_buf[t] = obs
                act_buf[t] = act
                logp_buf[t] = logp
                val_buf[t] = v

                obs_dict, rew, terminated, truncated, _info = env.step(act)
                obs = obs_dict["policy"]
                rew_buf[t] = rew
                done = (terminated | truncated).float()
                done_buf[t] = done

            with torch.no_grad():
                _mu, next_v = model(obs)
            adv, ret = _compute_gae(rew_buf, done_buf, val_buf, next_v, gamma=gamma, lam=lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Flatten (T, N, ..) -> (T*N, ..)
            b_obs = obs_buf.reshape(-1, obs_dim)
            b_act = act_buf.reshape(-1, act_dim)
            b_logp = logp_buf.reshape(-1)
            b_adv = adv.reshape(-1)
            b_ret = ret.reshape(-1)

            for _ in range(epochs):
                mu, v = model(b_obs)
                std = model.log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                new_logp = dist.log_prob(b_act).sum(-1)
                ratio = torch.exp(new_logp - b_logp)

                unclipped = ratio * b_adv
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (v.reshape(-1) - b_ret).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()

                loss = policy_loss + 0.5 * value_loss - 0.001 * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_env_steps += rollout_T * int(cfg.num_envs)
            last_mean_reward = float(rew_buf.mean().item())
            log(
                f"[ppo] update={update+1}/{cfg.updates} mean_rew={last_mean_reward:.4f} "
                f"steps={total_env_steps}"
            )

        ckpt_path = out_path / f"teacher_ppo_{task}.pt"
        payload = {
            "algo": "ppo",
            "task": task,
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "seed": cfg.seed,
            "config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
        }
        torch.save(payload, ckpt_path)

        wall_s = time.time() - start_time
        log(f"[done] checkpoint={ckpt_path} wall_s={wall_s:.1f}")
        metrics = {
            "checkpoint_path": str(ckpt_path),
            "task": task,
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "num_envs": int(cfg.num_envs),
            "env_steps": int(total_env_steps),
            "mean_reward": float(last_mean_reward),
            "wall_s": float(wall_s),
        }
        try:
            write_result({"ok": True, "metrics": metrics})
        except Exception:
            pass
        return metrics
    except Exception as exc:
        try:
            write_result(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        raise
    finally:
        # best-effort cleanup
        try:
            env.close()  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass
        try:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
        except Exception:
            pass
