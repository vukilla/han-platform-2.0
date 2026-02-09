from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple

import yaml

from xmimic.train.ppo import PPOConfig, train_ppo
from xmimic.train.checkpoint import save_checkpoint


@dataclass
class PhysHOIConfig:
    gamma: float
    tau: float
    learning_rate: float
    e_clip: float
    horizon_length: int
    mini_epochs: int


def load_physhoi_config(path: str | Path) -> PhysHOIConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = payload.get("params", {}).get("config", {})
    return PhysHOIConfig(
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.95)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        e_clip=float(cfg.get("e_clip", 0.2)),
        horizon_length=int(cfg.get("horizon_length", 512)),
        mini_epochs=int(cfg.get("mini_epochs", 4)),
    )


def ppo_config_from_physhoi(path: str | Path) -> PPOConfig:
    cfg = load_physhoi_config(path)
    return PPOConfig(
        steps=cfg.horizon_length,
        gamma=cfg.gamma,
        lam=cfg.tau,
        clip=cfg.e_clip,
        lr=cfg.learning_rate,
        epochs=cfg.mini_epochs,
    )


def distill_student_with_physhoi(
    env,
    obs_dim: int,
    action_dim: int,
    physhoi_cfg_path: str | Path,
    checkpoint_path: str | Path,
    teacher_policy: Optional[Callable[[Any], Any]] = None,
    bc_weight: float = 1.0,
) -> Dict[str, float]:
    ppo_cfg = ppo_config_from_physhoi(physhoi_cfg_path)
    metrics, model = train_ppo(
        env,
        obs_dim,
        action_dim,
        ppo_cfg,
        teacher_policy=teacher_policy,
        bc_weight=bc_weight,
        return_model=True,
    )
    save_checkpoint(model, checkpoint_path)
    return metrics
