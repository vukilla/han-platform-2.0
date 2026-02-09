from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from xmimic.controllers import ActionSemantics, PDController, PDGains
from xmimic.obs_pipeline import ObservationPipeline


@dataclass
class RuntimeConfig:
    control_rate_hz: float = 100.0
    pd_rate_hz: float = 1000.0
    action_mode: str = "target"
    action_scale: float = 1.0
    kp: float = 100.0
    kd: float = 10.0
    torque_limit: float = 150.0
    device: str = "cpu"
    action_dim: int = 24


class TorchPolicy:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            mu = self.model(tensor)[0] if isinstance(self.model(tensor), tuple) else self.model(tensor)
            return mu.squeeze(0).cpu().numpy()


class RuntimeLoop:
    def __init__(
        self,
        config: RuntimeConfig,
        obs_pipeline: ObservationPipeline,
        sensor_fn: Callable[[], Dict[str, np.ndarray]],
        actuator_fn: Callable[[np.ndarray], None],
        policy_fn: Callable[[np.ndarray], np.ndarray],
    ):
        self.config = config
        self.obs_pipeline = obs_pipeline
        self.sensor_fn = sensor_fn
        self.actuator_fn = actuator_fn
        gains = PDGains(
            kp=np.full(config.action_dim, config.kp, dtype=np.float32),
            kd=np.full(config.action_dim, config.kd, dtype=np.float32),
            torque_limit=np.full(config.action_dim, config.torque_limit, dtype=np.float32),
        )
        semantics = ActionSemantics(
            mode=config.action_mode,
            action_scale=config.action_scale,
            control_rate_hz=config.control_rate_hz,
            pd_rate_hz=config.pd_rate_hz,
        )
        self.pd = PDController(gains, semantics)
        self.policy_fn = policy_fn

    def run(self, max_steps: Optional[int] = None):
        policy_dt = 1.0 / self.config.control_rate_hz
        pd_dt = 1.0 / self.config.pd_rate_hz
        pd_steps = max(1, int(round(self.config.pd_rate_hz / self.config.control_rate_hz)))

        step = 0
        while max_steps is None or step < max_steps:
            start = time.time()
            raw = self.sensor_fn()
            obs = self.obs_pipeline.build(raw)
            action = self.policy_fn(obs)

            for _ in range(pd_steps):
                raw_pd = self.sensor_fn()
                q = np.asarray(raw_pd.get("q", np.zeros_like(action)), dtype=np.float32)
                qd = np.asarray(raw_pd.get("qd", np.zeros_like(action)), dtype=np.float32)
                torque = self.pd.compute_torque(q, qd, action)
                self.actuator_fn(torque)
                time.sleep(pd_dt)

            elapsed = time.time() - start
            if elapsed < policy_dt:
                time.sleep(policy_dt - elapsed)
            step += 1
