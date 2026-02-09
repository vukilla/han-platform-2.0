from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class EnvConfig:
    name: str
    obs_dim: int
    action_dim: int
    max_steps: int = 200


class BaseEnv:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.step_count = 0
        self.last_obs = np.zeros((self.config.obs_dim,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.last_obs = np.zeros((self.config.obs_dim,), dtype=np.float32)
        return self.last_obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = 0.0
        done = self.step_count >= self.config.max_steps
        info: Dict[str, Any] = {}
        return self.last_obs, reward, done, info


class CargoPickupEnv(BaseEnv):
    def __init__(self):
        super().__init__(EnvConfig(name="cargo_pickup_v0", obs_dim=48, action_dim=24, max_steps=240))
