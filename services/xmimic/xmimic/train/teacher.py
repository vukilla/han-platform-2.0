from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from xmimic.envs import BaseEnv
from xmimic.rewards import RewardWeights, compute_reward_terms, compute_total_reward
from xmimic.train.ppo import PPOConfig, train_ppo


@dataclass
class TeacherConfig:
    steps: int = 256
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def _flatten_obs(obs) -> np.ndarray:
    if isinstance(obs, dict) and "root_states" in obs:
        flat = []
        for state in obs["root_states"]:
            flat.extend(state.get("humanoid_pos", []))
            flat.extend(state.get("object_pos", []))
        return np.array(flat, dtype=np.float32)
    return np.array(obs, dtype=np.float32)


class _EnvAdapter:
    def __init__(self, env: BaseEnv):
        self.env = env
        sample = _flatten_obs(env.reset())
        self.obs_dim = sample.shape[0]
        self.action_dim = env.config.action_dim if hasattr(env, "config") else 12

    def reset(self):
        return _flatten_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _flatten_obs(obs), reward, done, info


def train_teacher(env: BaseEnv, config: TeacherConfig) -> Dict[str, float]:
    adapter = _EnvAdapter(env)
    losses = train_ppo(adapter, adapter.obs_dim, adapter.action_dim, config.ppo)
    return losses
