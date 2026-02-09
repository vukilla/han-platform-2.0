from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from xmimic.envs import BaseEnv
from xmimic.rewards import RewardWeights, compute_reward_terms, compute_total_reward
from xmimic.train.ppo import PPOConfig, train_ppo


@dataclass
class StudentConfig:
    steps: int = 256
    bc_weight: float = 0.5
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def distill_student(env: BaseEnv, config: StudentConfig, teacher_policy=None) -> Dict[str, float]:
    obs = env.reset()
    total_reward = 0.0
    bc_loss = 0.0
    for _ in range(config.steps):
        action = np.zeros((env.config.action_dim,), dtype=np.float32)
        next_obs, _, done, _ = env.step(action)
        terms = compute_reward_terms(
            obs={"body": obs, "object": obs, "relative": obs, "contact": obs, "action": action},
            targets={"body": obs, "object": obs, "relative": obs, "contact": obs},
            weights=config.reward_weights,
        )
        total_reward += compute_total_reward(terms)
        bc_loss += float(np.mean(action**2)) * config.bc_weight
        obs = next_obs
        if done:
            obs = env.reset()
    return {"total_reward": total_reward, "bc_loss": bc_loss}


class _EnvAdapter:
    def __init__(self, env: BaseEnv):
        self.env = env
        sample = self.reset()
        self.obs_dim = sample.shape[0]
        self.action_dim = env.config.action_dim if hasattr(env, "config") else 12

    def reset(self):
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32) if not isinstance(obs, np.ndarray) else obs.astype(np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.array(obs, dtype=np.float32) if not isinstance(obs, np.ndarray) else obs.astype(np.float32)
        return obs, reward, done, info


def train_student_ppo(env: BaseEnv, config: StudentConfig, teacher_policy) -> Dict[str, float]:
    adapter = _EnvAdapter(env)
    losses = train_ppo(
        adapter,
        adapter.obs_dim,
        adapter.action_dim,
        config.ppo,
        teacher_policy=teacher_policy,
        bc_weight=config.bc_weight,
    )
    return losses
