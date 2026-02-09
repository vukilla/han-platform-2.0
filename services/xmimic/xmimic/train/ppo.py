from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from xmimic.generalization import (
    GeneralizationConfig,
    disturbed_initialization,
    interaction_termination,
    domain_randomization,
    should_apply_external_force,
    sample_external_force,
)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.policy(obs)
        value = self.value(obs).squeeze(-1)
        return mu, value


@dataclass
class PPOConfig:
    steps: int = 512
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4


def _compute_advantages(rewards, values, dones, gamma, lam):
    adv = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        lastgaelam = delta + gamma * lam * (1 - dones[t]) * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[: len(adv)]
    return adv, returns


def train_ppo(
    env,
    obs_dim: int,
    action_dim: int,
    config: PPOConfig,
    teacher_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    bc_weight: float = 0.0,
    generalization: Optional[GeneralizationConfig] = None,
    return_model: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if generalization and hasattr(env, "get_params") and hasattr(env, "set_params"):
        params = env.get_params()
        env.set_params(domain_randomization(params, generalization))

    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    if generalization:
        obs = torch.tensor(disturbed_initialization(obs.cpu().numpy(), generalization), dtype=torch.float32, device=device)
    trajectories = {"obs": [], "actions": [], "logp": [], "rewards": [], "values": [], "dones": [], "teacher": []}

    for step_idx in range(config.steps):
        if generalization and hasattr(env, "apply_external_force"):
            if should_apply_external_force(step_idx, generalization):
                force = sample_external_force(generalization)
                try:
                    env.apply_external_force(force, duration=generalization.external_force_duration)
                except TypeError:
                    env.apply_external_force(force)
        mu, value = model(obs.unsqueeze(0))
        std = model.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        next_obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
        if generalization and info and "relative_error" in info:
            in_contact = bool(info.get("in_contact", True))
            if interaction_termination(float(info["relative_error"]), generalization, in_contact=in_contact):
                done = True

        trajectories["obs"].append(obs.cpu().numpy())
        trajectories["actions"].append(action.squeeze(0).cpu().numpy())
        trajectories["logp"].append(logp.item())
        trajectories["rewards"].append(reward)
        trajectories["values"].append(value.item())
        trajectories["dones"].append(float(done))
        if teacher_policy is not None:
            trajectories["teacher"].append(teacher_policy(obs.cpu().numpy()))

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        if done:
            if generalization and hasattr(env, "get_params") and hasattr(env, "set_params"):
                params = env.get_params()
                env.set_params(domain_randomization(params, generalization))
            obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
            if generalization:
                obs = torch.tensor(disturbed_initialization(obs.cpu().numpy(), generalization), dtype=torch.float32, device=device)

    rewards = np.array(trajectories["rewards"], dtype=np.float32)
    values = np.array(trajectories["values"], dtype=np.float32)
    dones = np.array(trajectories["dones"], dtype=np.float32)
    advantages, returns = _compute_advantages(rewards, values, dones, config.gamma, config.lam)

    obs_tensor = torch.tensor(np.array(trajectories["obs"]), dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(np.array(trajectories["actions"]), dtype=torch.float32, device=device)
    old_logp = torch.tensor(np.array(trajectories["logp"]), dtype=torch.float32, device=device)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    teacher_tensor = None
    if teacher_policy is not None:
        teacher_tensor = torch.tensor(np.array(trajectories["teacher"]), dtype=torch.float32, device=device)

    for _ in range(config.epochs):
        mu, values_pred = model(obs_tensor)
        std = model.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(actions_tensor).sum(-1)
        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv_tensor
        policy_loss = -(torch.min(ratio * adv_tensor, clipped)).mean()
        value_loss = ((values_pred - ret_tensor) ** 2).mean()
        loss = policy_loss + 0.5 * value_loss
        if teacher_tensor is not None and bc_weight > 0:
            bc_loss = ((mu - teacher_tensor) ** 2).mean()
            loss = loss + bc_weight * bc_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "bc_weight": float(bc_weight),
    }
    if return_model:
        return metrics, model
    return metrics
