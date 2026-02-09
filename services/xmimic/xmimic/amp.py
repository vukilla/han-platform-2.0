from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AMPBatch:
    expert: torch.Tensor
    policy: torch.Tensor


def amp_discriminator_loss(discriminator: AMPDiscriminator, batch: AMPBatch) -> torch.Tensor:
    logits_expert = discriminator(batch.expert)
    logits_policy = discriminator(batch.policy)
    loss_expert = nn.functional.binary_cross_entropy_with_logits(logits_expert, torch.ones_like(logits_expert))
    loss_policy = nn.functional.binary_cross_entropy_with_logits(logits_policy, torch.zeros_like(logits_policy))
    return loss_expert + loss_policy


def amp_reward_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Higher reward when discriminator believes sample is expert-like.
    return torch.sigmoid(logits)


def compute_amp_reward(discriminator: AMPDiscriminator, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = discriminator(torch.tensor(obs, dtype=torch.float32))
        reward = amp_reward_from_logits(logits)
    return reward.squeeze(-1).cpu().numpy()
