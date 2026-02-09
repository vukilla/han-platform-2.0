from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class GeneralizationConfig:
    disturbed_init_std: float = 0.05
    termination_threshold: float = 0.5
    termination_prob: float = 0.2
    domain_randomization_scale: float = 0.1
    external_force_scale: float = 50.0
    external_force_interval: int = 200
    external_force_duration: int = 20


def disturbed_initialization(state: np.ndarray, config: GeneralizationConfig) -> np.ndarray:
    noise = np.random.normal(scale=config.disturbed_init_std, size=state.shape)
    return state + noise


def interaction_termination(relative_error: float, config: GeneralizationConfig, in_contact: bool = True) -> bool:
    if not in_contact:
        return False
    if relative_error < config.termination_threshold:
        return False
    return np.random.rand() < config.termination_prob


def contact_relative_error(
    object_pos: np.ndarray,
    key_body_pos: np.ndarray,
    ref_relative: np.ndarray,
) -> float:
    """Compute relative error between object and key-body positions."""
    rel = key_body_pos - object_pos
    return float(np.mean(np.linalg.norm(rel - ref_relative, axis=-1)))


def domain_randomization(params: Dict[str, float], config: GeneralizationConfig) -> Dict[str, float]:
    randomized = {}
    for key, value in params.items():
        scale = 1.0 + np.random.uniform(-config.domain_randomization_scale, config.domain_randomization_scale)
        randomized[key] = value * scale
    return randomized


def should_apply_external_force(step: int, config: GeneralizationConfig) -> bool:
    if config.external_force_interval <= 0:
        return False
    return step % config.external_force_interval == 0


def sample_external_force(config: GeneralizationConfig, dims: int = 3) -> np.ndarray:
    direction = np.random.normal(size=(dims,)).astype(np.float32)
    norm = np.linalg.norm(direction) + 1e-8
    direction = direction / norm
    magnitude = np.random.uniform(0.0, config.external_force_scale)
    return direction * magnitude
