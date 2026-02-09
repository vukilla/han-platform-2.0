from __future__ import annotations

import numpy as np


def build_nep_observation(proprio: np.ndarray) -> np.ndarray:
    return proprio.astype(np.float32)


def build_mocap_observation(
    proprio: np.ndarray,
    object_pose: np.ndarray,
    dropout_prob: float = 0.1,
) -> np.ndarray:
    if np.random.rand() < dropout_prob:
        object_features = np.zeros_like(object_pose, dtype=np.float32)
    else:
        object_features = object_pose.astype(np.float32)
    return np.concatenate([proprio.astype(np.float32), object_features], axis=-1)


def build_obs_from_root_states(root_states, include_object: bool = True, dropout_prob: float = 0.1) -> np.ndarray:
    """Build NEP/MoCap observation from Isaac Gym root states list."""
    proprio = []
    object_pose = []
    for state in root_states:
        proprio.extend(state.get("humanoid_pos", []))
        object_pose.extend(state.get("object_pos", []))
    proprio = np.array(proprio, dtype=np.float32)
    object_pose = np.array(object_pose, dtype=np.float32)
    if include_object:
        return build_mocap_observation(proprio, object_pose, dropout_prob=dropout_prob)
    return build_nep_observation(proprio)
