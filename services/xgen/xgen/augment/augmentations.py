from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from xgen.interaction import simulate_noncontact


@dataclass
class AugmentationResult:
    trajectory: np.ndarray
    tags: list[str]


def scale_mesh(vertices: np.ndarray, scale: float) -> np.ndarray:
    return vertices * scale


def substitute_mesh(source_mesh_path: str, replacement_mesh_path: str, scale: float = 1.0) -> trimesh.Trimesh:
    replacement = trimesh.load(replacement_mesh_path, force="mesh")
    if scale != 1.0:
        replacement.apply_scale(scale)
    return replacement


def transform_contact_trajectory(
    trajectory: np.ndarray,
    translation: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> AugmentationResult:
    updated = trajectory.copy()
    tags: list[str] = []
    if scale is not None:
        updated *= scale
        tags.append(f"scale:{scale:.2f}")
    if translation is not None:
        updated += translation
        tags.append("translate")
    return AugmentationResult(trajectory=updated, tags=tags)


def randomize_velocity(trajectory: np.ndarray, magnitude: float = 0.05) -> AugmentationResult:
    noise = np.random.normal(scale=magnitude, size=trajectory.shape)
    updated = trajectory + noise
    return AugmentationResult(trajectory=updated, tags=[f"velocity_noise:{magnitude:.2f}"])


def sweep_augmentations(
    object_pose: np.ndarray,
    contact_indices: np.ndarray,
    fps: float,
    scales: list[float] | None = None,
    velocity_noises: list[float] | None = None,
    gravity: float = 9.81,
    base_object_size: float = 0.1,
) -> list[AugmentationResult]:
    """
    Generate augmentation sweeps by scaling object size and injecting velocity noise,
    then re-running non-contact simulation.
    """
    results: list[AugmentationResult] = []
    scales = scales or [1.0]
    velocity_noises = velocity_noises or [0.0]

    for scale in scales:
        for vel_noise in velocity_noises:
            pose = object_pose.copy()
            tags = [f"scale:{scale:.2f}"] if scale != 1.0 else []
            if vel_noise > 0:
                pose[:, :3] += np.random.normal(scale=vel_noise, size=pose[:, :3].shape)
                tags.append(f"vel_noise:{vel_noise:.2f}")
            sim = simulate_noncontact(
                pose,
                contact_indices=contact_indices,
                fps=fps,
                gravity=gravity,
                reverse=False,
                object_size=base_object_size * scale,
            )
            results.append(AugmentationResult(trajectory=sim, tags=tags))

    return results
