from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_contact_graph(
    body_positions: Dict[str, np.ndarray],
    object_pose: np.ndarray,
    body_parts: List[str],
    contact_distance: float = 0.05,
) -> np.ndarray:
    """Compute binary contact graph (T, J) based on distance to object center."""
    if object_pose.shape[1] < 3:
        raise ValueError("object_pose must include xyz translation")
    object_center = object_pose[:, :3]
    frames = object_center.shape[0]
    graph = np.zeros((frames, len(body_parts)), dtype=np.int32)
    for j, name in enumerate(body_parts):
        positions = body_positions.get(name)
        if positions is None:
            continue
        distances = np.linalg.norm(positions - object_center, axis=1)
        graph[:, j] = (distances <= contact_distance).astype(np.int32)
    return graph
