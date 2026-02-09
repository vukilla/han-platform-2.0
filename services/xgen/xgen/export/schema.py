from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ClipData:
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    root_pose: np.ndarray
    root_vel: np.ndarray | None = None
    object_pose: np.ndarray
    object_vel: np.ndarray | None = None
    contact_graph: np.ndarray
    phase: np.ndarray
    metadata: Dict[str, Any]

    def validate(self) -> None:
        length = self.robot_qpos.shape[0]
        if length == 0:
            raise ValueError("clip has no frames")
        fields = {
            "robot_qvel": self.robot_qvel,
            "root_pose": self.root_pose,
            "object_pose": self.object_pose,
            "contact_graph": self.contact_graph,
            "phase": self.phase,
        }
        if self.root_vel is not None:
            fields["root_vel"] = self.root_vel
        if self.object_vel is not None:
            fields["object_vel"] = self.object_vel
        for name, arr in fields.items():
            if arr.shape[0] != length:
                raise ValueError(f"{name} length mismatch: {arr.shape[0]} != {length}")
