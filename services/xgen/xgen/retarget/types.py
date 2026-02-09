from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RetargetResult:
    robot_qpos: np.ndarray
    root_pose: np.ndarray
    joint_names: List[str]
