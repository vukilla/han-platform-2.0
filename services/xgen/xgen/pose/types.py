from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PoseSequence:
    fps: float
    joints_3d: np.ndarray
    joints_2d: np.ndarray
    visibility: np.ndarray
    landmark_names: List[str]
