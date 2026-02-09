from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MocapFrame:
    position: np.ndarray
    orientation: np.ndarray
    timestamp: float
    valid: bool = True
    mask: Optional[np.ndarray] = None


@dataclass
class MocapDropout:
    dropout_prob: float = 0.1
    hold_last: bool = True

    def apply(self, frame: MocapFrame, last_valid: Optional[MocapFrame] = None) -> MocapFrame:
        if np.random.rand() >= self.dropout_prob:
            frame.mask = np.ones_like(frame.position, dtype=np.float32)
            frame.valid = True
            return frame
        if self.hold_last and last_valid is not None:
            dropped = MocapFrame(
                position=last_valid.position.copy(),
                orientation=last_valid.orientation.copy(),
                timestamp=frame.timestamp,
                valid=False,
                mask=np.zeros_like(frame.position, dtype=np.float32),
            )
            return dropped
        frame.position = np.zeros_like(frame.position, dtype=np.float32)
        frame.orientation = np.zeros_like(frame.orientation, dtype=np.float32)
        frame.valid = False
        frame.mask = np.zeros_like(frame.position, dtype=np.float32)
        return frame


def transform_to_robot_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    translation: np.ndarray,
    rotation_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a rigid transform to map MoCap world frame into robot base frame."""
    pos = position + translation
    # Placeholder: proper quaternion multiply should be used in real integration.
    return pos, orientation
