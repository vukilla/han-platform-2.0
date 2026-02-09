from __future__ import annotations

import numpy as np


def object_tracking_error(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape:
        raise ValueError("shape mismatch for object tracking error")
    return float(np.mean(np.linalg.norm(pred - target, axis=-1)))


def key_body_tracking_error(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape:
        raise ValueError("shape mismatch for key body tracking error")
    return float(np.mean(np.linalg.norm(pred - target, axis=-1)))
