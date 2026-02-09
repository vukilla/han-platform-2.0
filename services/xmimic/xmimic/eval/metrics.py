from __future__ import annotations

import numpy as np


def success_rate(success_flags: np.ndarray) -> float:
    if success_flags.size == 0:
        return 0.0
    return float(np.mean(success_flags.astype(np.float32)))


def generalization_success_rate(success_flags: np.ndarray) -> float:
    return success_rate(success_flags)
