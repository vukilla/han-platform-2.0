from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _smooth_window(series: np.ndarray, left_idx: int, right_idx: int, window: int) -> np.ndarray:
    smoothed = series.copy()
    total_frames = series.shape[0]
    start = max(0, left_idx - window + 1)
    end = min(total_frames - 1, right_idx + window - 1)
    if end <= start:
        return smoothed

    left_val = series[left_idx]
    right_val = series[right_idx]
    for i in range(start, end + 1):
        alpha = (i - start) / float(end - start)
        smoothed[i] = (1.0 - alpha) * left_val + alpha * right_val
    return smoothed


def smooth_phase_transitions(
    phases: np.ndarray,
    series: Dict[str, np.ndarray],
    window: int = 5,
) -> Dict[str, np.ndarray]:
    """Blend body/object/velocity series around phase boundaries.

    `series` should contain arrays shaped (T, D). This function returns a new
    dict with smoothed copies.
    """
    if phases.ndim != 1:
        raise ValueError("phases must be 1D")

    transitions = np.where(np.diff(phases) != 0)[0]
    if transitions.size == 0:
        return {k: v.copy() for k, v in series.items()}

    output: Dict[str, np.ndarray] = {k: v.copy() for k, v in series.items()}
    for boundary in transitions:
        left_idx = int(boundary)
        right_idx = int(boundary + 1)
        for key, values in output.items():
            if values.shape[0] != phases.shape[0]:
                raise ValueError(f"Series {key} length {values.shape[0]} != phases length {phases.shape[0]}")
            output[key] = _smooth_window(values, left_idx, right_idx, window)

    return output
