from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PhaseSegments:
    pre_contact: np.ndarray
    contact: np.ndarray
    post_contact: np.ndarray


def segment_phases(num_frames: int, fps: float, ts_start: float, ts_end: float) -> np.ndarray:
    """Return phase labels for each frame.

    0 = pre-contact, 1 = contact, 2 = post-contact.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if ts_end < ts_start:
        raise ValueError("ts_end must be >= ts_start")

    ts_start_idx = int(round(ts_start * fps))
    ts_end_idx = int(round(ts_end * fps))

    ts_start_idx = max(0, min(num_frames - 1, ts_start_idx))
    ts_end_idx = max(ts_start_idx, min(num_frames - 1, ts_end_idx))

    phases = np.zeros((num_frames,), dtype=np.int32)
    phases[ts_start_idx : ts_end_idx + 1] = 1
    phases[ts_end_idx + 1 :] = 2
    return phases


def split_segments(phases: np.ndarray) -> PhaseSegments:
    return PhaseSegments(
        pre_contact=np.where(phases == 0)[0],
        contact=np.where(phases == 1)[0],
        post_contact=np.where(phases == 2)[0],
    )
