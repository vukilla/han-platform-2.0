from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class QualityResult:
    score: float
    breakdown: Dict[str, float]


def evaluate_demo(frame_count: int, pose_ok: bool) -> QualityResult:
    breakdown = {
        "completeness": 1.0 if frame_count > 0 else 0.0,
        "pose_ok": 1.0 if pose_ok else 0.0,
    }
    score = sum(breakdown.values()) / len(breakdown)
    return QualityResult(score=score, breakdown=breakdown)


def evaluate_clip(has_contact_graph: bool, joint_limits_ok: bool, object_spikes_ok: bool) -> QualityResult:
    breakdown = {
        "contact_graph": 1.0 if has_contact_graph else 0.0,
        "joint_limits": 1.0 if joint_limits_ok else 0.0,
        "object_spikes": 1.0 if object_spikes_ok else 0.0,
    }
    score = sum(breakdown.values()) / len(breakdown)
    return QualityResult(score=score, breakdown=breakdown)
