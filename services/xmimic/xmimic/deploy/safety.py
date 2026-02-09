from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SafetyConfig:
    torque_limit: Optional[np.ndarray] = None
    joint_lower: Optional[np.ndarray] = None
    joint_upper: Optional[np.ndarray] = None
    fall_pitch_limit_deg: float = 45.0
    fall_roll_limit_deg: float = 45.0
    watchdog_timeout_s: float = 0.2


@dataclass
class SafetyStatus:
    ok: bool
    reasons: List[str]


def apply_torque_limits(torque: np.ndarray, limit: Optional[np.ndarray]) -> np.ndarray:
    if limit is None:
        return torque
    limit = np.asarray(limit, dtype=np.float32)
    return np.clip(torque, -limit, limit)


def check_joint_limits(q: np.ndarray, lower: Optional[np.ndarray], upper: Optional[np.ndarray]) -> bool:
    if lower is None or upper is None:
        return True
    return bool(np.all(q >= lower) and np.all(q <= upper))


def check_fall(pitch_deg: float, roll_deg: float, cfg: SafetyConfig) -> bool:
    return abs(pitch_deg) < cfg.fall_pitch_limit_deg and abs(roll_deg) < cfg.fall_roll_limit_deg


def safety_check(
    q: np.ndarray,
    torque: np.ndarray,
    pitch_deg: float,
    roll_deg: float,
    time_since_cmd: float,
    cfg: SafetyConfig,
) -> SafetyStatus:
    reasons: List[str] = []
    if not check_joint_limits(q, cfg.joint_lower, cfg.joint_upper):
        reasons.append("joint_limits")
    if not check_fall(pitch_deg, roll_deg, cfg):
        reasons.append("fall_detect")
    if time_since_cmd > cfg.watchdog_timeout_s:
        reasons.append("watchdog")
    ok = len(reasons) == 0
    return SafetyStatus(ok=ok, reasons=reasons)
