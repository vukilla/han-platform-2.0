from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PDGains:
    kp: np.ndarray
    kd: np.ndarray
    torque_limit: Optional[np.ndarray] = None


@dataclass
class ActionSemantics:
    mode: str = "target"  # "target" or "delta"
    action_scale: float = 1.0
    control_rate_hz: float = 100.0
    pd_rate_hz: float = 1000.0


class PDController:
    def __init__(self, gains: PDGains, semantics: Optional[ActionSemantics] = None):
        self.gains = gains
        self.semantics = semantics or ActionSemantics()

    def target_from_action(self, q: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.semantics.mode == "delta":
            return q + action * self.semantics.action_scale
        return action * self.semantics.action_scale

    def compute_torque(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        action: np.ndarray,
        qd_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q_target = self.target_from_action(q, action)
        if qd_target is None:
            qd_target = np.zeros_like(qd)
        torque = self.gains.kp * (q_target - q) + self.gains.kd * (qd_target - qd)
        if self.gains.torque_limit is not None:
            limit = np.asarray(self.gains.torque_limit)
            torque = np.clip(torque, -limit, limit)
        return torque
