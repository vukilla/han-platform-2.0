from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

from .types import RetargetResult


@dataclass
class RetargetValidationResult:
    mean_eef_error: float
    max_eef_error: float
    joint_limit_violations: int
    joint_limit_rate: float
    foot_skate_score: float
    notes: List[str]


def _build_arm_chain(prefix: str) -> Chain:
    return Chain(
        name=f"{prefix}_arm",
        links=[
            OriginLink(),
            URDFLink(
                name=f"{prefix}_shoulder",
                translation_vector=[0, 0, 0],
                orientation=[0, 0, 0],
                rotation=[0, 0, 1],
            ),
            URDFLink(
                name=f"{prefix}_elbow",
                translation_vector=[0.3, 0, 0],
                orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            URDFLink(
                name=f"{prefix}_wrist",
                translation_vector=[0.25, 0, 0],
                orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
        ],
    )


def _forward_wrist(chain: Chain, angles: np.ndarray) -> np.ndarray:
    # angles expects shape (3,) for shoulder/elbow/wrist (origin excluded)
    pose = chain.forward_kinematics([0.0, *angles.tolist()])
    return pose[:3, 3]


def _extract_foot_positions(landmarks: Dict[str, np.ndarray]) -> List[np.ndarray]:
    keys = ["left_ankle", "right_ankle", "left_foot", "right_foot", "left_toe", "right_toe"]
    positions = []
    for key in keys:
        if key in landmarks:
            positions.append(landmarks[key])
    return positions


def validate_retarget(
    landmarks: Dict[str, np.ndarray],
    result: RetargetResult,
    joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ground_height: float = 0.0,
    contact_height_threshold: float = 0.05,
    skate_speed_threshold: float = 0.05,
) -> RetargetValidationResult:
    notes: List[str] = []

    # End-effector error (wrist) using the same kinematic chain as retargeting
    left_chain = _build_arm_chain("left")
    right_chain = _build_arm_chain("right")

    left_wrist = landmarks.get("left_wrist")
    right_wrist = landmarks.get("right_wrist")
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")

    eef_errors = []
    if left_wrist is not None and left_shoulder is not None:
        for i in range(result.robot_qpos.shape[0]):
            angles = result.robot_qpos[i, 0:3]
            pred_local = _forward_wrist(left_chain, angles)
            pred_world = pred_local + left_shoulder[i]
            eef_errors.append(np.linalg.norm(pred_world - left_wrist[i]))
    else:
        notes.append("left_wrist/left_shoulder missing for EEF validation")

    if right_wrist is not None and right_shoulder is not None:
        for i in range(result.robot_qpos.shape[0]):
            angles = result.robot_qpos[i, 3:6]
            pred_local = _forward_wrist(right_chain, angles)
            pred_world = pred_local + right_shoulder[i]
            eef_errors.append(np.linalg.norm(pred_world - right_wrist[i]))
    else:
        notes.append("right_wrist/right_shoulder missing for EEF validation")

    if eef_errors:
        mean_eef_error = float(np.mean(eef_errors))
        max_eef_error = float(np.max(eef_errors))
    else:
        mean_eef_error = 0.0
        max_eef_error = 0.0

    # Joint limit violations
    violations = 0
    total = result.robot_qpos.size
    for j, name in enumerate(result.joint_names):
        q = result.robot_qpos[:, j]
        if joint_limits and name in joint_limits:
            low, high = joint_limits[name]
        else:
            low, high = -np.pi, np.pi
        violations += int(np.sum((q < low) | (q > high)))
    joint_limit_rate = float(violations / total) if total > 0 else 0.0

    # Foot skating metric (if foot landmarks exist)
    foot_positions = _extract_foot_positions(landmarks)
    if foot_positions:
        skate_flags = []
        for foot in foot_positions:
            if foot.shape[0] < 2:
                continue
            velocities = np.linalg.norm(np.diff(foot, axis=0), axis=1)
            heights = foot[:-1, 2]
            contact = heights <= (ground_height + contact_height_threshold)
            skate = (velocities > skate_speed_threshold) & contact
            skate_flags.append(skate)
        if skate_flags:
            skate_concat = np.concatenate(skate_flags)
            foot_skate_score = float(np.mean(skate_concat))
        else:
            foot_skate_score = 0.0
    else:
        notes.append("foot landmarks missing; skipping foot skating metric")
        foot_skate_score = 0.0

    return RetargetValidationResult(
        mean_eef_error=mean_eef_error,
        max_eef_error=max_eef_error,
        joint_limit_violations=violations,
        joint_limit_rate=joint_limit_rate,
        foot_skate_score=foot_skate_score,
        notes=notes,
    )
