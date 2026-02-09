from __future__ import annotations

from typing import Dict, List

import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

from .types import RetargetResult


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


def _solve_arm(chain: Chain, target: np.ndarray) -> np.ndarray:
    # IK in chain coordinates
    ik = chain.inverse_kinematics(target)
    return ik[1:]  # skip origin


def retarget_upper_body(landmarks: Dict[str, np.ndarray]) -> RetargetResult:
    """Retarget pose landmarks to a simplified humanoid upper body.

    landmarks: dict of key landmarks -> (T, 3)
    """
    left_chain = _build_arm_chain("left")
    right_chain = _build_arm_chain("right")

    required = ["left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "pelvis"]
    for key in required:
        if key not in landmarks:
            raise ValueError(f"missing landmark: {key}")

    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    left_wrist = landmarks["left_wrist"]
    right_wrist = landmarks["right_wrist"]
    pelvis = landmarks["pelvis"]

    frames = left_wrist.shape[0]
    joint_names = [
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist",
    ]
    robot_qpos = np.zeros((frames, len(joint_names)), dtype=np.float32)
    root_pose = np.zeros((frames, 7), dtype=np.float32)

    for idx in range(frames):
        left_target = left_wrist[idx] - left_shoulder[idx]
        right_target = right_wrist[idx] - right_shoulder[idx]

        left_angles = _solve_arm(left_chain, left_target)
        right_angles = _solve_arm(right_chain, right_target)

        # clamp to simple joint limits [-pi, pi]
        robot_qpos[idx, 0:3] = np.clip(left_angles[:3], -np.pi, np.pi)
        robot_qpos[idx, 3:6] = np.clip(right_angles[:3], -np.pi, np.pi)

        root_pose[idx, :3] = pelvis[idx]
        root_pose[idx, 6] = 1.0  # identity quaternion

    return RetargetResult(robot_qpos=robot_qpos, root_pose=root_pose, joint_names=joint_names)
