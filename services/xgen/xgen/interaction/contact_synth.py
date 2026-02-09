from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .force_closure_refine import force_closure_refine, refine_contact_ik, solve_force_closure_qp


def synthesize_contact(
    object_pose: np.ndarray,
    anchor_positions: np.ndarray,
    contact_indices: np.ndarray,
) -> np.ndarray:
    """Propagate anchor-object relative transform during contact frames.

    object_pose: (T, 7) [x,y,z,qx,qy,qz,qw]
    anchor_positions: (T, 3)
    contact_indices: array of frame indices in contact phase
    """
    if object_pose.shape[0] != anchor_positions.shape[0]:
        raise ValueError("object_pose and anchor_positions length mismatch")
    if object_pose.shape[1] < 3:
        raise ValueError("object_pose must include xyz translation")
    if anchor_positions.shape[1] != 3:
        raise ValueError("anchor_positions must be (T, 3)")
    if contact_indices.size == 0:
        return object_pose

    first_idx = int(contact_indices[0])
    offset = object_pose[first_idx, :3] - anchor_positions[first_idx]
    updated = object_pose.copy()
    updated[contact_indices, :3] = anchor_positions[contact_indices] + offset
    return updated


@dataclass
class ContactRefineResult:
    object_pose: np.ndarray
    contact_forces: np.ndarray | None = None
    robot_qpos: np.ndarray | None = None


def refine_contact_phase(
    object_pose: np.ndarray,
    anchor_positions: np.ndarray,
    contact_indices: np.ndarray,
    contact_points: np.ndarray | None = None,
    contact_normals: np.ndarray | None = None,
    desired_wrench: np.ndarray | None = None,
    friction_coeff: float = 0.5,
    *,
    robot_urdf_path: str | None = None,
    robot_joint_names: list[str] | None = None,
    robot_qpos: np.ndarray | None = None,
    tip_targets: dict[str, np.ndarray] | None = None,
    max_joint_delta: float = 0.35,
    smooth_window: int = 0,
    ik_method: str = "auto",  # auto|dls|ikpy|sqp
    posture_weight: float = 0.0,
) -> ContactRefineResult:
    updated = synthesize_contact(object_pose, anchor_positions, contact_indices)
    refined = force_closure_refine(updated, contact_indices, anchor_positions=anchor_positions)
    forces = None
    if contact_points is not None and contact_indices.size > 0:
        forces = solve_force_closure_qp(
            contact_points=contact_points,
            contact_normals=contact_normals,
            desired_wrench=desired_wrench,
            friction_coeff=friction_coeff,
        )

    refined_robot = None
    if robot_qpos is not None or tip_targets is not None:
        if robot_urdf_path is None or robot_joint_names is None or robot_qpos is None or tip_targets is None:
            raise ValueError(
                "robot refinement requires robot_urdf_path, robot_joint_names, robot_qpos, and tip_targets"
            )
        method = ik_method
        if method == "auto":
            method = "sqp" if len(tip_targets) > 1 else "dls"
        pw = posture_weight
        if pw <= 0.0 and method == "sqp":
            # Small posture regularization keeps solutions close to the retargeted motion.
            pw = 1e-3
        refined_robot = refine_contact_ik(
            urdf_path=robot_urdf_path,
            joint_names=robot_joint_names,
            robot_qpos=robot_qpos,
            contact_indices=contact_indices,
            tip_targets=tip_targets,
            max_joint_delta=max_joint_delta,
            smooth_window=smooth_window,
            method=method,
            posture_weight=pw,
        )

    return ContactRefineResult(object_pose=refined, contact_forces=forces, robot_qpos=refined_robot)
