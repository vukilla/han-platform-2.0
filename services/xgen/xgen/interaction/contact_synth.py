from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .force_closure_refine import force_closure_refine, solve_force_closure_qp


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


def refine_contact_phase(
    object_pose: np.ndarray,
    anchor_positions: np.ndarray,
    contact_indices: np.ndarray,
    contact_points: np.ndarray | None = None,
    contact_normals: np.ndarray | None = None,
    desired_wrench: np.ndarray | None = None,
    friction_coeff: float = 0.5,
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
    return ContactRefineResult(object_pose=refined, contact_forces=forces)
