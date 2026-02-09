from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
try:
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore[assignment]

try:
    from ikpy.chain import Chain
except Exception:  # pragma: no cover
    Chain = None  # type: ignore[assignment]


def force_closure_refine(
    object_pose: np.ndarray,
    contact_indices: np.ndarray,
    anchor_positions: np.ndarray | None = None,
    max_offset: float = 0.05,
) -> np.ndarray:
    """Constraint-based refinement in contact phase.

    Enforces:
    - object translation remains close to anchor offset at first contact
    - object stays above ground plane (z >= 0)
    """
    if contact_indices.size == 0:
        return object_pose
    refined = object_pose.copy()
    coords = refined[contact_indices, :3]

    if anchor_positions is not None:
        first_idx = int(contact_indices[0])
        base_offset = refined[first_idx, :3] - anchor_positions[first_idx]
        for idx in contact_indices:
            desired = anchor_positions[idx] + base_offset
            delta = desired - refined[idx, :3]
            delta = np.clip(delta, -max_offset, max_offset)
            refined[idx, :3] = refined[idx, :3] + delta

    refined[contact_indices, 2] = np.maximum(refined[contact_indices, 2], 0.0)

    if coords.shape[0] > 2:
        window = 3
        smoothed = refined[contact_indices, :3].copy()
        for idx in range(smoothed.shape[0]):
            start = max(0, idx - window)
            end = min(smoothed.shape[0], idx + window + 1)
            smoothed[idx] = smoothed[start:end].mean(axis=0)
        refined[contact_indices, :3] = smoothed
    return refined


def solve_force_closure_qp(
    contact_points: np.ndarray,
    contact_normals: np.ndarray | None = None,
    desired_wrench: np.ndarray | None = None,
    friction_coeff: float = 0.5,
) -> np.ndarray:
    """
    Solve a force-closure QP for contact forces.
    Returns forces stacked as (N, 3).
    """
    if cp is None:
        # Allow CPU-only environments/tests to run without pulling in heavy solvers.
        return np.zeros((contact_points.shape[0], 3), dtype=np.float32)
    num_contacts = contact_points.shape[0]
    if num_contacts == 0:
        return np.zeros((0, 3), dtype=np.float32)

    desired_wrench = desired_wrench if desired_wrench is not None else np.zeros(6)
    contact_normals = contact_normals if contact_normals is not None else np.tile(
        np.array([0.0, 0.0, 1.0]), (num_contacts, 1)
    )

    # Grasp matrix G (6 x 3N)
    G = np.zeros((6, 3 * num_contacts), dtype=np.float32)
    for i, p in enumerate(contact_points):
        px, py, pz = p
        # force component
        G[0:3, 3 * i : 3 * i + 3] = np.eye(3)
        # torque component (p x f)
        G[3:, 3 * i : 3 * i + 3] = np.array(
            [
                [0, -pz, py],
                [pz, 0, -px],
                [-py, px, 0],
            ],
            dtype=np.float32,
        )

    f = cp.Variable((3 * num_contacts,))
    objective = cp.Minimize(cp.sum_squares(G @ f - desired_wrench) + 1e-4 * cp.sum_squares(f))
    constraints = []
    for i in range(num_contacts):
        fx = f[3 * i + 0]
        fy = f[3 * i + 1]
        fz = f[3 * i + 2]
        constraints += [
            fz >= 0,
            fx <= friction_coeff * fz,
            fx >= -friction_coeff * fz,
            fy <= friction_coeff * fz,
            fy >= -friction_coeff * fz,
        ]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, warm_start=True)
        forces = f.value.reshape(num_contacts, 3)
    except Exception:
        forces = np.zeros((num_contacts, 3), dtype=np.float32)
    return forces.astype(np.float32)


@dataclass
class URDFJoint:
    name: str
    parent: str
    child: str


def _parse_urdf_joints(urdf_path: str | Path) -> List[URDFJoint]:
    root = ET.parse(str(urdf_path)).getroot()
    joints: List[URDFJoint] = []
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        joints.append(URDFJoint(name=name, parent=parent, child=child))
    return joints


def _infer_root_link(joints: List[URDFJoint]) -> str:
    parents = {j.parent for j in joints}
    children = {j.child for j in joints}
    roots = sorted(list(parents - children))
    if not roots:
        raise ValueError("Could not infer URDF root link (cycle?)")
    return roots[0]


def _base_elements_for_tip(urdf_path: str | Path, tip_link: str, root_link: Optional[str] = None) -> List[str]:
    """Return ikpy base_elements list (alternating link/joint/link...) from root_link to tip_link."""
    joints = _parse_urdf_joints(urdf_path)
    root_link = root_link or _infer_root_link(joints)

    adj: Dict[str, List[Tuple[str, str]]] = {}
    for j in joints:
        adj.setdefault(j.parent, []).append((j.name, j.child))

    # BFS over links.
    queue = [root_link]
    parent: Dict[str, Optional[str]] = {root_link: None}
    via_joint: Dict[str, str] = {}
    while queue:
        cur = queue.pop(0)
        if cur == tip_link:
            break
        for jname, child in adj.get(cur, []):
            if child in parent:
                continue
            parent[child] = cur
            via_joint[child] = jname
            queue.append(child)

    if tip_link not in parent:
        raise ValueError(f"tip_link {tip_link} not reachable from root_link {root_link}")

    # Reconstruct link path root -> tip.
    links: List[str] = []
    joints_path: List[str] = []
    cur = tip_link
    while cur != root_link:
        links.append(cur)
        joints_path.append(via_joint[cur])
        cur = parent[cur]  # type: ignore[assignment]
    links.append(root_link)
    links.reverse()
    joints_path.reverse()

    base_elements: List[str] = [links[0]]
    for i, jname in enumerate(joints_path):
        base_elements.append(jname)
        base_elements.append(links[i + 1])
    return base_elements


def _build_ik_chain(
    urdf_path: str | Path,
    tip_link: str,
    *,
    root_link: Optional[str] = None,
    active_joints: Optional[Iterable[str]] = None,
) -> "Chain":
    if Chain is None:
        raise ImportError("ikpy is required for IK refinement (missing dependency)")
    base_elements = _base_elements_for_tip(urdf_path, tip_link=tip_link, root_link=root_link)
    # ikpy emits a noisy warning for fixed base links before we can adjust the active mask.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Link Base link .* is of type 'fixed' but set as active.*",
            category=UserWarning,
        )
        chain = Chain.from_urdf_file(
            str(urdf_path),
            base_elements=base_elements,
            base_element_type="link",
            symbolic=False,
        )
    if active_joints is not None:
        allowed = set(active_joints)
        # ikpy expects an active mask per link (URDF joint), including base.
        mask = [False] * len(chain.links)
        for i, link in enumerate(chain.links):
            if i == 0:
                mask[i] = False
                continue
            mask[i] = link.name in allowed
        chain.active_links_mask = np.asarray(mask, dtype=bool)

    # Always mark fixed joints (including the synthetic base) as inactive.
    try:
        mask = np.asarray(chain.active_links_mask, dtype=bool).copy()
        if mask.size > 0:
            mask[0] = False
        for i, link in enumerate(chain.links):
            if getattr(link, "joint_type", None) == "fixed":
                mask[i] = False
        chain.active_links_mask = mask
    except Exception:
        pass
    return chain


def refine_contact_ik(
    *,
    urdf_path: str | Path,
    joint_names: List[str],
    robot_qpos: np.ndarray,
    contact_indices: np.ndarray,
    tip_targets: Dict[str, np.ndarray],
    root_link: Optional[str] = None,
    active_joints_by_tip: Optional[Dict[str, Iterable[str]]] = None,
    max_joint_delta: float = 0.35,
    smooth_window: int = 0,
    method: str = "dls",
    max_iters: int = 25,
    tol: float = 1e-4,
    damping: float = 1e-2,
    jac_eps: float = 1e-4,
) -> np.ndarray:
    """Refine robot joint angles during contact frames using CPU IK.

    This is a kinematics-only refinement pass, intended to move end-effectors
    (e.g. hands) to better match contact-phase targets before running any physics.

    Inputs:
    - robot_qpos: (T, nq) joint angles (radians) matching `joint_names`.
    - contact_indices: frame indices to refine.
    - tip_targets: map tip_link -> (T, 3) target tip positions (same coordinate frame as URDF root).

    Notes:
    - This does not modify object_pose. Pair with `force_closure_refine`/QP for object refinement.
    - For general URDFs, tip chains are discovered from the URDF tree (no hardcoded paths).
    """
    if contact_indices.size == 0:
        return robot_qpos
    if robot_qpos.ndim != 2:
        raise ValueError("robot_qpos must be (T, nq)")
    t_steps, nq = robot_qpos.shape
    if len(joint_names) != nq:
        raise ValueError("joint_names length mismatch")
    for tip, targets in tip_targets.items():
        if targets.shape[0] != t_steps or targets.shape[1] != 3:
            raise ValueError(f"tip_targets[{tip}] must be (T, 3)")

    jidx = {name: i for i, name in enumerate(joint_names)}
    refined = robot_qpos.copy()
    contact = np.asarray(contact_indices, dtype=int)

    chains: Dict[str, "Chain"] = {}
    for tip in tip_targets.keys():
        active = None
        if active_joints_by_tip is not None:
            active = active_joints_by_tip.get(tip)
        else:
            # By default, only solve over joints we actually control (avoid URDF joints not in joint_names).
            active = joint_names
        chains[tip] = _build_ik_chain(urdf_path, tip_link=tip, root_link=root_link, active_joints=active)

    if method not in ("ikpy", "dls"):
        raise ValueError("method must be 'ikpy' or 'dls'")

    for tip, chain in chains.items():
        targets = tip_targets[tip]
        # Cache active indices once per tip.
        try:
            active_mask = np.asarray(chain.active_links_mask, dtype=bool)
        except Exception:
            active_mask = np.ones((len(chain.links),), dtype=bool)
            if active_mask.size > 0:
                active_mask[0] = False
        active_idx = np.nonzero(active_mask)[0].astype(int)

        for fi in contact:
            if fi < 0 or fi >= t_steps:
                continue
            # Build initial position vector (len(chain.links)), including base at 0.
            q_init = np.zeros((len(chain.links),), dtype=np.float32)
            for li, link in enumerate(chain.links):
                if li == 0:
                    continue
                if link.name in jidx:
                    q_init[li] = float(refined[fi, jidx[link.name]])

            if method == "ikpy":
                sol = chain.inverse_kinematics(targets[fi], initial_position=q_init)
            else:
                # Damped-least-squares IK (Gauss-Newton) with numerical Jacobian, CPU-only.
                q = q_init.astype(np.float32).copy()
                tgt = np.asarray(targets[fi], dtype=np.float32).reshape(3)
                for _ in range(max_iters):
                    fk = chain.forward_kinematics(q)
                    pos = np.asarray(fk[:3, 3], dtype=np.float32).reshape(3)
                    err = tgt - pos
                    if float(np.linalg.norm(err)) < tol:
                        break
                    if active_idx.size == 0:
                        break
                    # Numerical Jacobian J: (3, m) for active joints.
                    J = np.zeros((3, int(active_idx.size)), dtype=np.float32)
                    for ci, ji in enumerate(active_idx.tolist()):
                        q2 = q.copy()
                        q2[int(ji)] += float(jac_eps)
                        fk2 = chain.forward_kinematics(q2)
                        pos2 = np.asarray(fk2[:3, 3], dtype=np.float32).reshape(3)
                        J[:, ci] = (pos2 - pos) / float(jac_eps)
                    # dq = J^T (J J^T + lambda I)^-1 err
                    JJt = J @ J.T
                    JJt = JJt + float(damping) * np.eye(3, dtype=np.float32)
                    dq = J.T @ np.linalg.solve(JJt, err)
                    dq = np.clip(dq, -max_joint_delta, max_joint_delta).astype(np.float32)
                    q[active_idx] = q[active_idx] + dq
                sol = q

            # Write back solution for joints present in robot_qpos, with per-step delta clamp.
            for li, link in enumerate(chain.links):
                if li == 0:
                    continue
                if link.name not in jidx:
                    continue
                idx = jidx[link.name]
                prev = float(refined[fi, idx])
                new = float(sol[li])
                delta = float(np.clip(new - prev, -max_joint_delta, max_joint_delta))
                refined[fi, idx] = prev + delta

    if smooth_window > 0:
        # Simple moving average smoothing across the contact window for all joints.
        w = int(smooth_window)
        for j in range(nq):
            series = refined[contact, j]
            if series.size == 0:
                continue
            smoothed = series.copy()
            for k in range(series.shape[0]):
                start = max(0, k - w)
                end = min(series.shape[0], k + w + 1)
                smoothed[k] = float(np.mean(series[start:end]))
            refined[contact, j] = smoothed

    return refined
