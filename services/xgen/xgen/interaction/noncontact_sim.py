from __future__ import annotations

import numpy as np


def _simulate_ballistic(object_pose: np.ndarray, contact_indices: np.ndarray, fps: float, gravity: float, reverse: bool) -> np.ndarray:
    dt = 1.0 / fps
    updated = object_pose.copy()
    contact_set = set(int(idx) for idx in contact_indices)
    indices = range(updated.shape[0] - 1, -1, -1) if reverse else range(updated.shape[0])
    velocity = np.zeros(3)

    for idx in indices:
        if idx in contact_set:
            continue
        translation = updated[idx, :3]
        velocity[2] -= gravity * dt
        updated[idx, :3] = translation + velocity * dt
    return updated


def simulate_noncontact(
    object_pose: np.ndarray,
    contact_indices: np.ndarray,
    fps: float,
    gravity: float = 9.81,
    reverse: bool = False,
    object_size: float = 0.1,
    damping: float = 0.0,
) -> np.ndarray:
    """Physics-based non-contact simulation using MuJoCo when available."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if object_pose.shape[0] == 0:
        return object_pose
    try:
        import mujoco
    except Exception:
        return _simulate_ballistic(object_pose, contact_indices, fps, gravity, reverse)

    dt = 1.0 / fps
    model_xml = f"""
    <mujoco>
      <option timestep="{dt}" gravity="0 0 -{gravity}"/>
      <worldbody>
        <body name="object" pos="{object_pose[0,0]} {object_pose[0,1]} {object_pose[0,2]}">
          <freejoint/>
          <geom type="box" size="{object_size} {object_size} {object_size}" density="300"/>
        </body>
        <geom name="ground" type="plane" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)

    data.qpos[:7] = object_pose[0]
    if damping != 0.0:
        model.dof_damping[:] = damping if not reverse else -abs(damping)
    if reverse:
        data.qvel[:3] *= -1

    out = []
    for _ in range(object_pose.shape[0]):
        mujoco.mj_step(model, data)
        out.append(data.qpos[:7].copy())

    return np.stack(out, axis=0)


def simulate_noncontact_forward(
    object_pose: np.ndarray,
    contact_indices: np.ndarray,
    fps: float,
    gravity: float = 9.81,
    object_size: float = 0.1,
    damping: float = 0.0,
) -> np.ndarray:
    return simulate_noncontact(
        object_pose,
        contact_indices,
        fps,
        gravity=gravity,
        reverse=False,
        object_size=object_size,
        damping=damping,
    )


def simulate_noncontact_backward(
    object_pose: np.ndarray,
    contact_indices: np.ndarray,
    fps: float,
    gravity: float = 9.81,
    object_size: float = 0.1,
    damping: float = 0.0,
) -> np.ndarray:
    return simulate_noncontact(
        object_pose,
        contact_indices,
        fps,
        gravity=gravity,
        reverse=True,
        object_size=object_size,
        damping=damping,
    )
