from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch


def _expmap_to_quat(r: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, r / angle, np.zeros_like(r))
    half = angle * 0.5
    sin_half = np.sin(half)
    quat = np.concatenate([axis * sin_half, np.cos(half)], axis=-1)
    return quat.astype(np.float32)


def convert_physhoi_motion_to_clip(
    motion_path: str | Path,
    output_npz: str | Path,
    metadata: dict | None = None,
) -> Path:
    data = torch.load(Path(motion_path), map_location="cpu")
    if isinstance(data, torch.Tensor):
        hoi = data.detach().cpu().numpy()
    else:
        hoi = np.array(data, dtype=np.float32)

    root_pos = hoi[:, 0:3]
    root_rot = hoi[:, 3:6]
    dof_pos = hoi[:, 9:162]
    obj_pos = hoi[:, 318:321]
    obj_rot = hoi[:, 321:324]
    contact = hoi[:, 330:331]

    robot_qpos = dof_pos
    robot_qvel = np.vstack([np.zeros_like(dof_pos[:1]), np.diff(dof_pos, axis=0)])
    root_pose = np.concatenate([root_pos, _expmap_to_quat(root_rot)], axis=-1)
    object_pose = np.concatenate([obj_pos, _expmap_to_quat(obj_rot)], axis=-1)
    root_vel = np.vstack([np.zeros_like(root_pos[:1]), np.diff(root_pos, axis=0)])
    object_vel = np.vstack([np.zeros_like(obj_pos[:1]), np.diff(obj_pos, axis=0)])
    phase = np.zeros((hoi.shape[0],), dtype=np.int32)

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        robot_qpos=robot_qpos.astype(np.float32),
        robot_qvel=robot_qvel.astype(np.float32),
        root_pose=root_pose.astype(np.float32),
        root_vel=root_vel.astype(np.float32),
        object_pose=object_pose.astype(np.float32),
        object_vel=object_vel.astype(np.float32),
        contact_graph=contact.astype(np.float32),
        phase=phase,
    )

    meta = metadata or {}
    meta_path = output_npz.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return output_npz
