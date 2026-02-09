from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import smplx

from .smplx_convert import ensure_smplx_fields, SmplxSequence


def smplx_npz_to_physhoi_motion(
    smplx_npz: Path,
    output_path: Path,
    model_dir: Path,
    obj_pos: Optional[np.ndarray] = None,
    obj_rot: Optional[np.ndarray] = None,
    contact: Optional[np.ndarray] = None,
) -> Path:
    """
    Convert SMPL-X NPZ into PhysHOI motion tensor (.pt).
    This produces a (T, 331) tensor compatible with PhysHOI's _load_motion.
    """
    data = np.load(smplx_npz)
    seq = ensure_smplx_fields({k: data[k] for k in data.files})

    device = "cpu"
    smplx_model = smplx.create(
        model_path=str(model_dir),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=seq.global_orient.shape[0],
    ).to(device)

    with torch.no_grad():
        out = smplx_model(
            global_orient=torch.tensor(seq.global_orient, device=device),
            body_pose=torch.tensor(seq.body_pose, device=device),
            betas=torch.tensor(seq.betas, device=device),
            transl=torch.tensor(seq.transl, device=device),
            left_hand_pose=torch.tensor(seq.left_hand_pose, device=device),
            right_hand_pose=torch.tensor(seq.right_hand_pose, device=device),
            jaw_pose=torch.tensor(seq.jaw_pose, device=device),
            leye_pose=torch.tensor(seq.leye_pose, device=device),
            reye_pose=torch.tensor(seq.reye_pose, device=device),
            expression=torch.tensor(seq.expression, device=device),
        )
        joints = out.joints.detach().cpu().numpy()  # (T, J, 3)

    frames = seq.global_orient.shape[0]
    hoi = np.zeros((frames, 331), dtype=np.float32)

    # root pos/rot
    hoi[:, 0:3] = seq.transl
    hoi[:, 3:6] = seq.global_orient

    # dof_pos: body + hands (51*3 = 153)
    dof_pos = np.concatenate([seq.body_pose, seq.left_hand_pose, seq.right_hand_pose], axis=1)
    if dof_pos.shape[1] != 153:
        raise ValueError(f"Expected dof_pos dim 153, got {dof_pos.shape[1]}")
    hoi[:, 9:9 + 153] = dof_pos

    # body positions: first 52 joints
    hoi[:, 162:162 + 52 * 3] = joints[:, :52, :].reshape(frames, -1)

    # object pose placeholders
    if obj_pos is not None:
        hoi[:, 318:321] = obj_pos
    if obj_rot is not None:
        hoi[:, 321:324] = obj_rot

    # contact placeholder
    if contact is not None:
        hoi[:, 330:331] = contact.reshape(frames, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(hoi), output_path)
    return output_path


def load_physhoi_motion(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def validate_physhoi_motion(tensor: torch.Tensor) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != 331:
        raise ValueError(f"PhysHOI motion must be (T, 331), got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("PhysHOI motion contains NaN/Inf values")
    # contact channel should be in [0,1] if present
    contact = tensor[:, 330]
    if torch.any((contact < 0) | (contact > 1)):
        raise ValueError("PhysHOI contact channel has values outside [0, 1]")


def summarize_physhoi_motion(tensor: torch.Tensor) -> dict:
    return {
        "frames": int(tensor.shape[0]),
        "dims": int(tensor.shape[1]),
        "root_pos_mean": tensor[:, 0:3].mean(dim=0).tolist(),
        "root_pos_std": tensor[:, 0:3].std(dim=0).tolist(),
        "contact_mean": float(tensor[:, 330].mean().item()),
    }
