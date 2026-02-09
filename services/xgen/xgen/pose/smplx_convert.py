from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import smplx


@dataclass
class SmplxSequence:
    global_orient: np.ndarray
    body_pose: np.ndarray
    betas: np.ndarray
    transl: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    expression: np.ndarray


def _zeros(shape):
    return np.zeros(shape, dtype=np.float32)


def ensure_smplx_fields(params: dict[str, np.ndarray]) -> SmplxSequence:
    global_orient = params.get("global_orient")
    body_pose = params.get("body_pose")
    betas = params.get("betas")
    transl = params.get("transl")
    if global_orient is None or body_pose is None or betas is None or transl is None:
        raise ValueError("Missing required SMPL/SMPL-X keys: global_orient, body_pose, betas, transl")

    frames = global_orient.shape[0]
    left_hand_pose = params.get("left_hand_pose", _zeros((frames, 45)))
    right_hand_pose = params.get("right_hand_pose", _zeros((frames, 45)))
    jaw_pose = params.get("jaw_pose", _zeros((frames, 3)))
    leye_pose = params.get("leye_pose", _zeros((frames, 3)))
    reye_pose = params.get("reye_pose", _zeros((frames, 3)))
    expression = params.get("expression", _zeros((frames, 10)))

    return SmplxSequence(
        global_orient=global_orient.astype(np.float32),
        body_pose=body_pose.astype(np.float32),
        betas=betas.astype(np.float32),
        transl=transl.astype(np.float32),
        left_hand_pose=left_hand_pose.astype(np.float32),
        right_hand_pose=right_hand_pose.astype(np.float32),
        jaw_pose=jaw_pose.astype(np.float32),
        leye_pose=leye_pose.astype(np.float32),
        reye_pose=reye_pose.astype(np.float32),
        expression=expression.astype(np.float32),
    )


def save_smplx_npz(output_path: Path, seq: SmplxSequence) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        global_orient=seq.global_orient,
        body_pose=seq.body_pose,
        betas=seq.betas,
        transl=seq.transl,
        left_hand_pose=seq.left_hand_pose,
        right_hand_pose=seq.right_hand_pose,
        jaw_pose=seq.jaw_pose,
        leye_pose=seq.leye_pose,
        reye_pose=seq.reye_pose,
        expression=seq.expression,
    )
    return output_path


def convert_smpl_to_smplx(
    smpl_npz: Path,
    output_path: Path,
    model_dir: Path,
    num_iters: int = 15,
    device: str = "cpu",
) -> Path:
    """
    Fit SMPL parameters to SMPL-X by matching joints via gradient descent.
    Requires SMPL/SMPL-X model files in model_dir.
    """
    data = np.load(smpl_npz)
    smpl_params = {
        "global_orient": data["global_orient"],
        "body_pose": data["body_pose"],
        "betas": data["betas"],
        "transl": data["transl"],
    }

    smpl_model = smplx.create(
        model_path=str(model_dir),
        model_type="smpl",
        gender="neutral",
        use_pca=False,
        batch_size=1,
    ).to(device)
    smplx_model = smplx.create(
        model_path=str(model_dir),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=1,
    ).to(device)

    frames = smpl_params["global_orient"].shape[0]
    results = []
    for idx in range(frames):
        target = {
            "global_orient": torch.tensor(smpl_params["global_orient"][idx: idx + 1], device=device),
            "body_pose": torch.tensor(smpl_params["body_pose"][idx: idx + 1], device=device),
            "betas": torch.tensor(smpl_params["betas"][idx: idx + 1], device=device),
            "transl": torch.tensor(smpl_params["transl"][idx: idx + 1], device=device),
        }
        with torch.no_grad():
            smpl_out = smpl_model(**target)
            target_joints = smpl_out.joints[:, :22].detach()

        # Initialize SMPL-X with SMPL params
        smplx_params = {
            "global_orient": target["global_orient"].clone().requires_grad_(True),
            "body_pose": target["body_pose"].clone().requires_grad_(True),
            "betas": target["betas"].clone().requires_grad_(True),
            "transl": target["transl"].clone().requires_grad_(True),
            "left_hand_pose": torch.zeros((1, 45), device=device, requires_grad=True),
            "right_hand_pose": torch.zeros((1, 45), device=device, requires_grad=True),
            "jaw_pose": torch.zeros((1, 3), device=device),
            "leye_pose": torch.zeros((1, 3), device=device),
            "reye_pose": torch.zeros((1, 3), device=device),
            "expression": torch.zeros((1, 10), device=device),
        }

        optimizer = torch.optim.Adam(
            [
                smplx_params["global_orient"],
                smplx_params["body_pose"],
                smplx_params["betas"],
                smplx_params["transl"],
                smplx_params["left_hand_pose"],
                smplx_params["right_hand_pose"],
            ],
            lr=1e-2,
        )
        for _ in range(num_iters):
            optimizer.zero_grad()
            out = smplx_model(
                global_orient=smplx_params["global_orient"],
                body_pose=smplx_params["body_pose"],
                betas=smplx_params["betas"],
                transl=smplx_params["transl"],
                left_hand_pose=smplx_params["left_hand_pose"],
                right_hand_pose=smplx_params["right_hand_pose"],
                jaw_pose=smplx_params["jaw_pose"],
                leye_pose=smplx_params["leye_pose"],
                reye_pose=smplx_params["reye_pose"],
                expression=smplx_params["expression"],
            )
            loss = torch.nn.functional.mse_loss(out.joints[:, :22], target_joints)
            loss.backward()
            optimizer.step()

        results.append({k: v.detach().cpu().numpy()[0] for k, v in smplx_params.items()})

    stacked = {k: np.stack([r[k] for r in results], axis=0) for k in results[0].keys()}
    seq = ensure_smplx_fields(stacked)
    return save_smplx_npz(output_path, seq)
