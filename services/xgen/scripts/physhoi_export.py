from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from xgen.pose.physhoi_motion import (
    smplx_npz_to_physhoi_motion,
    load_physhoi_motion,
    validate_physhoi_motion,
    summarize_physhoi_motion,
)


def _load_optional_array(path: Optional[str], frames: int) -> Optional[np.ndarray]:
    if not path:
        return None
    arr = np.load(path)
    if arr.ndim == 1:
        arr = np.repeat(arr.reshape(1, -1), frames, axis=0)
    if arr.shape[0] != frames:
        raise ValueError(f"Array {path} frames mismatch: {arr.shape[0]} != {frames}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SMPL-X NPZ to PhysHOI motion .pt")
    parser.add_argument("--smplx-npz", type=str, help="Path to SMPL-X NPZ")
    parser.add_argument("--model-dir", type=str, help="Path to SMPL-X model directory")
    parser.add_argument("--output", type=str, required=True, help="Output .pt path")
    parser.add_argument("--obj-pos", type=str, help="Optional npy for object pos (T,3)")
    parser.add_argument("--obj-rot", type=str, help="Optional npy for object rot expmap (T,3)")
    parser.add_argument("--contact", type=str, help="Optional npy for contact (T,1 or T)")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing .pt and exit")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.validate_only:
        tensor = load_physhoi_motion(output_path)
        validate_physhoi_motion(tensor)
        summary = summarize_physhoi_motion(tensor)
        print("Validation OK:", summary)
        return

    if not args.smplx_npz or not args.model_dir:
        raise SystemExit("--smplx-npz and --model-dir are required unless --validate-only")

    smplx_npz = Path(args.smplx_npz)
    model_dir = Path(args.model_dir)
    data = np.load(smplx_npz)
    frames = data["global_orient"].shape[0]

    obj_pos = _load_optional_array(args.obj_pos, frames)
    obj_rot = _load_optional_array(args.obj_rot, frames)
    contact = _load_optional_array(args.contact, frames)
    if contact is not None and contact.shape[1] != 1:
        contact = contact[:, :1]

    smplx_npz_to_physhoi_motion(
        smplx_npz=smplx_npz,
        output_path=output_path,
        model_dir=model_dir,
        obj_pos=obj_pos,
        obj_rot=obj_rot,
        contact=contact,
    )

    tensor = load_physhoi_motion(output_path)
    validate_physhoi_motion(tensor)
    summary = summarize_physhoi_motion(tensor)
    print("Export complete:", summary)


if __name__ == "__main__":
    main()
