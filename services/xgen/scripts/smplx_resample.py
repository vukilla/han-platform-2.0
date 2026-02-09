from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xgen.pose.smplx_convert import ensure_smplx_fields, save_smplx_npz
from xgen.pose.smplx_resample import resample_smplx_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample SMPL-X NPZ to target FPS with optional smoothing")
    parser.add_argument("--input", required=True, help="Input SMPL-X NPZ")
    parser.add_argument("--output", required=True, help="Output SMPL-X NPZ")
    parser.add_argument("--source-fps", type=float, required=True, help="Source FPS")
    parser.add_argument("--target-fps", type=float, required=True, help="Target FPS")
    parser.add_argument("--smooth-window", type=int, default=5, help="Smoothing window (frames)")
    parser.add_argument("--no-smooth-yaw", action="store_true", help="Disable yaw smoothing")
    args = parser.parse_args()

    data = np.load(args.input)
    seq = ensure_smplx_fields({k: data[k] for k in data.files})
    resampled = resample_smplx_sequence(
        seq,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        smooth_window=args.smooth_window,
        smooth_yaw=not args.no_smooth_yaw,
    )
    save_smplx_npz(Path(args.output), resampled)
    print(f"Saved resampled SMPL-X to {args.output}")


if __name__ == "__main__":
    main()
