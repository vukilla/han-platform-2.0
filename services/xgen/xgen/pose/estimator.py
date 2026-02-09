from pathlib import Path

from .gvhmr_pose import estimate_smplx_from_video


def estimate_smpl_from_video(video_path: Path, output_path: Path) -> Path:
    """
    Run GVHMR to estimate SMPL-X parameters from a monocular video.
    Output is a compressed NPZ with SMPL-X params.
    """
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return estimate_smplx_from_video(video_path, output_dir)
