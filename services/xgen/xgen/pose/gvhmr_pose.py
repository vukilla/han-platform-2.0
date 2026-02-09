from __future__ import annotations

from pathlib import Path
import subprocess
import os
import json
from typing import Optional

import numpy as np
import torch


class GVHMRResult:
    def __init__(self, smpl_params_global: dict[str, np.ndarray], metadata: dict[str, object]):
        self.smpl_params_global = smpl_params_global
        self.metadata = metadata


def _resolve_gvhmr_root() -> Path:
    env = os.environ.get("GVHMR_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "external" / "gvhmr"


def _resolve_heavy_checkpoints_root() -> Path:
    # Large checkpoint staging (kept out of the GVHMR repo clone).
    return (
        Path(__file__).resolve().parents[4]
        / "external"
        / "humanoid-projects"
        / "GVHMR"
        / "inputs"
        / "checkpoints"
    )


def _ensure_checkpoints(gvhmr_root: Path) -> None:
    """
    GVHMR expects checkpoints under <repo>/inputs/checkpoints.

    In this project we often stage large checkpoints under:
      external/humanoid-projects/GVHMR/inputs/checkpoints

    This function creates a symlink from the GVHMR repo clone to the staged checkpoints when needed.
    """
    expected = gvhmr_root / "inputs" / "checkpoints"
    required = expected / "gvhmr" / "gvhmr_siga24_release.ckpt"
    if required.exists():
        return

    heavy = _resolve_heavy_checkpoints_root()
    if not heavy.exists():
        return

    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists():
        # Avoid destructive behavior; if someone created an empty directory already, leave it as-is.
        return
    os.symlink(heavy, expected, target_is_directory=True)


def _find_hmr4d_results(output_root: Path, video_stem: str) -> Path:
    candidate = output_root / video_stem / "hmr4d_results.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"GVHMR results not found at {candidate}")


def _load_smpl_params(results_path: Path) -> GVHMRResult:
    pred = torch.load(results_path, map_location="cpu")
    smpl_params = pred.get("smpl_params_global", {})
    if not smpl_params:
        raise ValueError("GVHMR results missing smpl_params_global")
    smpl_np = {k: v.detach().cpu().numpy() for k, v in smpl_params.items()}
    metadata = {"source": str(results_path)}
    return GVHMRResult(smpl_params_global=smpl_np, metadata=metadata)


def estimate_smplx_from_video(
    video_path: Path,
    output_dir: Path,
    static_cam: bool = True,
    use_dpvo: bool = False,
    f_mm: Optional[int] = None,
) -> Path:
    gvhmr_root = _resolve_gvhmr_root()
    if not gvhmr_root.exists():
        raise FileNotFoundError(
            "GVHMR repo not found. Set GVHMR_ROOT or clone into external/gvhmr."
        )
    _ensure_checkpoints(gvhmr_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir / "gvhmr"
    output_root.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "python",
        str(gvhmr_root / "tools" / "demo" / "demo.py"),
        "--video",
        str(video_path),
        "--output_root",
        str(output_root),
    ]
    if static_cam:
        cmd.append("-s")
    if use_dpvo:
        cmd.append("--use_dpvo")
    if f_mm is not None:
        cmd.extend(["--f_mm", str(f_mm)])

    subprocess.run(cmd, check=True, cwd=str(gvhmr_root))

    results_path = _find_hmr4d_results(output_root, video_path.stem)
    result = _load_smpl_params(results_path)

    npz_path = output_dir / f"{video_path.stem}_gvhmr_smplx.npz"
    np.savez_compressed(npz_path, **result.smpl_params_global)
    metadata_path = output_dir / f"{video_path.stem}_gvhmr_meta.json"
    metadata = {
        "video": str(video_path),
        "results_path": str(results_path),
        "output_npz": str(npz_path),
        "static_cam": static_cam,
        "use_dpvo": use_dpvo,
        "f_mm": f_mm,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return npz_path
