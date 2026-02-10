from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import shutil


def _repo_root() -> Path:
    # apps/api/app/gpu/<file>.py -> repo root is parents[4]
    return Path(__file__).resolve().parents[4]


def _resolve_gvhmr_root() -> Path:
    env = os.environ.get("GVHMR_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "external" / "gvhmr"


def _resolve_heavy_checkpoints_root() -> Path:
    return _repo_root() / "external" / "humanoid-projects" / "GVHMR" / "inputs" / "checkpoints"


def _ensure_checkpoints(gvhmr_root: Path, *, require_dpvo: bool) -> None:
    """Ensure `external/gvhmr/inputs/checkpoints` exists by linking staged checkpoints.

    GVHMR expects checkpoints under `<gvhmr_root>/inputs/checkpoints`. We stage large files under:
    `external/humanoid-projects/GVHMR/inputs/checkpoints`.
    """
    expected = gvhmr_root / "inputs" / "checkpoints"
    required = expected / "gvhmr" / "gvhmr_siga24_release.ckpt"
    if required.exists():
        return

    heavy = _resolve_heavy_checkpoints_root()
    if not heavy.exists():
        raise FileNotFoundError(
            "GVHMR heavy checkpoints not found. Expected staged checkpoints under: "
            f"{heavy}. See docs/GVHMR.md."
        )

    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists():
        # Don't delete user data. If the directory exists but is incomplete, let GVHMR fail with a clear error.
        return

    try:
        os.symlink(heavy, expected, target_is_directory=True)
    except OSError:
        # Symlinks are often unavailable on Windows without Developer Mode or Administrator privileges.
        # Fall back to copying the required checkpoints. This is slower but robust.
        required_files = [
            Path("gvhmr") / "gvhmr_siga24_release.ckpt",
            Path("vitpose") / "vitpose-h-multi-coco.pth",
            Path("hmr2") / "epoch=10-step=25000.ckpt",
            Path("yolo") / "yolov8x.pt",
        ]
        # DPVO is optional in GVHMR. Only require it when explicitly enabled.
        if require_dpvo:
            required_files.insert(1, Path("dpvo") / "dpvo.pth")

        expected.mkdir(parents=True, exist_ok=True)
        for rel in required_files:
            src = heavy / rel
            dst = expected / rel
            if dst.exists():
                continue
            if not src.exists():
                raise FileNotFoundError(f"Missing checkpoint file: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _find_hmr4d_results(output_root: Path, video_stem: str) -> Path:
    candidate = output_root / video_stem / "hmr4d_results.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"GVHMR results not found at {candidate}")


def _load_smpl_params(results_path: Path) -> dict[str, Any]:
    # Import torch lazily so Mac containers don't need it.
    import torch

    pred = torch.load(results_path, map_location="cpu")
    smpl_params = pred.get("smpl_params_global", {})
    if not smpl_params:
        raise ValueError("GVHMR results missing smpl_params_global")
    smpl_np = {k: v.detach().cpu().numpy() for k, v in smpl_params.items()}
    return {"smpl_params_global": smpl_np}


def run_gvhmr(video_path: Path, output_dir: Path, *, static_cam: bool, use_dpvo: bool, f_mm: int | None) -> dict[str, str]:
    gvhmr_root = _resolve_gvhmr_root()
    if not gvhmr_root.exists():
        raise FileNotFoundError(
            "GVHMR repo not found. Clone into external/gvhmr or set GVHMR_ROOT.\n"
            f"Expected: {gvhmr_root}"
        )
    _ensure_checkpoints(gvhmr_root, require_dpvo=bool(use_dpvo))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir / "gvhmr"
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(gvhmr_root / "tools" / "demo" / "demo.py"),
        "--video",
        str(video_path),
        "--output_root",
        str(output_root),
    ]
    # Requires our patched GVHMR demo that supports skipping rendering (avoids pytorch3d renderer dependency).
    cmd.append("--skip_render")
    if static_cam:
        cmd.append("-s")
    if use_dpvo:
        cmd.append("--use_dpvo")
    if f_mm is not None:
        cmd.extend(["--f_mm", str(int(f_mm))])

    env = dict(os.environ)
    # Ensure GVHMR can import its own modules and our local pytorch3d stub.
    pythonpath = env.get("PYTHONPATH", "")
    gvhmr_pp = str(gvhmr_root)
    env["PYTHONPATH"] = gvhmr_pp if not pythonpath else (gvhmr_pp + os.pathsep + pythonpath)

    # GVHMR can be very verbose; keep stdout clean so the parent process can parse our JSON result.
    gvhmr_log = output_dir / "gvhmr.log"
    with gvhmr_log.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, cwd=str(gvhmr_root), env=env, stdout=handle, stderr=handle)

    results_path = _find_hmr4d_results(output_root, video_path.stem)
    payload = _load_smpl_params(results_path)

    import numpy as np

    npz_path = output_dir / f"{video_path.stem}_gvhmr_smplx.npz"
    np.savez_compressed(npz_path, **payload["smpl_params_global"])

    meta_path = output_dir / f"{video_path.stem}_gvhmr_meta.json"
    meta = {
        "video": str(video_path),
        "results_path": str(results_path),
        "output_npz": str(npz_path),
        "gvhmr_log": str(gvhmr_log),
        "static_cam": bool(static_cam),
        "use_dpvo": bool(use_dpvo),
        "f_mm": int(f_mm) if f_mm is not None else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
        "results_path": str(results_path),
        "gvhmr_log_path": str(gvhmr_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GVHMR demo and export SMPL-X NPZ.")
    parser.add_argument("--video", required=True, help="Path to an input MP4 file")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs into")
    parser.add_argument("--static-cam", action="store_true", help="Skip DPVO (recommended for static camera)")
    parser.add_argument("--use-dpvo", action="store_true", help="Enable DPVO visual odometry")
    parser.add_argument("--f-mm", type=int, default=None, help="Focal length in mm for fullframe camera")
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    result = run_gvhmr(
        video,
        out_dir,
        static_cam=bool(args.static_cam),
        use_dpvo=bool(args.use_dpvo),
        f_mm=args.f_mm,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
