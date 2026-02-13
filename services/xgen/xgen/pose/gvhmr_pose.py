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


def _looks_like_gvhmr_repo(path: Path) -> bool:
    demo_script = path / "tools" / "demo" / "demo.py"
    if demo_script.exists():
        return True
    # Some installs may have `tools/demo` mounted without the script present at init.
    if (path / "tools" / "demo").is_dir():
        return True

    markers = [
        "tools",
        "hmr4d",
        "hmr4d_results",
        "inputs",
        "models",
        "model",
        "configs",
        "requirements.txt",
    ]
    if any((path / marker).exists() for marker in markers):
        return True

    return False


def _candidate_gvhmr_roots() -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = []

    def _add(candidate: Path | None) -> None:
        if candidate is None:
            return
        candidates.append(candidate)

    def _scan_for_demo_paths(base: Path) -> None:
        if not base.exists():
            return
        try:
            for demo in base.rglob("tools/demo/demo.py"):
                _add(demo.parents[2])
        except OSError:
            # Permission errors on mounted volumes are non-fatal here.
            return

    env = os.environ.get("GVHMR_ROOT")
    if env:
        env_root = Path(env).expanduser().resolve()
        _add(env_root)
        _add(env_root / "gvhmr")
        _add(env_root / "GVHMR")
        _add(env_root / "humanoid-projects" / "GVHMR")
        _add(env_root / "humanoid-projects" / "gvhmr")

    mount_bases = [
        here.parent,
        here.parents[1],
        here.parents[2],
        here.parents[3],
        here.parents[4],
        Path("/app"),
        Path("/app/app"),
        Path("/workspace"),
        Path("/opt/han-platform"),
        Path("/home/cheema/han-platform"),
        Path("/home"),
        Path("/opt"),
    ]
    for base in mount_bases:
        if not base.exists():
            continue
        ext = base / "external"
        if not ext.exists():
            continue
        _add(ext / "gvhmr")
        _add(ext / "GVHMR")
        _add(ext / "humanoid-projects" / "GVHMR")
        _add(ext / "humanoid-projects" / "gvhmr")

    _scan_for_demo_paths(Path("/app/external"))
    _scan_for_demo_paths(Path("/app"))
    _scan_for_demo_paths(Path("/workspace"))
    _scan_for_demo_paths(Path("/opt/han-platform"))
    _scan_for_demo_paths(Path("/home/cheema/han-platform"))
    _scan_for_demo_paths(Path("/home"))
    _scan_for_demo_paths(Path("/opt"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = candidate.resolve()
        except OSError:
            normalized = candidate
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _resolve_gvhmr_root() -> Path:
    for candidate in _candidate_gvhmr_roots():
        if candidate.exists() and _looks_like_gvhmr_repo(candidate):
            return candidate
    # If candidate paths fail shallow checks, inspect existing dirs for nested GVHMR
    # checkouts (e.g. `<root>/humanoid-projects/GVHMR`).
    for candidate in _candidate_gvhmr_roots():
        if not candidate.exists():
            continue
        try:
            for demo in candidate.rglob("tools/demo/demo.py"):
                parent = demo.parents[2]
                if parent.exists() and _looks_like_gvhmr_repo(parent):
                    return parent
        except OSError:
            continue
    repo_root = Path(__file__).resolve().parents[4]
    for legacy in [
        repo_root / "external" / "GVHMR",
        repo_root / "external" / "gvhmr",
    ]:
        if legacy.exists():
            return legacy
    return repo_root / "external" / "GVHMR"


def _resolve_heavy_checkpoints_root() -> Path:
    env = os.environ.get("GVHMR_CHECKPOINTS_ROOT")
    if env:
        override = Path(env).expanduser().resolve()
        if override.exists() and override.is_dir():
            candidates = [
                override,
                override / "inputs" / "checkpoints",
                override / "GVHMR" / "inputs" / "checkpoints",
                override / "gvhmr" / "inputs" / "checkpoints",
                override / "humanoid-projects" / "GVHMR" / "inputs" / "checkpoints",
                override / "humanoid-projects" / "gvhmr" / "inputs" / "checkpoints",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate

    for base in [
        Path("/app"),
        Path("/app/app"),
        Path("/workspace"),
        Path("/opt/han-platform"),
    ]:
        if not base.exists():
            continue
        for rel in [
            Path("external") / "GVHMR" / "inputs" / "checkpoints",
            Path("external") / "humanoid-projects" / "GVHMR" / "inputs" / "checkpoints",
            Path("external") / "humanoid-projects" / "gvhmr" / "inputs" / "checkpoints",
            Path("external") / "gvhmr" / "inputs" / "checkpoints",
        ]:
            candidate = (base / rel).resolve()
            if candidate.exists():
                return candidate

    # Large checkpoint staging (kept out of the GVHMR repo clone).
    return Path(__file__).resolve().parents[4] / "external" / "GVHMR" / "inputs" / "checkpoints"


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _remove_dangling_path(path: Path) -> None:
    """
    Remove dangling link/junction artifacts only.

    We intentionally avoid deleting real existing directories/files.
    """
    if path.exists() or not _path_lexists(path):
        return
    try:
        path.unlink()
    except OSError:
        if os.name == "nt":
            # Broken directory junctions on Windows may require rmdir.
            subprocess.run(
                ["cmd", "/c", "rmdir", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
        if _path_lexists(path):
            raise


def _link_checkpoint_root(src: Path, dst: Path) -> None:
    """
    Create a directory link for checkpoints.

    On Windows we fall back to a junction if symlink creation is not available.
    """
    try:
        os.symlink(src, dst, target_is_directory=True)
        return
    except FileExistsError:
        # Another worker may have created it between existence check and link call.
        if dst.exists() or _path_lexists(dst):
            return
        raise
    except (OSError, NotImplementedError):
        if os.name != "nt":
            raise

    proc = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 and not (dst.exists() or _path_lexists(dst)):
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit={proc.returncode}"
        raise RuntimeError(f"Failed to create checkpoint junction {dst} -> {src}: {detail}")


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
    # If we renamed/moved the repo, a dangling link from the old path may remain.
    _remove_dangling_path(expected)
    if expected.exists():
        return
    _link_checkpoint_root(heavy, expected)


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
    if not gvhmr_root.exists() or not _looks_like_gvhmr_repo(gvhmr_root):
        checked = "\n".join(f"- {p}" for p in _candidate_gvhmr_roots())
        raise FileNotFoundError(
            "GVHMR repo not found or incomplete. Set GVHMR_ROOT to a valid GVHMR checkout.\n"
            f"Expected: {gvhmr_root}\n"
            "Checked:\n"
            f"{checked}"
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
