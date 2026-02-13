from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any
import shutil


def _repo_root() -> Path:
    """Best-effort repo root resolver.

    This file can run in multiple layouts:
    - Dev checkout: `<repo>/apps/api/app/gpu/gvhmr_runner.py`
    - Docker images: `/app/app/gpu/gvhmr_runner.py` (where the API `app/` folder is copied into `/app/app`)

    Older code assumed a fixed directory depth and crashed with `IndexError` in Docker.
    """
    env = os.environ.get("HAN_REPO_ROOT") or os.environ.get("REPO_ROOT")
    if env:
        try:
            return Path(env).expanduser().resolve()
        except Exception:
            pass

    here = Path(__file__).resolve()
    # Search upward for known root markers.
    for p in [here.parent, *here.parents]:
        if (p / ".git").exists():
            return p
        if (p / "apps").is_dir() and (p / "infra").is_dir():
            return p
        # Monorepo structure: `<repo>/apps/...`
        if p.name == "apps" and (p / "api" / "app").is_dir():
            return p.parent
        if p.name == "app" and (p / "worker.py").exists():
            # Docker images often use `/app/app/...` (repo root dir name == python package dir name).
            if p.parent.name == "app":
                return p.parent
            # If the parent already looks like a project root, use it.
            if (p.parent / "external").is_dir() or (p.parent / "apps").is_dir() or (p.parent / ".git").exists():
                return p.parent
        # Generic heuristic: if the directory has `external/`, treat it as the root.
        if (p / "external").is_dir():
            return p

    # Fall back to filesystem root (avoids nonsensical relative paths under `.../gpu`).
    return Path(here.anchor)


def _candidate_gvhmr_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("GVHMR_ROOT")
    if env:
        env_root = Path(env).expanduser().resolve()
        roots.append(env_root)
        for rel in [
            Path("gvhmr"),
            Path("humanoid-projects") / "GVHMR",
            Path("GVHMR"),
            Path("humanoid-projects") / "gvhmr",
        ]:
            candidate = (env_root / rel).resolve()
            if candidate != env_root:
                roots.append(candidate)

    repo_root = _repo_root()
    here = Path(__file__).resolve()
    script_roots = [here.parent, here.parent.parent, here.parent.parent.parent]
    for base in script_roots:
        ext = base / "external"
        roots.append(base / "external" / "gvhmr")
        roots.append(base / "external" / "humanoid-projects" / "GVHMR")
        roots.append(base / "external" / "GVHMR")
        if ext.exists():
            for rel in [
                Path("gvhmr"),
                Path("humanoid-projects") / "GVHMR",
                Path("GVHMR"),
                Path("humanoid-projects") / "gvhmr",
            ]:
                roots.append(ext / rel)

    roots.append(repo_root / "external" / "gvhmr")
    roots.append(repo_root / "external" / "humanoid-projects" / "GVHMR")
    roots.append(repo_root / "external" / "GVHMR")
    # Common container mount points used by docker and SSH workers.
    for mounted_root in (
        Path("/app"),
        Path("/app/app"),
        Path("/workspace"),
        Path("/opt/han-platform"),
        Path("/home/cheema/han-platform"),
        Path("/home"),
        Path("/opt"),
    ):
        if mounted_root.exists():
            roots.append(mounted_root / "external" / "gvhmr")
            roots.append(mounted_root / "external" / "humanoid-projects" / "GVHMR")
            roots.append(mounted_root / "external" / "GVHMR")
            external_root = mounted_root / "external"
            if external_root.exists():
                for rel in [
                    Path("gvhmr"),
                    Path("humanoid-projects") / "GVHMR",
                    Path("GVHMR"),
                    Path("humanoid-projects") / "gvhmr",
                ]:
                    roots.append(external_root / rel)

    # Fallback scan for nested checkouts that may live under a non-standard mount path.
    for scan_base in [Path("/app"), Path("/app/app"), Path("/workspace"), Path("/opt/han-platform"), Path("/home/cheema/han-platform"), Path("/home"), Path("/opt")]:
        if not scan_base.exists():
            continue
        try:
            for demo in scan_base.rglob("tools/demo/demo.py"):
                candidate = demo.parents[2]
                if not candidate.name.startswith(".") and candidate.is_dir():
                    roots.append(candidate)
        except OSError:
            continue

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in roots:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _looks_like_gvhmr_repo(path: Path) -> bool:
    """Heuristic used to locate a GVHMR checkout in different local layouts.

    The repo can be mounted in different folders depending on deployment path (e.g.
    `external/gvhmr`, `external/humanoid-projects/GVHMR`, etc.). Keep this broad
    so valid checkouts are not rejected by strict folder layout assumptions.
    """
    demo_script = path / "tools" / "demo" / "demo.py"
    if demo_script.exists():
        return True

    # Some environments only have `tools/demo/` present before the script is copied.
    demo_dir = path / "tools" / "demo"
    if demo_dir.is_dir():
        return True

    markers = [
        "tools",
        "hmr4d",
        "hmr4d_results",
        "hmr4d_results.pt",
        "inputs",
        "models",
        "model",
        "configs",
        "requirements.txt",
    ]
    if any((path / marker).exists() for marker in markers):
        return True

    return False


def _resolve_gvhmr_root() -> Path:
    candidates = _candidate_gvhmr_roots()
    for candidate in candidates:
        if candidate.exists() and _looks_like_gvhmr_repo(candidate):
            return candidate

    # If no explicit repo matches, probe each existing candidate for nested checkout
    # roots that match the standard `tools/demo/demo.py` layout.
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            for demo in candidate.rglob("tools/demo/demo.py"):
                parent = demo.parents[2]
                if parent.exists() and _looks_like_gvhmr_repo(parent):
                    return parent
        except OSError:
            continue

    # Last resort for legacy/staging layouts used by some CI setups.
    for legacy in [
        _repo_root() / "external" / "GVHMR",
        _repo_root() / "external" / "humanoid-projects" / "GVHMR",
        _repo_root() / "external" / "gvhmr",
        _repo_root() / "external" / "humanoid-projects" / "gvhmr",
    ]:
        if legacy.exists() and _looks_like_gvhmr_repo(legacy):
            return legacy

    # Preserve explicit `GVHMR_ROOT` intent if provided but malformed; this keeps
    # error messages actionable.
    env = os.environ.get("GVHMR_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return candidates[0] if candidates else (_repo_root() / "external" / "humanoid-projects" / "GVHMR")


def _resolve_heavy_checkpoints_root() -> Path:
    env = os.environ.get("GVHMR_CHECKPOINTS_ROOT")
    if env:
        override = Path(env).expanduser().resolve()
        if override.exists():
            if override.is_dir():
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
    for base in (Path("/app"), Path("/app/app"), Path("/workspace"), Path("/opt/han-platform"), _repo_root()):
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
    return _repo_root() / "external" / "GVHMR" / "inputs" / "checkpoints"


def _resolve_isaac_sim_python_bat() -> Path | None:
    """Return Isaac Sim's `python.bat` if present (Windows).

    When running inside Isaac Sim/Isaac Lab on Windows, many prebundled Python packages
    (e.g., OpenCV) live in Isaac's extension `pip_prebundle` folders and are only
    available when the environment is set up via `python.bat` (it calls `setup_python_env.bat`).
    """
    candidate = _repo_root() / "external" / "isaaclab" / "_isaac_sim" / "python.bat"
    if candidate.exists():
        return candidate
    return None


def _gvhmr_python_cmd() -> list[str]:
    """Return a command prefix to run GVHMR demo scripts.

    Priority:
    1) `GVHMR_DEMO_PYTHON` env var (string path to an executable)
    2) On Windows: Isaac Sim `python.bat` (ensures prebundled deps like `cv2` are importable)
    3) Fallback: `sys.executable`
    """
    override = os.environ.get("GVHMR_DEMO_PYTHON")
    if override:
        return [override]

    if os.name == "nt":
        python_bat = _resolve_isaac_sim_python_bat()
        if python_bat:
            # Batch files require `cmd.exe /c`.
            return ["cmd.exe", "/c", str(python_bat)]

    return [sys.executable]


def _env_bool(name: str, default: bool) -> bool:
    """Parse an env var as a boolean.

    Empty/unset => default. Truthy values: 1/true/yes/y/on.
    """
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "y", "on")


def _mp4_has_moov_in_head(path: Path, *, head_bytes: int = 8192) -> bool:
    """Return True if `moov` atom appears near the beginning of the file (faststart)."""
    try:
        with path.open("rb") as f:
            head = f.read(head_bytes)
        return b"moov" in head
    except Exception:
        return False


def _resolve_ffmpeg_cmd() -> list[str] | None:
    """Resolve an ffmpeg command we can execute.

    Prefer a system ffmpeg, but fall back to imageio-ffmpeg (often installed with GVHMR deps).
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return [exe]
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return [exe]
    except Exception:
        pass
    return None


def _faststart_mp4(path: Path) -> bool:
    """Remux an MP4 so the `moov` atom is at the start (better streaming/seek on Safari)."""
    if not path.exists() or path.suffix.lower() != ".mp4":
        return False
    if _mp4_has_moov_in_head(path):
        return False
    ffmpeg = _resolve_ffmpeg_cmd()
    if not ffmpeg:
        return False

    tmp = path.with_suffix(".faststart.mp4")
    try:
        proc = subprocess.run(
            [
                *ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False
        if not tmp.exists() or tmp.stat().st_size <= 0:
            return False
        tmp.replace(path)
        return True
    except Exception:
        return False
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _optimize_preview_mp4_for_webkit(path: Path, *, fps: int = 30, gop_seconds: float = 1.0) -> bool:
    """Re-encode preview MP4s for smooth seeking/switching in Safari/WebKit.

    GVHMR's default encoder settings can produce very long GOPs (e.g. 8+ seconds between keyframes).
    When the user switches camera angles mid-playback we seek into the newly-selected preview; with
    sparse keyframes, Safari can appear to lag while it decodes from a distant keyframe.

    We trade a small file-size increase for predictable keyframes + faststart.
    """
    if not path.exists() or path.suffix.lower() != ".mp4":
        return False
    ffmpeg = _resolve_ffmpeg_cmd()
    if not ffmpeg:
        return False

    gop = max(1, int(round(float(fps) * float(gop_seconds))))
    tmp = path.with_suffix(".webkit.mp4")
    try:
        # Prefer H.264 when available, but some ffmpeg builds (for example imageio-ffmpeg)
        # may not ship with libx264. Fall back to MPEG-4 Part 2 so we still get:
        # - frequent keyframes (short GOP)
        # - faststart moov atom placement
        # which are the main requirements for smooth WebKit playback.
        encoder_variants: list[list[str]] = [
            # Fastest path on NVIDIA GPUs (Windows worker, Pegasus GPU nodes) when ffmpeg is built with NVENC.
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "constqp",
                "-qp",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "main",
                "-bf",
                "0",
            ],
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "main",
                "-level",
                "3.1",
                "-bf",
                "0",
            ],
            [
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                "-pix_fmt",
                "yuv420p",
            ],
        ]

        ok = False
        for enc_args in encoder_variants:
            proc = subprocess.run(
                [
                    *ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(path),
                    "-an",
                    *enc_args,
                    "-g",
                    str(gop),
                    "-keyint_min",
                    str(gop),
                    "-sc_threshold",
                    "0",
                    "-movflags",
                    "+faststart",
                    str(tmp),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                ok = True
                break
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

        if not ok:
            return False
        if not tmp.exists() or tmp.stat().st_size <= 0:
            return False
        tmp.replace(path)
        return True
    except Exception:
        return False
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _can_import_pytorch3d_renderer() -> bool:
    try:
        import pytorch3d.renderer  # noqa: F401

        return True
    except Exception:
        return False


def _required_checkpoint_relpaths(*, require_dpvo: bool) -> list[Path]:
    required = [
        Path("gvhmr") / "gvhmr_siga24_release.ckpt",
        Path("vitpose") / "vitpose-h-multi-coco.pth",
        Path("hmr2") / "epoch=10-step=25000.ckpt",
        Path("yolo") / "yolov8x.pt",
        # GVHMR also requires SMPL-X body model files (licensed; not distributed with this repo).
        Path("body_models") / "smplx" / "SMPLX_NEUTRAL.npz",
    ]
    if require_dpvo:
        required.insert(1, Path("dpvo") / "dpvo.pth")
    return required


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _remove_dangling_path(path: Path) -> None:
    """
    Remove dangling link/junction artifacts only.

    Real existing files/directories are preserved.
    """
    if path.exists() or not _path_lexists(path):
        return
    try:
        path.unlink()
    except OSError:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "rmdir", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
        if _path_lexists(path):
            raise


def _ensure_checkpoints(gvhmr_root: Path, *, require_dpvo: bool) -> None:
    """Ensure `external/gvhmr/inputs/checkpoints` exists by linking staged checkpoints.

    GVHMR expects checkpoints under `<gvhmr_root>/inputs/checkpoints`. We stage large files under:
    `external/humanoid-projects/GVHMR/inputs/checkpoints`.
    """
    expected = gvhmr_root / "inputs" / "checkpoints"
    required_files = _required_checkpoint_relpaths(require_dpvo=require_dpvo)
    if all((expected / rel).exists() for rel in required_files):
        return

    heavy = _resolve_heavy_checkpoints_root()
    if not heavy.exists():
        raise FileNotFoundError(
            "GVHMR heavy checkpoints not found. Expected staged checkpoints under: "
            f"{heavy}. See docs/GVHMR.md."
        )

    expected.parent.mkdir(parents=True, exist_ok=True)
    # When checkpoints are staged outside the repo (for example on Linux node-local `/tmp`),
    # prefer linking `external/gvhmr/inputs/checkpoints` to that staged root instead of copying
    # multi-GB files back into the repo path.
    if os.environ.get("GVHMR_CHECKPOINTS_ROOT"):
        _remove_dangling_path(expected)
        if expected.is_symlink():
            # If a previous run linked to an old/stale location (for example in $HOME),
            # force it to the current staged root to avoid quota issues.
            try:
                if expected.resolve() != heavy.resolve():
                    expected.unlink()
            except OSError:
                expected.unlink()
        elif expected.exists() and expected.is_dir():
            shutil.rmtree(expected, ignore_errors=True)

    # If the expected directory already exists but is incomplete, try to copy missing files from the staged root.
    if expected.exists() and expected.is_dir():
        for rel in required_files:
            src = heavy / rel
            dst = expected / rel
            if dst.exists():
                continue
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # After repo moves/renames, Windows can leave a dangling link/junction behind.
    _remove_dangling_path(expected)

    try:
        if not expected.exists():
            os.symlink(heavy, expected, target_is_directory=True)
    except OSError:
        # Symlinks are often unavailable on Windows without Developer Mode or Administrator privileges.
        # Fall back to copying the required checkpoints. This is slower but robust.
        _remove_dangling_path(expected)
        if not expected.exists():
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

    missing = [str(expected / rel) for rel in required_files if not (expected / rel).exists()]
    if missing:
        smplx_hint = str(heavy / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz")
        raise FileNotFoundError(
            "GVHMR required assets missing:\n"
            + "\n".join(f"- {p}" for p in missing)
            + "\n\n"
            "This includes the SMPL-X model file `SMPLX_NEUTRAL.npz` (licensed; you must download it separately)\n"
            f"and place it under: {smplx_hint}\n\n"
            "Tip: for the platform flow, you can upload this file once via the Web UI (`/studio`) or API:\n"
            "  POST /admin/gvhmr/smplx-model\n"
            "It will be stored under the object-storage key:\n"
            "  gvhmr/body_models/smplx/SMPLX_NEUTRAL.npz\n"
            "and the Windows GPU worker will pull it into the staged checkpoints folder automatically.\n\n"
            "See docs/GVHMR.md for details."
        )


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


def _render_gvhmr_global_preview(
    video_path: Path,
    results_path: Path,
    out_mp4_path: Path,
    *,
    yaw_deg: float = 45.0,
) -> dict[str, Any]:
    """Create a lightweight GVHMR-style preview video without PyTorch3D renderer.

    GVHMR's official demo produces `<video_stem>_3_incam_global_horiz.mp4`:
    left = in-camera mesh overlay, right = global mesh with a ground plane.

    In this platform we run GVHMR with `--skip_render` (PyTorch3D renderer/structures is
    frequently unavailable on Windows). The Web UI already shows the original video in the
    left panel, so this function generates the *right panel only*:
    global-view skeleton with a perspective checkered ground plane.
    """
    import math

    import cv2
    import numpy as np
    import torch

    # Ensure GVHMR python packages (hmr4d/...) and the pytorch3d stub are importable.
    gvhmr_root = _resolve_gvhmr_root()
    if str(gvhmr_root) not in sys.path:
        sys.path.insert(0, str(gvhmr_root))

    from hmr4d.utils.body_model.smplx_lite import SmplxLite, SmplxLiteSmplN24
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor
    from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay

    def _as_tensor(x) -> torch.Tensor:
        if torch.is_tensor(x):
            return x
        return torch.from_numpy(np.asarray(x))

    def _clamp_frame_ids(frame_ids: np.ndarray, max_frame_index: int) -> np.ndarray:
        if max_frame_index <= 0:
            return np.array([0], dtype=np.int32)
        if frame_ids.size == 0:
            return np.array([0], dtype=np.int32)
        max_idx = int(max_frame_index)
        if max_idx == 0:
            return np.zeros_like(frame_ids, dtype=np.int32)
        clipped = np.clip(frame_ids, 0, max_idx)
        return clipped.astype(np.int32)

    def _frame_count_for_param(x: Any) -> int:
        t = _as_tensor(x)
        if t.ndim == 0:
            return 1
        if t.ndim == 1:
            return 1
        return int(t.shape[1]) if t.ndim >= 2 else 1

    def _normalize_to_seq(t: torch.Tensor, *, target_frames: int) -> torch.Tensor:
        if t.ndim == 0:
            # scalar -> expand as constant per frame
            return t.view(1, 1, 1).repeat(1, target_frames, 1)
        if t.ndim == 1:
            # per-param vector -> repeat for every frame
            return t.view(1, 1, -1).repeat(1, target_frames, 1)
        if t.ndim == 2:
            # sequence -> make shape (1, F, ...)
            t = t[None, ...]
        if t.ndim == 3:
            if t.shape[1] == target_frames:
                return t
            if t.shape[1] == 1:
                return t.repeat(1, target_frames, 1)
            if t.shape[1] > target_frames:
                return t[:, :target_frames]
            raise ValueError(f"cannot align frame count {t.shape[1]} to {target_frames}")
        raise ValueError(f"Unsupported tensor rank for alignment: {t.ndim}")

    pred = torch.load(results_path, map_location="cpu")
    smpl_global = pred.get("smpl_params_global") or {}
    if not smpl_global:
        raise ValueError("hmr4d_results.pt missing smpl_params_global")

    # Determine the frame count from global body_pose.
    body_pose_global = _as_tensor(smpl_global.get("body_pose")).float()
    if body_pose_global.ndim == 2:
        total_frames = int(body_pose_global.shape[0])
    else:
        total_frames = int(body_pose_global.shape[1])
    if total_frames <= 0:
        raise ValueError("smpl_params_global has no frames")

    # Cap preview cost by decimating long videos.
    max_preview_frames = 450
    stride = max(1, int(math.ceil(total_frames / max_preview_frames)))
    frame_ids = np.arange(0, total_frames, stride, dtype=np.int32)

    def _extract_params(
        params: dict[str, Any], frame_ids: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pose = _as_tensor(params.get("body_pose")).float()
        global_orient = _as_tensor(params.get("global_orient")).float()
        betas = _as_tensor(params.get("betas")).float()
        transl_raw = params.get("transl", None)
        transl = _as_tensor(transl_raw).float() if transl_raw is not None else None

        # Normalize shapes to (B, F, ...) so SmplxLite behaves consistently across outputs.
        if body_pose.ndim == 2:
            body_pose = body_pose[None, ...]
        if global_orient.ndim == 2:
            global_orient = global_orient[None, ...]
        if transl is not None and transl.ndim == 2:
            transl = transl[None, ...]

        if betas.ndim == 1:
            betas = betas[None, ...]
        elif betas.ndim == 2:
            # GVHMR commonly stores betas per-frame as (F, 10). Wrap to (1, F, 10).
            if betas.shape[0] != body_pose.shape[0]:
                betas = betas[None, ...]

        B = int(body_pose.shape[0])
        F = int(body_pose.shape[1])
        # Body pose and camera orientation are expected as (B, F, ...).
        if transl is None:
            transl = torch.zeros((B, F, 3), dtype=torch.float32)

        target_frames = min(
            _frame_count_for_param(body_pose),
            _frame_count_for_param(global_orient),
            _frame_count_for_param(transl),
            _frame_count_for_param(betas),
        )
        if target_frames <= 0:
            raise ValueError("smpl params contain no frames")

        frame_ids = _clamp_frame_ids(frame_ids, target_frames - 1)
        if target_frames <= 1:
            frame_ids[:] = 0

        body_pose = _normalize_to_seq(body_pose, target_frames=target_frames)
        global_orient = _normalize_to_seq(global_orient, target_frames=target_frames)
        transl = _normalize_to_seq(transl, target_frames=target_frames)
        betas_out = _normalize_to_seq(betas, target_frames=target_frames)

        return (
            body_pose[:, frame_ids],
            betas_out,
            global_orient[:, frame_ids],
            transl[:, frame_ids],
        )

    # Prefer GVHMR's internal 30fps re-encoded video if present; it matches `hmr4d_results.pt` length exactly.
    gvhmr_video = results_path.parent / "0_input_video.mp4"
    video_for_preview = gvhmr_video if gvhmr_video.exists() else video_path

    cap = cv2.VideoCapture(str(video_for_preview))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video with OpenCV: {video_for_preview}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps_out = max(1.0, fps_in / float(stride))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w_in <= 0 or h_in <= 0:
        w_in, h_in = 640, 360
    cap.release()

    panel_w = int(w_in)
    panel_h = int(h_in)
    out_w = int(panel_w)
    out_h = int(panel_h)

    # Global panel intrinsics (match GVHMR demo: render as a 24mm lens).
    _, _, K_global = create_camera_sensor(panel_w, panel_h, 24)
    K_global = _as_tensor(K_global).float()

    coco_edges = [
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ]

    def _project_pinhole(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        # points: (..., 3), K: (3, 3) -> (..., 2)
        z = points[..., 2:3].clone()
        z.masked_fill_(z.abs() < 1e-6, 1e-6)
        norm = points / z
        uvw = torch.matmul(norm, K.T)
        return uvw[..., :2]

    def _draw_skeleton(
        img: np.ndarray,
        uv: np.ndarray,
        *,
        color: tuple[int, int, int],
        thickness: int,
        radius: int,
    ) -> None:
        if uv.ndim != 2 or uv.shape[0] == 0:
            return
        for a, b in coco_edges:
            if a >= uv.shape[0] or b >= uv.shape[0]:
                continue
            ax, ay = int(uv[a, 0]), int(uv[a, 1])
            bx, by = int(uv[b, 0]), int(uv[b, 1])
            cv2.line(img, (ax, ay), (bx, by), color, thickness, lineType=cv2.LINE_AA)
        for k in range(uv.shape[0]):
            xk, yk = int(uv[k, 0]), int(uv[k, 1])
            cv2.circle(img, (xk, yk), radius, color, -1, lineType=cv2.LINE_AA)

    # === Compute mesh + joints for sampled frames ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_smpl24 = SmplxLiteSmplN24().to(device)
    model_mesh = SmplxLite().to(device)

    bp_g, be_g, go_g, tr_g = _extract_params(smpl_global, frame_ids)

    with torch.no_grad():
        joints_global_smpl24 = model_smpl24(bp_g.to(device), be_g.to(device), go_g.to(device), tr_g.to(device)).squeeze(0)
        verts_global = model_mesh(bp_g.to(device), be_g.to(device), go_g.to(device), tr_g.to(device)).squeeze(0)

    debug: dict[str, Any] = {}

    # === Normalize + face-Z transform for global view (approx GVHMR `move_to_start_point_face_z`) ===
    # We estimate a stable "floor" height from joints. We intentionally prefer a low percentile so
    # we do not push the body down into the ground. Then, after the transform, we clamp away any
    # remaining floating (never shift up).
    min_y_per_frame = joints_global_smpl24[..., 1].amin(dim=1).detach().cpu().numpy()
    floor_y_p10 = float(np.percentile(min_y_per_frame, 10))
    floor_y_p50 = float(np.percentile(min_y_per_frame, 50))
    floor_y = float(floor_y_p10)
    debug["floor_y_joints_p10_before_T"] = float(floor_y_p10)
    debug["floor_y_joints_p50_before_T"] = float(floor_y_p50)
    debug["floor_y_before_T"] = float(floor_y)

    offset = joints_global_smpl24[0, 0].detach().clone()
    offset[1] = floor_y
    joints_global_smpl24 = joints_global_smpl24 - offset[None, None, :]
    verts_global = verts_global - offset[None, None, :]

    T_ay2ayfz = compute_T_ayfz2ay(joints_global_smpl24[[0]], inverse=True)  # (1, 4, 4)
    T_seq = T_ay2ayfz.repeat(joints_global_smpl24.shape[0], 1, 1)  # (F, 4, 4)
    joints_global_smpl24 = apply_T_on_points(joints_global_smpl24, T_seq)
    verts_global = apply_T_on_points(verts_global, T_seq)

    # Re-level after the global transform by shifting down only if we are floating.
    min_y_per_frame2 = joints_global_smpl24[..., 1].amin(dim=1).detach().cpu().numpy()
    floor_y2_p10 = float(np.percentile(min_y_per_frame2, 10))
    floor_y2_p50 = float(np.percentile(min_y_per_frame2, 50))
    debug["floor_y_joints_p10_after_T"] = float(floor_y2_p10)
    debug["floor_y_joints_p50_after_T"] = float(floor_y2_p50)
    if floor_y2_p10 > 0:
        joints_global_smpl24[..., 1] = joints_global_smpl24[..., 1] - float(floor_y2_p10)
        verts_global[..., 1] = verts_global[..., 1] - float(floor_y2_p10)
        debug["post_T_shift_down"] = float(floor_y2_p10)
    else:
        debug["post_T_shift_down"] = 0.0

    # Final vertical range diagnostics (helps debug "floating above ground" issues).
    debug["joints_y_min_final"] = float(joints_global_smpl24[..., 1].min().item())
    debug["joints_y_max_final"] = float(joints_global_smpl24[..., 1].max().item())
    debug["verts_y_min_final"] = float(verts_global[..., 1].min().item())
    debug["verts_y_max_final"] = float(verts_global[..., 1].max().item())

    # If the global mesh is entirely behind the camera (z <= 0), projection collapses and the preview
    # looks empty or "floating". Shift the whole scene forward so the body is in front of the camera.
    min_z = float(verts_global[..., 2].min().item())
    debug["verts_z_min_final"] = float(min_z)
    if min_z < 0.25:
        z_shift = float(0.25 - min_z)
        verts_global[..., 2] = verts_global[..., 2] + z_shift
        joints_global_smpl24[..., 2] = joints_global_smpl24[..., 2] + z_shift
        debug["scene_z_shift"] = float(z_shift)
    else:
        debug["scene_z_shift"] = 0.0

    # === Static global camera (match GVHMR `get_global_cameras_static`) ===
    targets = verts_global.mean(dim=1).detach().cpu()  # (F, 3)
    targets[:, 1] = 0.0
    target_center = targets.mean(dim=0)  # (3,)
    target_scale = torch.norm(targets - target_center[None, :], dim=-1).max()
    cam_dist = float(max(target_scale.item(), 1.0)) * 2.0  # beta=2.0 (GVHMR default)

    vec_rad = float(math.radians(float(yaw_deg)))
    vec = torch.tensor([math.sin(vec_rad), 0.0, math.cos(vec_rad)], dtype=torch.float32)
    vec = vec / torch.norm(vec)
    target_center_height = 1.0

    def _look_at_world_to_cam(pos: torch.Tensor, at: torch.Tensor, up_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (R, t) such that p_cam = p_world @ R.T + t
        z = at - pos
        z = z / torch.norm(z)
        x = torch.cross(up_vec, z)
        x = x / torch.norm(x)
        # "y-down" camera frame so pinhole projection matches image coordinates.
        y = -torch.cross(z, x)
        R = torch.stack([x, y, z], dim=0)  # rows, world->cam
        t = -(R @ pos)
        return R, t

    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    cam_pos = target_center + vec * float(cam_dist)
    cam_pos[1] = float(cam_dist) * float(math.tan(math.radians(20.0))) + float(target_center_height)
    cam_target = target_center.clone()
    cam_target[1] = float(target_center_height)
    R_wc, t_wc = _look_at_world_to_cam(cam_pos, cam_target, up)
    debug["cam_dist"] = float(cam_dist)
    debug["cam_pos"] = [float(x) for x in cam_pos.detach().cpu().tolist()]
    debug["cam_target"] = [float(x) for x in cam_target.detach().cpu().tolist()]
    debug["target_center"] = [float(x) for x in target_center.detach().cpu().tolist()]
    debug["target_center_height"] = float(target_center_height)

    def _project_global(points_world: torch.Tensor) -> torch.Tensor:
        pw = points_world.detach().cpu()
        pc = torch.matmul(pw, R_wc.T) + t_wc
        return _project_pinhole(pc, K_global)

    # === Ground grid parameters (fixed across frames) ===
    root_points = joints_global_smpl24[:, 0, :].detach().cpu().numpy()
    cx = float((root_points[:, 0].max() + root_points[:, 0].min()) * 0.5)
    cz = float((root_points[:, 2].max() + root_points[:, 2].min()) * 0.5)
    all_pts = verts_global.detach().cpu().numpy().reshape(-1, 3)
    scale_x = float(all_pts[:, 0].max() - all_pts[:, 0].min())
    scale_z = float(all_pts[:, 2].max() - all_pts[:, 2].min())
    grid_scale = max(scale_x, scale_z, 1.0) * 1.5
    grid_half = 0.5 * grid_scale
    grid_steps = 10
    grid_step = grid_scale / float(grid_steps)

    # Precompute the checkerboard quads once (static camera + static ground plane).
    checker_quads: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    checker_a = (200, 205, 208)
    checker_b = (160, 165, 168)
    for ix in range(grid_steps):
        for iz in range(grid_steps):
            x0 = cx - grid_half + float(ix) * grid_step
            x1 = x0 + grid_step
            z0 = cz - grid_half + float(iz) * grid_step
            z1 = z0 + grid_step
            corners = torch.tensor(
                [[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]],
                dtype=torch.float32,
            )
            uv = _project_global(corners)
            uv_np = uv.detach().cpu().numpy()
            if not np.all(np.isfinite(uv_np)):
                continue
            quad = uv_np.astype(np.int32)
            color = checker_a if (ix + iz) % 2 == 0 else checker_b
            checker_quads.append((quad, color))

    verts_global_np = verts_global.detach().cpu().numpy()  # (F', V, 3)
    K_global_np = K_global.detach().cpu().numpy()  # (3, 3)
    R_wc_np = R_wc.detach().cpu().numpy()  # (3, 3)
    t_wc_np = t_wc.detach().cpu().numpy()  # (3,)

    def _render_body_mesh(img: np.ndarray, verts_world: np.ndarray) -> bool:
        # Render as a depth-shaded dense point cloud (fast + no extra deps).
        verts_cam = verts_world @ R_wc_np.T + t_wc_np  # (V, 3)
        z = verts_cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return False

        x = verts_cam[valid, 0]
        y = verts_cam[valid, 1]
        z = z[valid]
        u = (x / z) * float(K_global_np[0, 0]) + float(K_global_np[0, 2])
        v = (y / z) * float(K_global_np[1, 1]) + float(K_global_np[1, 2])
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        in_img = (ui >= 0) & (ui < panel_w) & (vi >= 0) & (vi < panel_h)
        if not np.any(in_img):
            return False

        ui = ui[in_img]
        vi = vi[in_img]
        z = z[in_img].astype(np.float32)

        idx = (vi.astype(np.int64) * int(panel_w) + ui.astype(np.int64)).astype(np.int64)
        depth_flat = np.full(int(panel_h) * int(panel_w), np.inf, dtype=np.float32)
        np.minimum.at(depth_flat, idx, z)
        depth = depth_flat.reshape((int(panel_h), int(panel_w)))

        valid_depth = np.isfinite(depth)
        if not np.any(valid_depth):
            return False

        # Close small holes so the body looks like a continuous surface.
        mask_u8 = (valid_depth.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = mask_u8 > 0
        if not np.any(mask):
            return False

        d_valid = depth[valid_depth]
        dmin = float(np.percentile(d_valid, 5))
        dmax = float(np.percentile(d_valid, 95))
        if not (dmax > dmin):
            dmin = float(d_valid.min())
            dmax = float(d_valid.max() + 1e-3)

        depth_clamped = depth.copy()
        depth_clamped[~valid_depth] = dmax
        depth_clamped = np.clip(depth_clamped, dmin, dmax)
        depth_norm = (depth_clamped - dmin) / (dmax - dmin + 1e-6)
        depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
        depth_u8[~valid_depth] = 0

        # Fill missing depth where the vertex cloud was sparse.
        inpaint_mask = (~valid_depth).astype(np.uint8) * 255
        depth_u8_filled = cv2.inpaint(depth_u8, inpaint_mask, 3, cv2.INPAINT_TELEA)

        depth_f = depth_u8_filled.astype(np.float32) / 255.0
        intensity = 0.88 - 0.38 * depth_f  # nearer -> brighter
        body = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)

        img[mask] = np.stack([body, body, body], axis=-1)[mask]

        # Subtle outline to make limbs read better.
        edges = cv2.Canny(mask_u8, 60, 160)
        img[edges > 0] = (120, 120, 120)
        return True

    # === Render global preview ===
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create VideoWriter for: {out_mp4_path}")

    try:
        for jidx in range(int(frame_ids.shape[0])):
            img = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
            for quad, color in checker_quads:
                cv2.fillConvexPoly(img, quad, color, lineType=cv2.LINE_AA)

            drew = _render_body_mesh(img, verts_global_np[jidx])
            if not drew:
                # Fallback to skeleton so users never see an empty "room" preview.
                uv = _project_global(joints_global_smpl24[jidx])
                uv_np = uv.detach().cpu().numpy()
                if np.all(np.isfinite(uv_np)):
                    _draw_skeleton(img, uv_np, color=(90, 90, 90), thickness=3, radius=4)

            writer.write(img)
    finally:
        writer.release()

    return {
        "ok": True,
        "preview_mp4": str(out_mp4_path),
        "fps_in": fps_in,
        "fps_out": fps_out,
        "stride": int(stride),
        "frames_out": int(frame_ids.shape[0]),
        "layout": "global_meshlike",
        "yaw_deg": float(yaw_deg),
        "debug": debug,
    }


def _render_gvhmr_incam_overlay(
    video_path: Path,
    results_path: Path,
    out_mp4_path: Path,
) -> dict[str, Any]:
    """Render an in-camera overlay preview without PyTorch3D.

    This approximates GVHMR's native `1_incam.mp4` output by projecting SMPL-X vertices
    using `smpl_params_incam` and `K_fullimg` from `hmr4d_results.pt`.
    """
    import math

    import cv2
    import numpy as np
    import torch

    pred = torch.load(results_path, map_location="cpu")
    smpl_incam = pred.get("smpl_params_incam") or {}
    K_fullimg = pred.get("K_fullimg", None)
    if not smpl_incam or K_fullimg is None:
        raise ValueError("hmr4d_results.pt missing smpl_params_incam/K_fullimg")

    # Prefer GVHMR's internal 30fps re-encoded video; it matches result length exactly.
    gvhmr_video = results_path.parent / "0_input_video.mp4"
    video_for_preview = gvhmr_video if gvhmr_video.exists() else video_path

    cap = cv2.VideoCapture(str(video_for_preview))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video with OpenCV: {video_for_preview}")
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w_in <= 0 or h_in <= 0:
        w_in, h_in = 640, 360

    K_arr = _as_tensor(K_fullimg).float()
    if K_arr.ndim != 3 or K_arr.shape[-1] != 3 or K_arr.shape[-2] != 3 or K_arr.shape[0] <= 0:
        raise ValueError("smpl4d_results.pt has invalid K_fullimg shape")

    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if cap_frames <= 0:
        cap.release()
        raise ValueError("Input preview video has no frames")

    body_pose = _as_tensor(smpl_incam.get("body_pose")).float()
    global_orient = _as_tensor(smpl_incam.get("global_orient")).float()
    transl_raw = smpl_incam.get("transl", None)
    transl = (
        _as_tensor(transl_raw).float()
        if transl_raw is not None
        else torch.zeros((int(body_pose.shape[-2]), 3), dtype=torch.float32)
    )
    betas = _as_tensor(smpl_incam.get("betas")).float()

    if body_pose.ndim == 2:
        body_pose = body_pose[None, ...]
    if global_orient.ndim == 2:
        global_orient = global_orient[None, ...]
    if transl.ndim == 2:
        transl = transl[None, ...]
    if betas.ndim == 1:
        betas = betas[None, ...]

    def _frame_count_for(x: torch.Tensor) -> int:
        if x.ndim == 0:
            return 1
        if x.ndim == 1:
            return 1
        return int(x.shape[1])

    def _align_for_render(x: torch.Tensor, target_frames: int) -> torch.Tensor:
        if x.ndim == 0:
            return x.view(1, 1, 1).repeat(1, target_frames, 1)
        if x.ndim == 1:
            return x.view(1, 1, -1).repeat(1, target_frames, 1)
        if x.ndim == 2:
            x = x[None, ...]
        if x.ndim == 3:
            if x.shape[1] == target_frames:
                return x
            if x.shape[1] == 1:
                return x.repeat(1, target_frames, 1)
            if x.shape[1] > target_frames:
                return x[:, :target_frames]
            raise ValueError(f"cannot align frame count {x.shape[1]} to {target_frames}")
        raise ValueError(f"Unsupported tensor rank for incam render: {x.ndim}")

    frame_count = min(
        int(cap_frames),
        int(K_arr.shape[0]),
        _frame_count_for(body_pose),
        _frame_count_for(global_orient),
        _frame_count_for(transl),
        _frame_count_for(betas),
    )
    if frame_count <= 0:
        cap.release()
        raise ValueError("smpl params have no valid frame dimension")

    total_frames = frame_count
    if total_frames <= 0:
        total_frames = int(K_arr.shape[0])

    max_preview_frames = 450
    stride = max(1, int(math.ceil(total_frames / max_preview_frames)))
    frame_ids = np.arange(0, total_frames, stride, dtype=np.int32)
    fps_out = max(1.0, fps_in / float(stride))
    if frame_ids.size <= 0:
        cap.release()
        raise ValueError("No preview frame ids generated")

    # Compute mesh vertices in camera space.
    gvhmr_root = _resolve_gvhmr_root()
    if str(gvhmr_root) not in sys.path:
        sys.path.insert(0, str(gvhmr_root))
    from hmr4d.utils.body_model.smplx_lite import SmplxLite

    if betas.ndim == 2:
        betas = betas[None, ...]

    max_frame_index = frame_ids.max(initial=0)
    frame_count_in_params = min(
        _frame_count_for(body_pose),
        _frame_count_for(global_orient),
        _frame_count_for(transl),
        _frame_count_for(betas),
    )
    frame_ids = np.clip(frame_ids, 0, max(0, frame_count_in_params - 1)).astype(np.int32)
    if frame_count_in_params <= 1:
        frame_ids[:] = 0

    body_pose = _align_for_render(body_pose, target_frames=frame_count_in_params)
    global_orient = _align_for_render(global_orient, target_frames=frame_count_in_params)
    transl = _align_for_render(transl, target_frames=frame_count_in_params)
    betas = _align_for_render(betas, target_frames=frame_count_in_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mesh = SmplxLite().to(device)

    betas_in = (
        betas[:, frame_ids] if (betas.ndim == 3 and betas.shape[1] >= int(frame_ids.max(initial=0) + 1)) else betas
    )

    with torch.no_grad():
        verts_cam = model_mesh(
            body_pose[:, frame_ids].to(device),
            betas_in.to(device),
            global_orient[:, frame_ids].to(device),
            transl[:, frame_ids].to(device),
        ).squeeze(0)
    verts_cam_np = verts_cam.detach().cpu().numpy()  # (F', V, 3)
    K_np = _as_tensor(K_fullimg[frame_ids]).detach().cpu().numpy()  # (F', 3, 3)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps_out, (int(w_in), int(h_in)))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create VideoWriter for: {out_mp4_path}")

    alpha = 0.65
    kernel = np.ones((5, 5), np.uint8)
    try:
        # Read sequentially and only render selected frames (avoid random seeks).
        want = set(int(x) for x in frame_ids.tolist())
        next_out = 0
        frame_idx = 0
        while next_out < int(frame_ids.shape[0]):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx not in want:
                frame_idx += 1
                continue

            verts = verts_cam_np[next_out]
            K = K_np[next_out]
            next_out += 1
            frame_idx += 1

            # Project dense vertex cloud to a depth map in image space.
            z = verts[:, 2].astype(np.float32)
            valid = z > 1e-6
            if not np.any(valid):
                writer.write(frame)
                continue
            x = verts[valid, 0].astype(np.float32)
            y = verts[valid, 1].astype(np.float32)
            z = z[valid]
            u = (x / z) * float(K[0, 0]) + float(K[0, 2])
            v = (y / z) * float(K[1, 1]) + float(K[1, 2])
            ui = np.rint(u).astype(np.int32)
            vi = np.rint(v).astype(np.int32)
            in_img = (ui >= 0) & (ui < int(w_in)) & (vi >= 0) & (vi < int(h_in))
            if not np.any(in_img):
                writer.write(frame)
                continue
            ui = ui[in_img]
            vi = vi[in_img]
            z = z[in_img].astype(np.float32)

            idx = (vi.astype(np.int64) * int(w_in) + ui.astype(np.int64)).astype(np.int64)
            depth_flat = np.full(int(h_in) * int(w_in), np.inf, dtype=np.float32)
            np.minimum.at(depth_flat, idx, z)
            depth = depth_flat.reshape((int(h_in), int(w_in)))

            valid_depth = np.isfinite(depth)
            if not np.any(valid_depth):
                writer.write(frame)
                continue

            mask_u8 = (valid_depth.astype(np.uint8) * 255)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = mask_u8 > 0
            if not np.any(mask):
                writer.write(frame)
                continue

            d_valid = depth[valid_depth]
            dmin = float(np.percentile(d_valid, 5))
            dmax = float(np.percentile(d_valid, 95))
            if not (dmax > dmin):
                dmin = float(d_valid.min())
                dmax = float(d_valid.max() + 1e-3)

            depth_clamped = depth.copy()
            depth_clamped[~valid_depth] = dmax
            depth_clamped = np.clip(depth_clamped, dmin, dmax)
            depth_norm = (depth_clamped - dmin) / (dmax - dmin + 1e-6)
            intensity = 0.92 - 0.42 * depth_norm  # nearer -> brighter
            body = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)
            overlay = np.stack([body, body, body], axis=-1)

            out = frame.copy()
            out[mask] = (out[mask].astype(np.float32) * (1.0 - alpha) + overlay[mask].astype(np.float32) * alpha).astype(
                np.uint8
            )

            edges = cv2.Canny(mask_u8, 60, 160)
            out[edges > 0] = (255, 255, 255)

            writer.write(out)
    finally:
        cap.release()
        writer.release()

    return {
        "ok": True,
        "preview_mp4": str(out_mp4_path),
        "fps_in": fps_in,
        "fps_out": fps_out,
        "stride": int(stride),
        "frames_out": int(frame_ids.shape[0]),
        "layout": "incam_overlay",
    }


def run_gvhmr(
    video_path: Path,
    output_dir: Path,
    *,
    static_cam: bool,
    use_dpvo: bool,
    f_mm: int | None,
    skip_render: bool = False,
) -> dict[str, Any]:
    gvhmr_root = _resolve_gvhmr_root()
    demo_script = gvhmr_root / "tools" / "demo" / "demo.py"
    if (not gvhmr_root.exists()) or (not _looks_like_gvhmr_repo(gvhmr_root)):
        checked = "\n".join(f"- {p}" for p in _candidate_gvhmr_roots())
        raise FileNotFoundError(
            "GVHMR repo not found. Clone into external/gvhmr, external/GVHMR, "
            "external/humanoid-projects/GVHMR or set GVHMR_ROOT.\n"
            f"Expected: {gvhmr_root}\n"
            "Checked:\n"
            f"{checked}"
        )
    _ensure_checkpoints(gvhmr_root, require_dpvo=bool(use_dpvo))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir / "gvhmr"
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = _gvhmr_python_cmd() + [
        str(gvhmr_root / "tools" / "demo" / "demo.py"),
        "--video",
        str(video_path),
        "--output_root",
        str(output_root),
    ]
    # Native mesh rendering:
    # - If `GVHMR_NATIVE_RENDER` is explicitly set, honor it.
    # - Otherwise, auto-enable it when `pytorch3d.renderer` is importable.
    # - If skip_render is requested, skip render setup entirely.
    enable_native_render = False if skip_render else _env_bool("GVHMR_NATIVE_RENDER", default=_can_import_pytorch3d_renderer())
    if skip_render:
        cmd.append("--skip_render")
    elif not enable_native_render:
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

    if enable_native_render:
        # Default to CUDA rendering (fast). GVHMR will fall back to CPU if rasterization fails.
        env.setdefault("GVHMR_RENDER_DEVICE", "cuda")
        # Render the in-camera overlay video so users can view the reconstruction from the
        # original camera view (helps debugging + validation).
        env.setdefault("GVHMR_RENDER_INCAM", "1")
        # Render extra global views (front/side) so the Studio UI can offer a view toggle.
        env.setdefault("GVHMR_RENDER_EXTRA_VIEWS", "1")
        # PyTorch3D's coarse rasterizer can overflow its per-bin face budget, which shows up as
        # missing/black fragments in the rendered mesh. Increase the budget, but keep coarse
        # rasterization enabled for speed.
        #
        # If you see "Bin size was too small..." warnings or black flicker frames again, set:
        #   GVHMR_P3D_BIN_SIZE=0
        # to force naive rasterization (slower, but very robust).
        env.setdefault("GVHMR_P3D_BIN_SIZE", "64")
        env.setdefault("GVHMR_P3D_MAX_FACES_PER_BIN", "50000")

    # GVHMR's demo script uses ffmpeg-python which shells out to a binary named `ffmpeg`.
    # Some environments (notably Pegasus compute nodes) do not have a system ffmpeg on PATH.
    # We vendor a tiny shim directory with an `ffmpeg` executable (symlink or copy) so the demo
    # can merge videos without requiring system-level packages.
    ffmpeg_cmd = _resolve_ffmpeg_cmd()
    if ffmpeg_cmd and not shutil.which("ffmpeg"):
        shim_dir = output_dir / "_ffmpeg_shim"
        try:
            shim_dir.mkdir(parents=True, exist_ok=True)
            shim_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            shim_path = shim_dir / shim_name
            if not shim_path.exists():
                src = Path(ffmpeg_cmd[0])
                try:
                    os.symlink(str(src), str(shim_path))
                except Exception:
                    try:
                        shutil.copy2(src, shim_path)
                        shim_path.chmod(0o755)
                    except Exception:
                        pass
            env["PATH"] = str(shim_dir) + os.pathsep + env.get("PATH", "")
        except Exception:
            pass

    # The Windows bootstrap installs a minimal `external/gvhmr/pytorch3d` stub so GVHMR can
    # run inference without the full PyTorch3D renderer. When native rendering is enabled
    # (GVHMR_NATIVE_RENDER=1), the stub would shadow the real `pytorch3d` package in
    # site-packages. Temporarily move it aside so GVHMR can import `pytorch3d.renderer`.
    stub_dir = gvhmr_root / "pytorch3d"
    stub_backup = gvhmr_root / "pytorch3d_stub_han"
    moved_stub = False
    # If a prior run crashed mid-rename, restore the stub so future `--skip_render` runs still work.
    if stub_backup.exists() and (not stub_dir.exists()) and stub_backup.is_dir():
        try:
            stub_backup.rename(stub_dir)
        except Exception:
            pass
    if not skip_render and enable_native_render and stub_dir.exists() and stub_dir.is_dir():
        try:
            if stub_backup.exists():
                shutil.rmtree(stub_backup)
            stub_dir.rename(stub_backup)
            moved_stub = True
        except Exception:
            moved_stub = False

    # GVHMR can be very verbose; keep stdout clean so the parent process can parse our JSON result.
    gvhmr_log = output_dir / "gvhmr.log"
    try:
        with gvhmr_log.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, check=False, cwd=str(gvhmr_root), env=env, stdout=handle, stderr=handle)
    finally:
        if moved_stub:
            try:
                if stub_dir.exists():
                    shutil.rmtree(stub_dir)
                stub_backup.rename(stub_dir)
            except Exception:
                # Best-effort restore; if this fails the next run may require re-running bootstrap_gvhmr.ps1.
                pass

    meta_path = output_dir / f"{video_path.stem}_gvhmr_meta.json"
    if proc.returncode != 0:
        meta = {
            "ok": False,
            "video": str(video_path),
            "output_root": str(output_root),
            "gvhmr_log": str(gvhmr_log),
            "returncode": int(proc.returncode),
            "python_cmd": cmd[:3] if cmd[:2] == ["cmd.exe", "/c"] else cmd[:1],
            "static_cam": bool(static_cam),
            "use_dpvo": bool(use_dpvo),
            "f_mm": int(f_mm) if f_mm is not None else None,
            "error": f"GVHMR demo failed (exit={proc.returncode}). See gvhmr.log: {gvhmr_log}",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "ok": False,
            "error": meta["error"],
            "returncode": int(proc.returncode),
            "meta_path": str(meta_path),
            "gvhmr_log_path": str(gvhmr_log),
        }

    results_path = _find_hmr4d_results(output_root, video_path.stem)
    payload = _load_smpl_params(results_path)

    import numpy as np

    npz_path = output_dir / f"{video_path.stem}_gvhmr_smplx.npz"
    np.savez_compressed(npz_path, **payload["smpl_params_global"])

    rendered_input_norm = results_path.parent / "0_input_video.mp4"
    rendered_incam = results_path.parent / "1_incam.mp4"
    rendered_global = results_path.parent / "2_global.mp4"
    rendered_global_front = results_path.parent / "2_global_front.mp4"
    rendered_global_side = results_path.parent / "2_global_side.mp4"

    # If render is disabled, skip fallback rendering and return only pose outputs.
    render_errors: dict[str, str] | None = None
    fallback_render_debug: dict[str, Any] = {}

    if not skip_render and (not enable_native_render):
        render_errors = {}
        if not rendered_global.exists():
            try:
                out = _render_gvhmr_global_preview(video_path, results_path, rendered_global, yaw_deg=45.0)
                fallback_render_debug["global45"] = out.get("debug", None)
            except Exception as exc:  # noqa: BLE001
                render_errors["global45"] = f"{type(exc).__name__}: {exc}"

        if not rendered_global_front.exists():
            try:
                out = _render_gvhmr_global_preview(video_path, results_path, rendered_global_front, yaw_deg=0.0)
                fallback_render_debug["global_front"] = out.get("debug", None)
            except Exception as exc:  # noqa: BLE001
                render_errors["global_front"] = f"{type(exc).__name__}: {exc}"

        if not rendered_global_side.exists():
            try:
                out = _render_gvhmr_global_preview(video_path, results_path, rendered_global_side, yaw_deg=90.0)
                fallback_render_debug["global_side"] = out.get("debug", None)
            except Exception as exc:  # noqa: BLE001
                render_errors["global_side"] = f"{type(exc).__name__}: {exc}"

        if not rendered_incam.exists():
            try:
                out = _render_gvhmr_incam_overlay(video_path, results_path, rendered_incam)
                fallback_render_debug["incam_overlay"] = {
                    "ok": bool(out.get("ok", True)),
                    "stride": out.get("stride", None),
                    "frames_out": out.get("frames_out", None),
                    "fps_out": out.get("fps_out", None),
                }
            except Exception as exc:  # noqa: BLE001
                render_errors["incam"] = f"{type(exc).__name__}: {exc}"

    preview_path = rendered_global if rendered_global.exists() else (output_dir / f"{video_path.stem}_gvhmr_preview.mp4")
    preview = None
    preview_error = None
    if preview_path.exists():
        preview = {
            "ok": True,
            "preview_mp4": str(preview_path),
            "source": "gvhmr_native_render" if enable_native_render else "han_fallback_render",
        }
    else:
        # Last-resort fallback to ensure the UI never shows an empty preview panel.
        try:
            out = _render_gvhmr_global_preview(video_path, results_path, preview_path, yaw_deg=45.0)
            if "global45" not in fallback_render_debug:
                fallback_render_debug["global45"] = out.get("debug", None)
            if preview_path.exists():
                preview = {
                    "ok": True,
                    "preview_mp4": str(preview_path),
                    "source": "han_fallback_render",
                }
        except Exception as exc:  # noqa: BLE001
            preview = None
            preview_error = f"{type(exc).__name__}: {exc}"

    optimized: dict[str, bool] = {}
    faststart: dict[str, bool] = {}
    if not skip_render:
        # Safari/WebKit can look laggy if MP4s have sparse keyframes or aren't "faststart".
        # Keep the input-normalized video as a cheap remux, but re-encode previews for responsive seeks.
        faststart = {
            "input_norm": _faststart_mp4(rendered_input_norm) if rendered_input_norm.exists() else False,
        }
        candidates = [
            ("incam", rendered_incam),
            ("global45", rendered_global),
            ("global_front", rendered_global_front),
            ("global_side", rendered_global_side),
            ("preview", preview_path),
        ]

        # Avoid re-encoding the same MP4 twice (e.g. when `preview_path == rendered_global`).
        opt_by_path: dict[str, bool] = {}
        faststart_by_path: dict[str, bool] = {}
        for _label, path in candidates:
            if not path.exists():
                continue
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key in opt_by_path:
                continue
            did_opt = _optimize_preview_mp4_for_webkit(path)
            opt_by_path[key] = did_opt
            # Fallback: even if re-encode fails, at least faststart remux for smoother buffering.
            if not did_opt:
                faststart_by_path[key] = _faststart_mp4(path)

        for label, path in candidates:
            if not path.exists():
                optimized[label] = False
                continue
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            optimized[label] = bool(opt_by_path.get(key, False))
            if key in faststart_by_path:
                faststart[label] = faststart_by_path[key]

    meta = {
        "ok": True,
        "video": str(video_path),
        "results_path": str(results_path),
        "output_npz": str(npz_path),
        "gvhmr_log": str(gvhmr_log),
        "preview_mp4": str(preview_path) if preview_path.exists() else None,
        "preview_error": preview_error,
        "fallback_render_errors": render_errors,
        "fallback_render_debug": fallback_render_debug if fallback_render_debug else None,
        "native_render": bool(enable_native_render),
        "native_render_global_mp4": str(rendered_global) if rendered_global.exists() else None,
        "native_render_incam_mp4": str(rendered_incam) if rendered_incam.exists() else None,
        "native_render_global_front_mp4": str(rendered_global_front) if rendered_global_front.exists() else None,
        "native_render_global_side_mp4": str(rendered_global_side) if rendered_global_side.exists() else None,
        "native_render_input_norm_mp4": str(rendered_input_norm) if rendered_input_norm.exists() else None,
        "python_cmd": cmd[:3] if cmd[:2] == ["cmd.exe", "/c"] else cmd[:1],
        "static_cam": bool(static_cam),
        "use_dpvo": bool(use_dpvo),
        "f_mm": int(f_mm) if f_mm is not None else None,
        "faststart": faststart,
        "optimized_for_webkit": optimized,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
        "results_path": str(results_path),
        "gvhmr_log_path": str(gvhmr_log),
        "preview_mp4_path": str(preview_path) if preview_path.exists() else None,
        "preview_incam_mp4_path": str(rendered_incam) if rendered_incam.exists() else None,
        "preview_global_front_mp4_path": str(rendered_global_front) if rendered_global_front.exists() else None,
        "preview_global_side_mp4_path": str(rendered_global_side) if rendered_global_side.exists() else None,
        "input_norm_mp4_path": str(rendered_input_norm) if rendered_input_norm.exists() else None,
        "preview_error": preview_error,
        "fallback_render_errors": render_errors if render_errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GVHMR demo and export SMPL-X NPZ.")
    parser.add_argument("--video", required=True, help="Path to an input MP4 file")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs into")
    parser.add_argument("--static-cam", action="store_true", help="Skip DPVO (recommended for static camera)")
    parser.add_argument("--use-dpvo", action="store_true", help="Enable DPVO visual odometry")
    parser.add_argument("--f-mm", type=int, default=None, help="Focal length in mm for fullframe camera")
    parser.add_argument("--skip_render", action="store_true", help="Skip preview rendering and directly return pose outputs")
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    try:
        result: dict[str, Any] = run_gvhmr(
            video,
            out_dir,
            static_cam=bool(args.static_cam),
            use_dpvo=bool(args.use_dpvo),
            f_mm=args.f_mm,
            skip_render=bool(args.skip_render),
        )
    except Exception as exc:
        # Best-effort: emit structured JSON so parent processes can surface a useful error.
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(result))
        raise SystemExit(1)

    print(json.dumps(result))
    if not bool(result.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
