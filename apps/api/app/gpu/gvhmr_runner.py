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

    try:
        if not expected.exists():
            os.symlink(heavy, expected, target_is_directory=True)
    except OSError:
        # Symlinks are often unavailable on Windows without Developer Mode or Administrator privileges.
        # Fall back to copying the required checkpoints. This is slower but robust.
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
            "Tip: for the platform flow, you can upload this file once via the Web UI (`/gvhmr`) or API:\n"
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


def _render_gvhmr_preview(video_path: Path, results_path: Path, out_mp4_path: Path) -> dict[str, Any]:
    """Create a lightweight skeleton preview video without PyTorch3D renderer.

    Output is a skeleton-only MP4. The Web UI shows it side-by-side with the original video.

    This is intentionally a fallback preview: GVHMR's official mesh renderer depends on
    PyTorch3D renderer components that are painful to install on Windows. The skeleton
    preview gives users immediate visual feedback in the Web UI.
    """
    import math

    import cv2
    import numpy as np
    import torch

    # Ensure GVHMR python packages (hmr4d/...) and the pytorch3d stub are importable.
    gvhmr_root = _resolve_gvhmr_root()
    if str(gvhmr_root) not in sys.path:
        sys.path.insert(0, str(gvhmr_root))

    from hmr4d.utils.body_model.smplx_lite import SmplxLiteCoco17

    def _as_tensor(x) -> torch.Tensor:
        if torch.is_tensor(x):
            return x
        return torch.from_numpy(np.asarray(x))

    pred = torch.load(results_path, map_location="cpu")
    smpl_params = pred.get("smpl_params_global") or {}
    if not smpl_params:
        raise ValueError("hmr4d_results.pt missing smpl_params_global")

    body_pose = _as_tensor(smpl_params.get("body_pose")).float()
    global_orient = _as_tensor(smpl_params.get("global_orient")).float()
    transl = smpl_params.get("transl", None)
    transl = _as_tensor(transl).float() if transl is not None else torch.zeros((body_pose.shape[0], 3), dtype=torch.float32)
    betas = _as_tensor(smpl_params.get("betas")).float()

    # Add batch dim expected by SmplxLite.
    if body_pose.ndim == 2:
        body_pose = body_pose[None, ...]
    if global_orient.ndim == 2:
        global_orient = global_orient[None, ...]
    if transl.ndim == 2:
        transl = transl[None, ...]
    if betas.ndim == 1:
        betas = betas[None, ...]

    total_frames = int(body_pose.shape[1])
    if total_frames <= 0:
        raise ValueError("smpl_params_global has no frames")

    # Cap preview cost by decimating long videos.
    max_preview_frames = 450
    stride = max(1, int(math.ceil(total_frames / max_preview_frames)))
    frame_ids = np.arange(0, total_frames, stride, dtype=np.int32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmplxLiteCoco17().to(device)
    with torch.no_grad():
        joints = model(
            body_pose[:, frame_ids].to(device),
            betas.to(device),
            global_orient[:, frame_ids].to(device),
            transl[:, frame_ids].to(device),
        )
    joints_np = joints.squeeze(0).detach().cpu().numpy()  # (F', 17, 3)

    # Center around pelvis (midpoint of left/right hip in COCO17 indices).
    pelvis = 0.5 * (joints_np[:, 11, :] + joints_np[:, 12, :])  # (F', 3)
    joints_np = joints_np - pelvis[:, None, :]

    # Rotate about Y axis so Z contributes to apparent X (gives a 3D feel in orthographic projection).
    theta = math.radians(35.0)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    joints_np = joints_np @ R.T

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video with OpenCV: {video_path}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps_out = max(1.0, fps_in / float(stride))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w_in <= 0 or h_in <= 0:
        w_in, h_in = 640, 360

    # Preserve the *exact* input resolution so the Web UI can display the original and GVHMR
    # preview at the same size with no aspect-ratio drift from rounding.
    #
    # (We used to downscale to a max height, but users expect a 1:1 match with their upload.)
    skel_w = int(w_in)
    skel_h = int(h_in)

    x = joints_np[:, :, 0]
    y = joints_np[:, :, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    x_center = 0.5 * (xmin + xmax)
    y_center = 0.5 * (ymin + ymax)
    span_x = max(1e-6, xmax - xmin)
    span_y = max(1e-6, ymax - ymin)
    skel_scale = 0.9 * min(float(skel_w) / span_x, float(skel_h) / span_y)

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps_out, (skel_w, skel_h))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create VideoWriter for: {out_mp4_path}")

    try:
        frame_idx = 0
        for jidx in range(joints_np.shape[0]):
            ok, frame = cap.read()
            if not ok:
                break
            # We asked OpenCV for all frames; only keep ones matching our stride.
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            skel = np.zeros((skel_h, skel_w, 3), dtype=np.uint8)
            skel[:] = (14, 12, 10)
            pts = joints_np[jidx]
            u = ((pts[:, 0] - x_center) * skel_scale + skel_w * 0.5).astype(np.int32)
            v = ((-(pts[:, 1] - y_center)) * skel_scale + skel_h * 0.55).astype(np.int32)

            for a, b in coco_edges:
                cv2.line(skel, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), (70, 255, 120), 3)
            for k in range(pts.shape[0]):
                cv2.circle(skel, (int(u[k]), int(v[k])), 4, (70, 255, 120), -1)

            cv2.putText(
                skel,
                "GVHMR 3D skeleton (preview)",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )

            writer.write(skel)
    finally:
        cap.release()
        writer.release()

    return {
        "ok": True,
        "preview_mp4": str(out_mp4_path),
        "fps_in": fps_in,
        "fps_out": fps_out,
        "stride": int(stride),
        "frames_out": int(joints_np.shape[0]),
    }


def run_gvhmr(video_path: Path, output_dir: Path, *, static_cam: bool, use_dpvo: bool, f_mm: int | None) -> dict[str, Any]:
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

    cmd = _gvhmr_python_cmd() + [
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
        proc = subprocess.run(cmd, check=False, cwd=str(gvhmr_root), env=env, stdout=handle, stderr=handle)

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

    preview_path = output_dir / f"{video_path.stem}_gvhmr_preview.mp4"
    preview = None
    preview_error = None
    try:
        preview = _render_gvhmr_preview(video_path, results_path, preview_path)
    except Exception as exc:  # noqa: BLE001
        preview = None
        preview_error = f"{type(exc).__name__}: {exc}"

    meta = {
        "ok": True,
        "video": str(video_path),
        "results_path": str(results_path),
        "output_npz": str(npz_path),
        "gvhmr_log": str(gvhmr_log),
        "preview_mp4": str(preview_path) if preview_path.exists() else None,
        "preview_error": preview_error,
        "python_cmd": cmd[:3] if cmd[:2] == ["cmd.exe", "/c"] else cmd[:1],
        "static_cam": bool(static_cam),
        "use_dpvo": bool(use_dpvo),
        "f_mm": int(f_mm) if f_mm is not None else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
        "results_path": str(results_path),
        "gvhmr_log_path": str(gvhmr_log),
        "preview_mp4_path": str(preview_path) if preview_path.exists() else None,
        "preview_error": preview_error,
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

    try:
        result: dict[str, Any] = run_gvhmr(
            video,
            out_dir,
            static_cam=bool(args.static_cam),
            use_dpvo=bool(args.use_dpvo),
            f_mm=args.f_mm,
        )
    except Exception as exc:
        # Best-effort: emit structured JSON so parent processes can surface a useful error.
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result))
        raise SystemExit(1)

    print(json.dumps(result))
    if not bool(result.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
