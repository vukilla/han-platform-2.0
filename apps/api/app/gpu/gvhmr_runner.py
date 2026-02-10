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


def _render_gvhmr_preview(video_path: Path, results_path: Path, out_mp4_path: Path) -> dict[str, Any]:
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
        transl = params.get("transl", None)
        transl = (
            _as_tensor(transl).float()
            if transl is not None
            else torch.zeros((body_pose.shape[0], 3), dtype=torch.float32)
        )
        betas = _as_tensor(params.get("betas")).float()

        # Add batch dim expected by SmplxLite.
        if body_pose.ndim == 2:
            body_pose = body_pose[None, ...]
        if global_orient.ndim == 2:
            global_orient = global_orient[None, ...]
        if transl.ndim == 2:
            transl = transl[None, ...]
        if betas.ndim == 1:
            betas = betas[None, ...]

        # Some outputs store betas per-frame; slice if needed.
        if betas.ndim == 3 and betas.shape[1] >= int(frame_ids.max(initial=0) + 1):
            betas_out = betas[:, frame_ids]
        else:
            betas_out = betas

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
        for a, b in coco_edges:
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

    # === Normalize + face-Z transform for global view (approx GVHMR `move_to_start_point_face_z`) ===
    min_y = verts_global[..., 1].min()
    offset = joints_global_smpl24[0, 0].detach().clone()
    offset[1] = min_y
    joints_global_smpl24 = joints_global_smpl24 - offset[None, None, :]
    verts_global = verts_global - offset[None, None, :]

    T_ay2ayfz = compute_T_ayfz2ay(joints_global_smpl24[[0]], inverse=True)  # (1, 4, 4)
    T_seq = T_ay2ayfz.repeat(joints_global_smpl24.shape[0], 1, 1)  # (F, 4, 4)
    joints_global_smpl24 = apply_T_on_points(joints_global_smpl24, T_seq)
    verts_global = apply_T_on_points(verts_global, T_seq)

    # === Static global camera (similar to GVHMR `get_global_cameras_static`) ===
    targets = joints_global_smpl24.mean(dim=1).detach().cpu()  # (F, 3)
    targets[:, 1] = 0.0
    target_center = targets.mean(dim=0)  # (3,)
    target_scale = torch.norm(targets - target_center[None, :], dim=-1).max()
    target_scale_val = float(max(target_scale.item(), 1.0)) * 2.0  # beta=2.0

    vec_rad = float(math.radians(45.0))
    vec = torch.tensor([math.sin(vec_rad), 0.0, math.cos(vec_rad)], dtype=torch.float32)
    vec = vec / torch.norm(vec)
    cam_pos = target_center + vec * target_scale_val
    cam_pos[1] = target_scale_val * float(math.tan(math.radians(20.0))) + 1.0  # cam_height_degree=20, target_center_height=1.0

    cam_target = target_center.clone()
    cam_target[1] = 1.0
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

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

    R_wc, t_wc = _look_at_world_to_cam(cam_pos, cam_target, up)

    def _project_global(points_world: torch.Tensor) -> torch.Tensor:
        pw = points_world.detach().cpu()
        pc = torch.matmul(pw, R_wc.T) + t_wc
        return _project_pinhole(pc, K_global)

    # === Ground grid parameters (fixed across frames) ===
    root_points = joints_global_smpl24[:, 0, :].detach().cpu().numpy()
    cx = float((root_points[:, 0].max() + root_points[:, 0].min()) * 0.5)
    cz = float((root_points[:, 2].max() + root_points[:, 2].min()) * 0.5)
    all_pts = joints_global_smpl24.detach().cpu().numpy().reshape(-1, 3)
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

    def _render_body_mesh(img: np.ndarray, verts_world: np.ndarray) -> None:
        # Render as a depth-shaded dense point cloud (fast + no extra deps).
        verts_cam = verts_world @ R_wc_np.T + t_wc_np  # (V, 3)
        z = verts_cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return

        x = verts_cam[valid, 0]
        y = verts_cam[valid, 1]
        z = z[valid]
        u = (x / z) * float(K_global_np[0, 0]) + float(K_global_np[0, 2])
        v = (y / z) * float(K_global_np[1, 1]) + float(K_global_np[1, 2])
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        in_img = (ui >= 0) & (ui < panel_w) & (vi >= 0) & (vi < panel_h)
        if not np.any(in_img):
            return

        ui = ui[in_img]
        vi = vi[in_img]
        z = z[in_img].astype(np.float32)

        idx = (vi.astype(np.int64) * int(panel_w) + ui.astype(np.int64)).astype(np.int64)
        depth_flat = np.full(int(panel_h) * int(panel_w), np.inf, dtype=np.float32)
        np.minimum.at(depth_flat, idx, z)
        depth = depth_flat.reshape((int(panel_h), int(panel_w)))

        valid_depth = np.isfinite(depth)
        if not np.any(valid_depth):
            return

        # Close small holes so the body looks like a continuous surface.
        mask_u8 = (valid_depth.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = mask_u8 > 0

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

            _render_body_mesh(img, verts_global_np[jidx])

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
        "layout": "global_skeleton",
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
