from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import trimesh


@dataclass
class ObjectPose:
    position: np.ndarray
    quaternion: np.ndarray


def estimate_depth_from_bbox(
    bbox: Tuple[int, int, int, int],
    object_size_m: float,
    frame_width: int,
    fov_degrees: float = 60.0,
) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError("invalid bbox size")
    fov_rad = np.deg2rad(fov_degrees)
    focal = (frame_width / 2) / np.tan(fov_rad / 2)
    return (object_size_m * focal) / max(w, h)


def estimate_object_pose_from_bbox(
    bbox: Tuple[int, int, int, int],
    object_size_m: float,
    frame_width: int,
    frame_height: int,
    fov_degrees: float = 60.0,
) -> ObjectPose:
    x, y, w, h = bbox
    depth = estimate_depth_from_bbox(bbox, object_size_m, frame_width, fov_degrees)
    cx = x + w / 2
    cy = y + h / 2
    nx = (cx - frame_width / 2) / frame_width
    ny = (cy - frame_height / 2) / frame_height
    position = np.array([nx * depth, ny * depth, depth], dtype=np.float32)
    quaternion = np.array([0, 0, 0, 1], dtype=np.float32)
    return ObjectPose(position=position, quaternion=quaternion)


def track_object_pose(
    video_path: str,
    init_bbox: Tuple[int, int, int, int],
    object_size_m: float,
    fov_degrees: float = 60.0,
    max_frames: int | None = None,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = cv2.TrackerCSRT_create()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("failed to read first frame")
    tracker.init(frame, init_bbox)

    poses = []
    count = 0
    while ok:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        if max_frames and count > max_frames:
            break
        success, bbox = tracker.update(frame)
        if not success:
            bbox = init_bbox
        pose = estimate_object_pose_from_bbox(
            bbox=tuple(int(v) for v in bbox),
            object_size_m=object_size_m,
            frame_width=frame_width,
            frame_height=frame_height,
            fov_degrees=fov_degrees,
        )
        poses.append(np.concatenate([pose.position, pose.quaternion]))
    cap.release()
    return np.stack(poses, axis=0) if poses else np.empty((0, 7), dtype=np.float32)


def estimate_object_pose_sam3d(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
    config_tag: str = "hf",
) -> Tuple[ObjectPose, str | None]:
    """
    Estimate object pose using SAM-3D Objects.
    Returns (ObjectPose, glb_path).
    """
    root = os.environ.get("SAM3D_ROOT")
    if root:
        sam3d_root = Path(root).expanduser().resolve()
    else:
        sam3d_root = Path(__file__).resolve().parents[3] / "external" / "sam3d"
    if not sam3d_root.exists():
        raise FileNotFoundError("SAM-3D repo not found. Set SAM3D_ROOT or clone into external/sam3d.")

    sys.path.append(str(sam3d_root / "notebook"))
    from inference import Inference, load_image, load_mask  # type: ignore

    config_path = sam3d_root / "checkpoints" / config_tag / "pipeline.yaml"
    inference = Inference(str(config_path), compile=False)
    image = load_image(str(image_path))
    mask = load_mask(str(mask_path)) if mask_path else None
    output = inference(image, mask, seed=seed)

    glb = output.get("glb")
    glb_path = None
    if output_dir is None:
        output_dir = Path(image_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if glb is not None:
        glb_path = str(output_dir / "sam3d_object.glb")
        try:
            glb.export(glb_path)
            mesh = trimesh.load(glb_path)
        except Exception:
            mesh = glb
    else:
        mesh = None

    if mesh is not None and hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
        centroid = np.array(mesh.vertices).mean(axis=0).astype(np.float32)
    else:
        centroid = np.zeros(3, dtype=np.float32)
    quaternion = np.array([0, 0, 0, 1], dtype=np.float32)
    return ObjectPose(position=centroid, quaternion=quaternion), glb_path
