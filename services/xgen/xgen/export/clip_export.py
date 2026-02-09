from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .schema import ClipData


def save_clip_npz(path: Path, clip: ClipData) -> None:
    clip.validate()
    payload = {
        "robot_qpos": clip.robot_qpos,
        "robot_qvel": clip.robot_qvel,
        "root_pose": clip.root_pose,
        "object_pose": clip.object_pose,
        "contact_graph": clip.contact_graph,
        "phase": clip.phase,
    }
    if clip.root_vel is not None:
        payload["root_vel"] = clip.root_vel
    if clip.object_vel is not None:
        payload["object_vel"] = clip.object_vel
    np.savez_compressed(path, **payload)


def save_metadata(path: Path, clip: ClipData) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(clip.metadata, handle, indent=2, sort_keys=True)


def export_clip(output_dir: str | Path, clip: ClipData, name: str = "clip") -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_path = output_path / f"{name}.npz"
    metadata_path = output_path / f"{name}.metadata.json"
    save_clip_npz(npz_path, clip)
    save_metadata(metadata_path, clip)
    return npz_path, metadata_path


def export_placeholder_clip(output_dir: str | Path, name: str = "clip") -> tuple[Path, Path]:
    """Generate a small placeholder clip for golden-path testing."""
    frames = 30
    robot_qpos = np.zeros((frames, 7))
    robot_qvel = np.zeros((frames, 7))
    root_pose = np.zeros((frames, 7))
    root_vel = np.zeros((frames, 3))
    object_pose = np.zeros((frames, 7))
    object_vel = np.zeros((frames, 3))
    contact_graph = np.zeros((frames, 4))
    phase = np.zeros((frames,))
    clip = ClipData(
        robot_qpos=robot_qpos,
        robot_qvel=robot_qvel,
        root_pose=root_pose,
        root_vel=root_vel,
        object_pose=object_pose,
        object_vel=object_vel,
        contact_graph=contact_graph,
        phase=phase,
        metadata={"placeholder": True, "frames": frames},
    )
    return export_clip(output_dir, clip, name=name)
