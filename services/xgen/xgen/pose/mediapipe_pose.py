from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .types import PoseSequence


def estimate_pose_from_video(
    video_path: str | Path,
    max_frames: Optional[int] = None,
    model_complexity: int = 1,
) -> PoseSequence:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    joints_3d = []
    joints_2d = []
    visibility = []
    landmark_names = [l.name for l in mp.solutions.pose.PoseLandmark]

    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_world_landmarks:
                world = results.pose_world_landmarks.landmark
                image_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
                joints_3d.append([[lm.x, lm.y, lm.z] for lm in world])
                if image_landmarks:
                    joints_2d.append([[lm.x, lm.y] for lm in image_landmarks])
                    visibility.append([lm.visibility for lm in image_landmarks])
                else:
                    joints_2d.append([[np.nan, np.nan] for _ in world])
                    visibility.append([0.0 for _ in world])
            else:
                joints_3d.append([[np.nan, np.nan, np.nan] for _ in landmark_names])
                joints_2d.append([[np.nan, np.nan] for _ in landmark_names])
                visibility.append([0.0 for _ in landmark_names])
    finally:
        cap.release()
        pose.close()

    return PoseSequence(
        fps=float(fps),
        joints_3d=np.array(joints_3d, dtype=np.float32),
        joints_2d=np.array(joints_2d, dtype=np.float32),
        visibility=np.array(visibility, dtype=np.float32),
        landmark_names=landmark_names,
    )


def save_pose_npz(output_path: str | Path, sequence: PoseSequence) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fps=sequence.fps,
        joints_3d=sequence.joints_3d,
        joints_2d=sequence.joints_2d,
        visibility=sequence.visibility,
        landmark_names=np.array(sequence.landmark_names),
    )
    return output
