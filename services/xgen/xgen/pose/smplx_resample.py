from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from .smplx_convert import SmplxSequence


def _interp_time_series(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    frames = values.shape[0]
    if frames <= 1:
        return values.copy()
    t_src = np.arange(frames) / source_fps
    t_dst = np.arange(int(round(frames * target_fps / source_fps))) / target_fps
    out = np.empty((t_dst.shape[0], values.shape[1]), dtype=np.float32)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(t_dst, t_src, values[:, i])
    return out


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if values.ndim == 1:
        values = values[:, None]
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.vstack([
        np.convolve(padded[:, i], kernel, mode="valid") for i in range(values.shape[1])
    ]).T
    return smoothed.astype(np.float32)


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, rotvec / angle, np.zeros_like(rotvec))
    half = angle * 0.5
    sin_half = np.sin(half)
    xyz = axis * sin_half
    w = np.cos(half)
    return np.concatenate([xyz, w], axis=-1)


def _quat_to_euler_zyx(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = [quat[..., i] for i in range(4)]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.stack([roll, pitch, yaw], axis=-1)


def _euler_zyx_to_quat(euler: np.ndarray) -> np.ndarray:
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([x, y, z, w], axis=-1)


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)
    w = np.clip(quat[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    axis = np.where(sin_half[..., None] > 1e-8, quat[..., :3] / sin_half[..., None], 0.0)
    return axis * angle[..., None]


def _smooth_yaw(global_orient: np.ndarray, window: int) -> np.ndarray:
    quat = _axis_angle_to_quat(global_orient)
    euler = _quat_to_euler_zyx(quat)
    yaw = np.unwrap(euler[:, 2])
    yaw_smoothed = _smooth_series(yaw, window=window).reshape(-1)
    euler[:, 2] = yaw_smoothed
    quat_smoothed = _euler_zyx_to_quat(euler)
    return _quat_to_axis_angle(quat_smoothed).astype(np.float32)


def resample_smplx_sequence(
    seq: SmplxSequence,
    source_fps: float,
    target_fps: float,
    smooth_window: int = 5,
    smooth_yaw: bool = True,
    contact_mask: Optional[np.ndarray] = None,
) -> SmplxSequence:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps must be > 0")

    def resample(arr: np.ndarray) -> np.ndarray:
        return _interp_time_series(arr, source_fps, target_fps)

    global_orient = resample(seq.global_orient)
    body_pose = resample(seq.body_pose)
    transl = resample(seq.transl)
    left_hand = resample(seq.left_hand_pose)
    right_hand = resample(seq.right_hand_pose)
    jaw = resample(seq.jaw_pose)
    leye = resample(seq.leye_pose)
    reye = resample(seq.reye_pose)
    expr = resample(seq.expression)

    if smooth_window > 1:
        transl_smoothed = _smooth_series(transl, smooth_window)
        if contact_mask is not None:
            contact_mask = resample(contact_mask.astype(np.float32))
            mask = (contact_mask > 0.5).reshape(-1)
            transl_smoothed[mask] = transl[mask]
        transl = transl_smoothed

    if smooth_yaw and smooth_window > 1:
        global_orient_smoothed = _smooth_yaw(global_orient, smooth_window)
        if contact_mask is not None:
            mask = (contact_mask > 0.5).reshape(-1)
            global_orient_smoothed[mask] = global_orient[mask]
        global_orient = global_orient_smoothed

    betas = seq.betas
    if betas.ndim > 1:
        betas = betas[0]

    return replace(
        seq,
        global_orient=global_orient.astype(np.float32),
        body_pose=body_pose.astype(np.float32),
        betas=betas.astype(np.float32),
        transl=transl.astype(np.float32),
        left_hand_pose=left_hand.astype(np.float32),
        right_hand_pose=right_hand.astype(np.float32),
        jaw_pose=jaw.astype(np.float32),
        leye_pose=leye.astype(np.float32),
        reye_pose=reye.astype(np.float32),
        expression=expr.astype(np.float32),
    )
