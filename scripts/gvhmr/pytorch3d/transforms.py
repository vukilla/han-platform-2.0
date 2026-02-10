"""Pure-PyTorch subset of `pytorch3d.transforms` used by GVHMR.

This is intentionally incomplete. It exists to make GVHMR runnable on Windows
where official PyTorch3D wheels are not reliably available.

Supported APIs (as imported by GVHMR):
- axis_angle_to_matrix
- matrix_to_axis_angle
- rotation_6d_to_matrix
- matrix_to_rotation_6d
- quaternion_to_matrix
- matrix_to_quaternion
- quaternion_to_axis_angle
- so3_exp_map
- so3_log_map
- euler_angles_to_matrix
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Return skew-symmetric matrices for vectors v (..., 3) -> (..., 3, 3)."""
    v = torch.as_tensor(v)
    if v.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3), got {tuple(v.shape)}")
    x, y, z = v.unbind(dim=-1)
    o = torch.zeros_like(x)
    row0 = torch.stack([o, -z, y], dim=-1)
    row1 = torch.stack([z, o, -x], dim=-1)
    row2 = torch.stack([-y, x, o], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations to rotation matrices.

    Args:
        axis_angle: (..., 3) rotation vectors (axis * angle).
    Returns:
        (..., 3, 3) rotation matrices.
    """
    aa = torch.as_tensor(axis_angle)
    if aa.shape[-1] != 3:
        raise ValueError(f"axis_angle must have last dim 3, got {tuple(aa.shape)}")

    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)  # (..., 1)
    theta2 = theta * theta
    eps = 1e-6

    # Rodrigues using the un-normalized skew matrix K(omega).
    K = _skew(aa)  # (..., 3, 3)
    K2 = K @ K

    # A = sin(theta)/theta ; B = (1-cos(theta))/theta^2 with Taylor fallback near 0.
    A = torch.where(
        theta < eps,
        1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0,
        torch.sin(theta) / theta,
    )
    B = torch.where(
        theta < eps,
        0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0,
        (1.0 - torch.cos(theta)) / theta2,
    )

    eye = torch.eye(3, device=aa.device, dtype=aa.dtype)
    eye = eye.expand(aa.shape[:-1] + (3, 3))
    A = A[..., None]  # (..., 1, 1)
    B = B[..., None]
    return eye + A * K + B * K2


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle rotation vectors.

    Args:
        matrix: (..., 3, 3)
    Returns:
        (..., 3) axis-angle vectors.
    """
    R = torch.as_tensor(matrix)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (..., 3, 3), got {tuple(R.shape)}")

    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...,)

    # vee(R - R^T) / 2
    vee = torch.stack(
        [
            (R[..., 2, 1] - R[..., 1, 2]) * 0.5,
            (R[..., 0, 2] - R[..., 2, 0]) * 0.5,
            (R[..., 1, 0] - R[..., 0, 1]) * 0.5,
        ],
        dim=-1,
    )

    sin_theta = torch.sin(theta)
    eps = 1e-6
    scale = torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), theta / (sin_theta + 1e-8))
    # For small angles, vee ~= axis_angle already.
    aa = vee * scale[..., None]
    aa = torch.where((theta.abs() < eps)[..., None], vee * 2.0, aa)
    return aa


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices.

    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks".
    """
    x = torch.as_tensor(d6)
    if x.shape[-1] != 6:
        raise ValueError(f"rotation_6d must have last dim 6, got {tuple(x.shape)}")
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    R = torch.as_tensor(matrix)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (..., 3, 3), got {tuple(R.shape)}")
    return R[..., :, 0:2].reshape(R.shape[:-2] + (6,))


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (w, x, y, z) to rotation matrices."""
    q = torch.as_tensor(quaternion)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must have last dim 4, got {tuple(q.shape)}")
    q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (w, x, y, z)."""
    R = torch.as_tensor(matrix)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (..., 3, 3), got {tuple(R.shape)}")

    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    tr = m00 + m11 + m22

    # Numerically stable conversion based on the trace.
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)

    cond = tr > 0.0
    tr_sqrt = torch.sqrt(torch.clamp(tr + 1.0, min=0.0))
    q0 = 0.5 * tr_sqrt
    denom = 4.0 * q0 + 1e-8
    q1 = (R[..., 2, 1] - R[..., 1, 2]) / denom
    q2 = (R[..., 0, 2] - R[..., 2, 0]) / denom
    q3 = (R[..., 1, 0] - R[..., 0, 1]) / denom

    q = torch.where(cond[..., None], torch.stack([q0, q1, q2, q3], dim=-1), q)

    # Fallback branches when trace <= 0: pick the largest diagonal term.
    cond1 = (~cond) & (m00 >= m11) & (m00 >= m22)
    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 2.0
    qw1 = (R[..., 2, 1] - R[..., 1, 2]) / (s1 + 1e-8)
    qx1 = 0.25 * s1
    qy1 = (R[..., 0, 1] + R[..., 1, 0]) / (s1 + 1e-8)
    qz1 = (R[..., 0, 2] + R[..., 2, 0]) / (s1 + 1e-8)
    q = torch.where(cond1[..., None], torch.stack([qw1, qx1, qy1, qz1], dim=-1), q)

    cond2 = (~cond) & (~cond1) & (m11 >= m22)
    s2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=0.0)) * 2.0
    qw2 = (R[..., 0, 2] - R[..., 2, 0]) / (s2 + 1e-8)
    qx2 = (R[..., 0, 1] + R[..., 1, 0]) / (s2 + 1e-8)
    qy2 = 0.25 * s2
    qz2 = (R[..., 1, 2] + R[..., 2, 1]) / (s2 + 1e-8)
    q = torch.where(cond2[..., None], torch.stack([qw2, qx2, qy2, qz2], dim=-1), q)

    cond3 = (~cond) & (~cond1) & (~cond2)
    s3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=0.0)) * 2.0
    qw3 = (R[..., 1, 0] - R[..., 0, 1]) / (s3 + 1e-8)
    qx3 = (R[..., 0, 2] + R[..., 2, 0]) / (s3 + 1e-8)
    qy3 = (R[..., 1, 2] + R[..., 2, 1]) / (s3 + 1e-8)
    qz3 = 0.25 * s3
    q = torch.where(cond3[..., None], torch.stack([qw3, qx3, qy3, qz3], dim=-1), q)

    return torch.nn.functional.normalize(q, dim=-1, eps=1e-8)


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to axis-angle rotation vectors."""
    q = torch.as_tensor(quaternion)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must have last dim 4, got {tuple(q.shape)}")
    q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
    w = torch.clamp(q[..., 0], -1.0, 1.0)
    xyz = q[..., 1:4]
    sin_half = torch.linalg.norm(xyz, dim=-1)

    angle = 2.0 * torch.atan2(sin_half, w)
    eps = 1e-6
    axis = xyz / (sin_half[..., None] + 1e-8)
    aa = axis * angle[..., None]
    # For small angles, aa ~= 2 * xyz.
    aa = torch.where((sin_half < eps)[..., None], 2.0 * xyz, aa)
    return aa


def so3_exp_map(log_rot: torch.Tensor) -> torch.Tensor:
    """Exponential map from so(3) to SO(3)."""
    return axis_angle_to_matrix(log_rot)


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """Log map from SO(3) to so(3)."""
    return matrix_to_axis_angle(R)


def _axis_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    o = torch.zeros_like(angle)
    i = torch.ones_like(angle)
    if axis == "X":
        row0 = torch.stack([i, o, o], dim=-1)
        row1 = torch.stack([o, c, -s], dim=-1)
        row2 = torch.stack([o, s, c], dim=-1)
    elif axis == "Y":
        row0 = torch.stack([c, o, s], dim=-1)
        row1 = torch.stack([o, i, o], dim=-1)
        row2 = torch.stack([-s, o, c], dim=-1)
    elif axis == "Z":
        row0 = torch.stack([c, -s, o], dim=-1)
        row1 = torch.stack([s, c, o], dim=-1)
        row2 = torch.stack([o, o, i], dim=-1)
    else:
        raise ValueError(f"Invalid axis {axis!r}. Expected one of 'X','Y','Z'.")
    return torch.stack([row0, row1, row2], dim=-2)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """Convert Euler angles to rotation matrices.

    Args:
        euler_angles: (..., 3) angles in radians.
        convention: 3-letter string like "XYZ" or "YXZ".
    """
    if not isinstance(convention, str) or len(convention) != 3:
        raise ValueError("convention must be a 3-letter string, e.g. 'XYZ'")
    for c in convention:
        if c not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid convention {convention!r}")

    angles = torch.as_tensor(euler_angles)
    if angles.shape[-1] != 3:
        raise ValueError(f"euler_angles must have last dim 3, got {tuple(angles.shape)}")

    a0, a1, a2 = angles.unbind(dim=-1)
    R0 = _axis_rotation(convention[0], a0)
    R1 = _axis_rotation(convention[1], a1)
    R2 = _axis_rotation(convention[2], a2)
    return R0 @ R1 @ R2

