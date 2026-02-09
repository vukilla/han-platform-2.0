"""Pose-related utilities.

This subpackage intentionally uses lazy imports to avoid importing every CV/SMPL dependency
when only a single helper is needed (e.g., GVHMR wrapper, PhysHOI motion export).
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PoseSequence": (".types", "PoseSequence"),
    "estimate_pose_from_video": (".mediapipe_pose", "estimate_pose_from_video"),
    "save_pose_npz": (".mediapipe_pose", "save_pose_npz"),
    "estimate_smpl_from_video": (".estimator", "estimate_smpl_from_video"),
    "estimate_smplx_from_video": (".gvhmr_pose", "estimate_smplx_from_video"),
    "ensure_smplx_fields": (".smplx_convert", "ensure_smplx_fields"),
    "save_smplx_npz": (".smplx_convert", "save_smplx_npz"),
    "convert_smpl_to_smplx": (".smplx_convert", "convert_smpl_to_smplx"),
    "resample_smplx_sequence": (".smplx_resample", "resample_smplx_sequence"),
    "smplx_npz_to_physhoi_motion": (".physhoi_motion", "smplx_npz_to_physhoi_motion"),
    "load_physhoi_motion": (".physhoi_motion", "load_physhoi_motion"),
    "validate_physhoi_motion": (".physhoi_motion", "validate_physhoi_motion"),
    "summarize_physhoi_motion": (".physhoi_motion", "summarize_physhoi_motion"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals().keys()) + list(_EXPORTS.keys())))

