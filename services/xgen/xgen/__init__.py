"""XGen pipeline modules.

This package intentionally avoids importing heavy optional dependencies at import-time.
Many submodules depend on simulation/geometry stacks (mujoco, trimesh, cvxpy, etc.). To keep
lightweight utilities usable (e.g. pose wrappers, motion export), we expose top-level names
via lazy module attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # export/
    "ClipData": (".export", "ClipData"),
    "export_clip": (".export", "export_clip"),
    "export_placeholder_clip": (".export", "export_placeholder_clip"),
    "render_preview": (".export", "render_preview"),
    # pose/
    "PoseSequence": (".pose", "PoseSequence"),
    "estimate_pose_from_video": (".pose", "estimate_pose_from_video"),
    "save_pose_npz": (".pose", "save_pose_npz"),
    "estimate_smpl_from_video": (".pose", "estimate_smpl_from_video"),
    "estimate_smplx_from_video": (".pose", "estimate_smplx_from_video"),
    "ensure_smplx_fields": (".pose", "ensure_smplx_fields"),
    "save_smplx_npz": (".pose", "save_smplx_npz"),
    "convert_smpl_to_smplx": (".pose", "convert_smpl_to_smplx"),
    "resample_smplx_sequence": (".pose", "resample_smplx_sequence"),
    "smplx_npz_to_physhoi_motion": (".pose", "smplx_npz_to_physhoi_motion"),
    "load_physhoi_motion": (".pose", "load_physhoi_motion"),
    "validate_physhoi_motion": (".pose", "validate_physhoi_motion"),
    "summarize_physhoi_motion": (".pose", "summarize_physhoi_motion"),
    # retarget/
    "RetargetResult": (".retarget", "RetargetResult"),
    "retarget_upper_body": (".retarget", "retarget_upper_body"),
    "RetargetValidationResult": (".retarget", "RetargetValidationResult"),
    "validate_retarget": (".retarget", "validate_retarget"),
    # ingest/
    "convert_physhoi_motion_to_clip": (".ingest", "convert_physhoi_motion_to_clip"),
    # augment/
    "AugmentationResult": (".augment", "AugmentationResult"),
    "randomize_velocity": (".augment", "randomize_velocity"),
    "scale_mesh": (".augment", "scale_mesh"),
    "transform_contact_trajectory": (".augment", "transform_contact_trajectory"),
    "sweep_augmentations": (".augment", "sweep_augmentations"),
    # interaction/
    "AnchorResult": (".interaction", "AnchorResult"),
    "AnchorType": (".interaction", "AnchorType"),
    "compute_anchor": (".interaction", "compute_anchor"),
    "compute_contact_graph": (".interaction", "compute_contact_graph"),
    "force_closure_refine": (".interaction", "force_closure_refine"),
    "solve_force_closure_qp": (".interaction", "solve_force_closure_qp"),
    "ObjectPose": (".interaction", "ObjectPose"),
    "estimate_object_pose_from_bbox": (".interaction", "estimate_object_pose_from_bbox"),
    "track_object_pose": (".interaction", "track_object_pose"),
    "estimate_object_pose_sam3d": (".interaction", "estimate_object_pose_sam3d"),
    "PhaseSegments": (".interaction", "PhaseSegments"),
    "simulate_noncontact": (".interaction", "simulate_noncontact"),
    "simulate_noncontact_forward": (".interaction", "simulate_noncontact_forward"),
    "simulate_noncontact_backward": (".interaction", "simulate_noncontact_backward"),
    "segment_phases": (".interaction", "segment_phases"),
    "split_segments": (".interaction", "split_segments"),
    "synthesize_contact": (".interaction", "synthesize_contact"),
    "refine_contact_phase": (".interaction", "refine_contact_phase"),
    "ContactRefineResult": (".interaction", "ContactRefineResult"),
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

