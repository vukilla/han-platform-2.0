from .anchors import AnchorResult, AnchorType, compute_anchor
from .contact_graph import compute_contact_graph
from .contact_synth import synthesize_contact, refine_contact_phase, ContactRefineResult
from .force_closure_refine import force_closure_refine, solve_force_closure_qp
from .noncontact_sim import simulate_noncontact, simulate_noncontact_forward, simulate_noncontact_backward
from .segment import PhaseSegments, segment_phases, split_segments
from .stitch import smooth_phase_transitions

try:  # Optional deps (opencv, etc.)
    from .object_init import ObjectPose, estimate_object_pose_from_bbox, track_object_pose, estimate_object_pose_sam3d
except Exception:  # pragma: no cover
    ObjectPose = None  # type: ignore[assignment]
    estimate_object_pose_from_bbox = None  # type: ignore[assignment]
    track_object_pose = None  # type: ignore[assignment]
    estimate_object_pose_sam3d = None  # type: ignore[assignment]

__all__ = [
    "AnchorResult",
    "AnchorType",
    "compute_anchor",
    "PhaseSegments",
    "segment_phases",
    "split_segments",
    "compute_contact_graph",
    "synthesize_contact",
    "refine_contact_phase",
    "ContactRefineResult",
    "force_closure_refine",
    "solve_force_closure_qp",
    "simulate_noncontact",
    "simulate_noncontact_forward",
    "simulate_noncontact_backward",
    "smooth_phase_transitions",
]

if ObjectPose is not None:  # pragma: no cover
    __all__ += [
        "ObjectPose",
        "estimate_object_pose_from_bbox",
        "track_object_pose",
        "estimate_object_pose_sam3d",
    ]
