from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable

import numpy as np


class AnchorType(str, Enum):
    PALMS_MIDPOINT = "palms_midpoint"
    SINGLE_BODY_PART = "single_body_part"


@dataclass
class AnchorResult:
    anchor: np.ndarray
    anchors_by_frame: np.ndarray


def compute_anchor(
    body_positions: Dict[str, np.ndarray],
    anchor_type: AnchorType,
    key_bodies: Iterable[str] | None = None,
) -> AnchorResult:
    if anchor_type == AnchorType.PALMS_MIDPOINT:
        left = body_positions.get("left_hand")
        right = body_positions.get("right_hand")
        if left is None or right is None:
            raise ValueError("missing left_hand/right_hand positions for palms_midpoint")
        anchors = (left + right) / 2.0
    elif anchor_type == AnchorType.SINGLE_BODY_PART:
        if not key_bodies:
            raise ValueError("key_bodies required for single_body_part")
        body = next(iter(key_bodies))
        positions = body_positions.get(body)
        if positions is None:
            raise ValueError(f"missing body positions for {body}")
        anchors = positions
    else:
        raise ValueError(f"unknown anchor_type: {anchor_type}")

    anchor = anchors[0] if anchors.ndim > 1 else anchors
    return AnchorResult(anchor=anchor, anchors_by_frame=anchors)
