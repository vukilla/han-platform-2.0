from __future__ import annotations

from typing import List
import random


def sample_generalization(augmented_variants: List[str], count: int) -> List[str]:
    if count <= 0:
        return []
    if not augmented_variants:
        return []
    if count >= len(augmented_variants):
        return list(augmented_variants)
    return random.sample(augmented_variants, count)
