from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SkillSample:
    skill_id: str
    pattern_id: Optional[str]
    path: Path


class MultiSkillSampler:
    def __init__(self, clips_by_skill: Dict[str, List[Path]], balanced: bool = True):
        self.clips_by_skill = clips_by_skill
        self.skills = list(clips_by_skill.keys())
        self.balanced = balanced
        if not self.skills:
            raise ValueError("No skills provided")
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        if self.balanced:
            return np.ones((len(self.skills),), dtype=np.float32) / len(self.skills)
        counts = np.array([len(self.clips_by_skill[s]) for s in self.skills], dtype=np.float32)
        weights = counts / counts.sum()
        return weights

    def sample(self) -> SkillSample:
        skill_idx = int(np.random.choice(len(self.skills), p=self.weights))
        skill_id = self.skills[skill_idx]
        clips = self.clips_by_skill[skill_id]
        clip = Path(np.random.choice(clips))
        return SkillSample(skill_id=skill_id, pattern_id=None, path=clip)
