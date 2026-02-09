from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from xmimic.robot_spec import RobotSpec

_FIELD_ALIASES: Dict[str, List[str]] = {
    # Paper naming -> common implementation naming.
    "gravity": ["projected_gravity"],
    # "Action" in Table IV is typically the previous action fed back as an observation.
    "action": ["prev_action"],
    "pd_error": ["dof_pos_error"],
    "ref_body_pos": ["ref_key_body_pos"],
    "delta_body_pos": ["delta_key_body_pos"],
}


@dataclass
class ObsField:
    name: str
    size: int


@dataclass
class ObservationConfig:
    fields: List[ObsField]
    history_fields: Optional[List[ObsField]] = None
    history: int = 1
    normalize: bool = True

    @property
    def dim(self) -> int:
        base = sum(field.size for field in self.fields)
        if not self.history_fields:
            return base * self.history
        hist = sum(field.size for field in self.history_fields)
        return base + hist * self.history


class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.count = 0
        self.mean = np.zeros((dim,), dtype=np.float32)
        self.var = np.ones((dim,), dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float32)
        self.count += 1
        if self.count == 1:
            self.mean = x.copy()
            self.var = np.ones_like(x)
            return
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var = ((self.count - 1) * self.var + delta * delta2) / self.count

    def normalize(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.update(x)
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)


class HistoryBuffer:
    def __init__(self, dim: int, history: int):
        self.dim = dim
        self.history = history
        self.buffer = np.zeros((history, dim), dtype=np.float32)
        self.index = 0
        self.filled = False

    def push(self, x: np.ndarray) -> np.ndarray:
        self.buffer[self.index] = x
        self.index = (self.index + 1) % self.history
        if self.index == 0:
            self.filled = True
        if not self.filled:
            return self.buffer[: self.index].reshape(-1)
        # return stacked oldest -> newest
        idx = np.arange(self.index, self.index + self.history) % self.history
        return self.buffer[idx].reshape(-1)


class ObservationPipeline:
    def __init__(self, config: ObservationConfig):
        self.config = config
        base_dim = sum(f.size for f in config.fields)
        hist_dim = sum(f.size for f in config.history_fields) if config.history_fields else 0

        self.curr_normalizer = RunningNorm(base_dim) if config.normalize else None
        self.hist_normalizer = RunningNorm(hist_dim) if (config.normalize and config.history_fields) else None

        # Two modes:
        # 1) legacy: no explicit history_fields, stack full vector (base_dim) for `history` frames
        # 2) HumanX-style: output = [current_fields] + stack(history_fields) for `history` frames
        if config.history_fields:
            self.history = HistoryBuffer(hist_dim, config.history)
        else:
            self.history = HistoryBuffer(base_dim, config.history)

    def _get(self, obs_dict: Dict[str, np.ndarray], field: ObsField) -> np.ndarray:
        if field.name in obs_dict:
            return obs_dict[field.name]
        for alt in _FIELD_ALIASES.get(field.name, []):
            if alt in obs_dict:
                return obs_dict[alt]
        raise KeyError(f"Missing observation field: {field.name}")

    def build(self, obs_dict: Dict[str, np.ndarray], update_norm: bool = True) -> np.ndarray:
        curr_parts = []
        for field in self.config.fields:
            value = np.asarray(self._get(obs_dict, field), dtype=np.float32).reshape(-1)
            if value.size != field.size:
                raise ValueError(f"Field {field.name} expected {field.size} values, got {value.size}")
            curr_parts.append(value)

        curr = np.concatenate(curr_parts, axis=0)
        if self.curr_normalizer is not None:
            curr = self.curr_normalizer.normalize(curr, update=update_norm)

        if not self.config.history_fields:
            return self.history.push(curr)

        hist_parts = []
        for field in self.config.history_fields:
            value = np.asarray(self._get(obs_dict, field), dtype=np.float32).reshape(-1)
            if value.size != field.size:
                raise ValueError(f"Field {field.name} expected {field.size} values, got {value.size}")
            hist_parts.append(value)
        hist = np.concatenate(hist_parts, axis=0)
        if self.hist_normalizer is not None:
            hist = self.hist_normalizer.normalize(hist, update=update_norm)
        hist_stack = self.history.push(hist)
        return np.concatenate([curr, hist_stack], axis=0)


def default_obs_config() -> ObservationConfig:
    """Default observation ordering based on HumanX Table IV (approx).

    Adjust sizes to match your simulator/robot.
    """
    return ObservationConfig(
        fields=[
            ObsField("base_ang_vel", 3),
            ObsField("gravity", 3),
            ObsField("dof_pos", 24),
            ObsField("dof_vel", 24),
            ObsField("action", 24),
            ObsField("pd_error", 24),
            ObsField("object_pos", 3),
            ObsField("object_rot", 4),
        ],
        history=4,
        normalize=True,
    )


def humanx_teacher_obs_config(
    robot: RobotSpec,
    *,
    num_skills: int = 10,
    include_object_rot: bool = False,
    include_target_object_pos: bool = False,
    include_target_object_rot: bool = False,
    history: int = 4,
    normalize: bool = True,
) -> ObservationConfig:
    """HumanX-style privileged teacher observation.

    This follows the intent of Table IV:
    - proprioception + history
    - skill label
    - optional object state (privileged)
    - optional target object pose (for goal-conditioned tasks)
    - optional reference/key-body positions (privileged)
    """
    dof = len(robot.joint_names)
    key = len(robot.key_bodies)

    fields: List[ObsField] = [
        ObsField("base_ang_vel", 3),
        ObsField("gravity", 3),
        ObsField("dof_pos", dof),
        ObsField("dof_vel", dof),
        ObsField("action", dof),
        ObsField("pd_error", dof),  # PD error proxy
        # Privileged reference signals. These are *not* given to the student.
        ObsField("ref_body_pos", key * 3),
        ObsField("delta_body_pos", key * 3),
        # Teacher observes object position (privileged state).
        ObsField("object_pos", 3),
    ]

    if include_object_rot:
        fields.append(ObsField("object_rot", 4))

    if include_target_object_pos:
        fields.append(ObsField("target_object_pos", 3))
    if include_target_object_rot:
        fields.append(ObsField("target_object_rot", 4))

    fields.append(ObsField("skill_label", num_skills))

    history_fields = [
        ObsField("base_ang_vel", 3),
        ObsField("gravity", 3),
        ObsField("dof_pos", dof),
        ObsField("dof_vel", dof),
        ObsField("action", dof),
    ]
    return ObservationConfig(fields=fields, history_fields=history_fields, history=history, normalize=normalize)


def humanx_student_obs_config(
    robot: RobotSpec,
    *,
    num_skills: int = 10,
    mode: str = "nep",  # "nep" or "mocap"
    include_object_rot: bool = False,
    mocap_dropout_mask: bool = False,
    history: int = 4,
    normalize: bool = True,
) -> ObservationConfig:
    """HumanX deployable student observation (NEP or MoCap)."""
    dof = len(robot.joint_names)

    fields: List[ObsField] = [
        ObsField("base_ang_vel", 3),
        ObsField("gravity", 3),
        ObsField("dof_pos", dof),
        ObsField("dof_vel", dof),
        ObsField("action", dof),
        ObsField("pd_error", dof),
    ]

    if mode not in ("nep", "mocap"):
        raise ValueError("mode must be 'nep' or 'mocap'")

    if mode == "mocap":
        fields.append(ObsField("object_pos", 3))
        if include_object_rot:
            fields.append(ObsField("object_rot", 4))
        if mocap_dropout_mask:
            # Not shown in Table IV; keep as an opt-in debugging hook.
            fields.append(ObsField("object_valid", 1))

    fields.append(ObsField("skill_label", num_skills))

    history_fields = [
        ObsField("base_ang_vel", 3),
        ObsField("gravity", 3),
        ObsField("dof_pos", dof),
        ObsField("dof_vel", dof),
        ObsField("action", dof),
    ]
    return ObservationConfig(fields=fields, history_fields=history_fields, history=history, normalize=normalize)
