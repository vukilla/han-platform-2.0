from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml


@dataclass
class RewardWeights:
    body_pos: float = 1.0
    body_rot: float = 1.0
    dof_pos: float = 1.0
    body_vel: float = 0.5
    body_ang_vel: float = 0.5
    dof_vel: float = 0.5
    object_pos: float = 1.0
    object_rot: float = 1.0
    relative_pos: float = 0.5
    relative_rot: float = 0.5
    contact: float = 0.5
    regularization: float = 0.1
    amp: float = 0.5


@dataclass
class RewardScales:
    body_pos: float = 2.0
    body_rot: float = 2.0
    dof_pos: float = 2.0
    body_vel: float = 1.0
    body_ang_vel: float = 1.0
    dof_vel: float = 1.0
    object_pos: float = 2.0
    object_rot: float = 2.0
    relative_pos: float = 1.0
    relative_rot: float = 1.0
    contact: float = 2.0


@dataclass
class RewardTerm:
    gamma: float = 1.0
    lambda_: float = 1.0


@dataclass
class HumanXRewardConfig:
    body_pos: RewardTerm = field(default_factory=RewardTerm)
    body_rot: RewardTerm = field(default_factory=RewardTerm)
    dof_pos: RewardTerm = field(default_factory=RewardTerm)
    body_vel: RewardTerm = field(default_factory=RewardTerm)
    body_ang_vel: RewardTerm = field(default_factory=RewardTerm)
    dof_vel: RewardTerm = field(default_factory=RewardTerm)
    object_pos: RewardTerm = field(default_factory=RewardTerm)
    object_rot: RewardTerm = field(default_factory=RewardTerm)
    relative_pos: RewardTerm = field(default_factory=RewardTerm)
    relative_rot: RewardTerm = field(default_factory=RewardTerm)
    contact: RewardTerm = field(default_factory=RewardTerm)
    # Overall scale for regularization penalties. Individual penalty weights live in reg_terms.
    regularization: float = 0.1
    # Optional per-contact-body weights for Eq (12). If omitted, all ones are used.
    contact_weights: Optional[List[float]] = None
    # Optional per-term weights for r_reg. Terms are only applied when present in `obs`.
    # Suggested keys: torque, action_rate, feet_slip, termination, waist_dof, action_l2.
    reg_terms: Dict[str, float] = field(default_factory=dict)
    amp_weight: float = 0.5


def _term_from_payload(payload: Dict, key: str, default: RewardTerm) -> RewardTerm:
    raw = payload.get(key, {}) or {}
    return RewardTerm(
        gamma=float(raw.get("gamma", default.gamma)),
        lambda_=float(raw.get("lambda", default.lambda_)),
    )


def load_reward_config(path: str | Path) -> HumanXRewardConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    defaults = HumanXRewardConfig()
    # Contact weights can be specified either at top-level or nested under `contact`.
    contact_raw = payload.get("contact", {}) or {}
    contact_weights = payload.get("contact_weights", None)
    if contact_weights is None:
        contact_weights = contact_raw.get("weights", None)

    reg_terms: Dict[str, float] = {}
    raw_reg = payload.get("reg_terms", None)
    if isinstance(raw_reg, dict):
        reg_terms = {str(k): float(v) for k, v in raw_reg.items()}
    elif isinstance(payload.get("regularization", None), dict):
        # Alternative schema:
        # regularization:
        #   weight: 0.1
        #   action_l2: 1.0
        raw = payload.get("regularization", {}) or {}
        reg_terms = {str(k): float(v) for k, v in raw.items() if k != "weight"}

    return HumanXRewardConfig(
        body_pos=_term_from_payload(payload, "body_pos", defaults.body_pos),
        body_rot=_term_from_payload(payload, "body_rot", defaults.body_rot),
        dof_pos=_term_from_payload(payload, "dof_pos", defaults.dof_pos),
        body_vel=_term_from_payload(payload, "body_vel", defaults.body_vel),
        body_ang_vel=_term_from_payload(payload, "body_ang_vel", defaults.body_ang_vel),
        dof_vel=_term_from_payload(payload, "dof_vel", defaults.dof_vel),
        object_pos=_term_from_payload(payload, "object_pos", defaults.object_pos),
        object_rot=_term_from_payload(payload, "object_rot", defaults.object_rot),
        relative_pos=_term_from_payload(payload, "relative_pos", defaults.relative_pos),
        relative_rot=_term_from_payload(payload, "relative_rot", defaults.relative_rot),
        contact=_term_from_payload(payload, "contact", defaults.contact),
        regularization=float(
            (payload.get("regularization", defaults.regularization) or {}).get("weight", defaults.regularization)
            if isinstance(payload.get("regularization", None), dict)
            else payload.get("regularization", defaults.regularization)
        ),
        contact_weights=list(contact_weights) if contact_weights is not None else None,
        reg_terms=reg_terms,
        amp_weight=float(payload.get("amp_weight", defaults.amp_weight)),
    )


def _l2_mean_sq(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _l2_mean_sq_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared L2 norm across the last dimension.

    Paper mapping (HumanX Eq. 10): e_rel_p = ||u_t - u_hat_t||_2^2.
    We average the squared norms across key bodies (and time, if present).
    """
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    sq = np.sum(diff**2, axis=-1)
    return float(np.mean(sq))


def _quat_error(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(a_norm * b_norm, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    return float(np.mean(angle**2))


def _exp_term(err: float, term: RewardTerm) -> float:
    return float(term.gamma * np.exp(-term.lambda_ * err))


def _first_present(obs: Dict[str, np.ndarray], keys: Sequence[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in obs:
            return obs[k]
    return None


def _as_vec3_set(x: np.ndarray) -> np.ndarray:
    """Return shape (..., K, 3) or (K, 3) for key body positions."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"body_pos has {arr.size} values, expected multiple of 3")
        return arr.reshape(-1, 3)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dim == 3, got {arr.shape}")
    return arr


def _relative_vectors(body_pos: np.ndarray, object_pos: np.ndarray) -> np.ndarray:
    """Compute u_t vectors from key bodies to object position (Eq. 10 helper)."""
    body = _as_vec3_set(body_pos)
    obj = np.asarray(object_pos, dtype=np.float32)
    if obj.ndim == 1:
        if obj.size != 3:
            raise ValueError(f"object_pos must be (3,), got {obj.shape}")
        return obj.reshape((1, 3)) - body
    if obj.ndim != 2 or obj.shape[-1] != 3:
        raise ValueError(f"object_pos must be (T, 3) or (3,), got {obj.shape}")
    if body.ndim == 3 and body.shape[0] == obj.shape[0]:
        return obj[:, None, :] - body
    raise ValueError(f"Cannot broadcast body_pos {body.shape} with object_pos {obj.shape}")


def _compute_terms_from_config(
    obs: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    config: HumanXRewardConfig,
) -> Dict[str, float]:
    terms: Dict[str, float] = {}

    if "body_pos" in obs:
        terms["body_pos"] = _exp_term(_l2_mean_sq(obs["body_pos"], targets["body_pos"]), config.body_pos)
    if "body_rot" in obs:
        terms["body_rot"] = _exp_term(_quat_error(obs["body_rot"], targets["body_rot"]), config.body_rot)
    if "dof_pos" in obs:
        terms["dof_pos"] = _exp_term(_l2_mean_sq(obs["dof_pos"], targets["dof_pos"]), config.dof_pos)
    if "body_vel" in obs:
        terms["body_vel"] = _exp_term(_l2_mean_sq(obs["body_vel"], targets["body_vel"]), config.body_vel)
    if "body_ang_vel" in obs:
        terms["body_ang_vel"] = _exp_term(
            _l2_mean_sq(obs["body_ang_vel"], targets["body_ang_vel"]),
            config.body_ang_vel,
        )
    if "dof_vel" in obs:
        terms["dof_vel"] = _exp_term(_l2_mean_sq(obs["dof_vel"], targets["dof_vel"]), config.dof_vel)

    if "object_pos" in obs:
        terms["object_pos"] = _exp_term(_l2_mean_sq(obs["object_pos"], targets["object_pos"]), config.object_pos)
    if "object_rot" in obs:
        terms["object_rot"] = _exp_term(_quat_error(obs["object_rot"], targets["object_rot"]), config.object_rot)

    rel_obs = _first_present(obs, ["relative_pos"])
    if rel_obs is None:
        # Derive from key-body positions and object position (paper Eq. 10).
        if "key_body_pos" in obs and "object_pos" in obs:
            rel_obs = _relative_vectors(obs["key_body_pos"], obs["object_pos"])
        elif "body_pos" in obs and "object_pos" in obs:
            rel_obs = _relative_vectors(obs["body_pos"], obs["object_pos"])

    rel_tgt = _first_present(targets, ["relative_pos"])
    if rel_tgt is None:
        if "key_body_pos" in targets and "object_pos" in targets:
            rel_tgt = _relative_vectors(targets["key_body_pos"], targets["object_pos"])
        elif "body_pos" in targets and "object_pos" in targets:
            rel_tgt = _relative_vectors(targets["body_pos"], targets["object_pos"])

    if rel_obs is not None and rel_tgt is not None:
        # Eq (10): mean squared L2 norm between sets of relative vectors u_t and u_hat_t.
        terms["relative_pos"] = _exp_term(
            _l2_mean_sq_norm(rel_obs, rel_tgt),
            config.relative_pos,
        )
    if "relative_rot" in obs:
        terms["relative_rot"] = _exp_term(_quat_error(obs["relative_rot"], targets["relative_rot"]), config.relative_rot)

    if "contact" in obs:
        # Eq (12): r_cg = exp(-sum_j lambda_cg[j] * |s_cg[j] - s_hat_cg[j]|)
        scg = np.asarray(obs["contact"], dtype=np.float32)
        shat = np.asarray(targets["contact"], dtype=np.float32)
        ecg = np.abs(scg - shat)
        weights: Sequence[float]
        if config.contact_weights is None:
            weights = np.ones((ecg.shape[-1],), dtype=np.float32)
        else:
            weights = np.asarray(config.contact_weights, dtype=np.float32)
            if weights.shape[0] != ecg.shape[-1]:
                raise ValueError(f"contact_weights len {weights.shape[0]} != contact dim {ecg.shape[-1]}")
        if ecg.ndim == 1:
            weighted = float(np.sum(weights * ecg))
        else:
            weighted = float(np.mean(np.sum(ecg * weights[None, :], axis=-1)))
        # Use contact.lambda_ as a global scale when per-body weights are provided.
        terms["contact"] = float(config.contact.gamma * np.exp(-config.contact.lambda_ * weighted))

    # Regularization: apply only terms that are present in `obs`.
    reg_terms = dict(config.reg_terms)
    if not reg_terms:
        reg_terms = {"action_l2": 1.0}

    reg_total = 0.0
    if "action_l2" in reg_terms and "action" in obs:
        reg_total += float(reg_terms["action_l2"]) * float(np.mean(np.asarray(obs["action"], dtype=np.float32) ** 2))
    if "torque" in reg_terms and "torque" in obs:
        reg_total += float(reg_terms["torque"]) * float(np.mean(np.asarray(obs["torque"], dtype=np.float32) ** 2))
    if "action_rate" in reg_terms and "action" in obs and "prev_action" in obs:
        a = np.asarray(obs["action"], dtype=np.float32)
        ap = np.asarray(obs["prev_action"], dtype=np.float32)
        reg_total += float(reg_terms["action_rate"]) * float(np.mean((a - ap) ** 2))
    if "feet_slip" in reg_terms and "feet_slip" in obs:
        reg_total += float(reg_terms["feet_slip"]) * float(np.mean(np.asarray(obs["feet_slip"], dtype=np.float32)))
    if "feet_slippage" in reg_terms:
        slip = _first_present(obs, ["feet_slippage", "feet_slip"])
        if slip is not None:
            reg_total += float(reg_terms["feet_slippage"]) * float(np.mean(np.asarray(slip, dtype=np.float32)))
    if "feet_orientation" in reg_terms:
        orient = _first_present(obs, ["feet_orientation", "feet_orientation_error"])
        if orient is not None:
            reg_total += float(reg_terms["feet_orientation"]) * float(
                np.mean(np.asarray(orient, dtype=np.float32) ** 2)
            )
    if "waist_dof" in reg_terms and "waist_dof" in obs:
        reg_total += float(reg_terms["waist_dof"]) * float(np.mean(np.asarray(obs["waist_dof"], dtype=np.float32) ** 2))
    if "dof_limit" in reg_terms:
        dof_lim = _first_present(obs, ["dof_limit", "dof_limit_violation"])
        if dof_lim is not None:
            reg_total += float(reg_terms["dof_limit"]) * float(np.mean(np.asarray(dof_lim, dtype=np.float32)))
    if "torque_limit" in reg_terms:
        tq_lim = _first_present(obs, ["torque_limit", "torque_limit_violation"])
        if tq_lim is not None:
            reg_total += float(reg_terms["torque_limit"]) * float(np.mean(np.asarray(tq_lim, dtype=np.float32)))
    if "termination" in reg_terms and "termination" in obs:
        reg_total += float(reg_terms["termination"]) * float(np.mean(np.asarray(obs["termination"], dtype=np.float32)))

    terms["regularization"] = -config.regularization * float(reg_total)
    if "amp" in obs:
        terms["amp"] = config.amp_weight * float(np.mean(obs["amp"]))
    return terms


def compute_reward_terms(
    obs: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    weights: RewardWeights | None = None,
    scales: Optional[RewardScales] = None,
    config: HumanXRewardConfig | None = None,
) -> Dict[str, float]:
    if config is not None:
        return _compute_terms_from_config(obs, targets, config)

    scales = scales or RewardScales()
    weights = weights or RewardWeights()
    terms: Dict[str, float] = {}

    if "body_pos" in obs:
        err = _l2_mean_sq(obs["body_pos"], targets["body_pos"])
        terms["body_pos"] = weights.body_pos * np.exp(-scales.body_pos * err)
    if "body_rot" in obs:
        err = _quat_error(obs["body_rot"], targets["body_rot"])
        terms["body_rot"] = weights.body_rot * np.exp(-scales.body_rot * err)
    if "dof_pos" in obs:
        err = _l2_mean_sq(obs["dof_pos"], targets["dof_pos"])
        terms["dof_pos"] = weights.dof_pos * np.exp(-scales.dof_pos * err)
    if "body_vel" in obs:
        err = _l2_mean_sq(obs["body_vel"], targets["body_vel"])
        terms["body_vel"] = weights.body_vel * np.exp(-scales.body_vel * err)
    if "body_ang_vel" in obs:
        err = _l2_mean_sq(obs["body_ang_vel"], targets["body_ang_vel"])
        terms["body_ang_vel"] = weights.body_ang_vel * np.exp(-scales.body_ang_vel * err)
    if "dof_vel" in obs:
        err = _l2_mean_sq(obs["dof_vel"], targets["dof_vel"])
        terms["dof_vel"] = weights.dof_vel * np.exp(-scales.dof_vel * err)

    if "object_pos" in obs:
        err = _l2_mean_sq(obs["object_pos"], targets["object_pos"])
        terms["object_pos"] = weights.object_pos * np.exp(-scales.object_pos * err)
    if "object_rot" in obs:
        err = _quat_error(obs["object_rot"], targets["object_rot"])
        terms["object_rot"] = weights.object_rot * np.exp(-scales.object_rot * err)

    if "relative_pos" in obs:
        err = _l2_mean_sq(obs["relative_pos"], targets["relative_pos"])
        terms["relative_pos"] = weights.relative_pos * np.exp(-scales.relative_pos * err)
    if "relative_rot" in obs:
        err = _quat_error(obs["relative_rot"], targets["relative_rot"])
        terms["relative_rot"] = weights.relative_rot * np.exp(-scales.relative_rot * err)

    if "contact" in obs:
        err = _l2_mean_sq(obs["contact"], targets["contact"])
        terms["contact"] = weights.contact * np.exp(-scales.contact * err)

    reg = float(np.mean(obs.get("action", np.zeros(1)) ** 2))
    terms["regularization"] = -weights.regularization * reg
    if "amp" in obs:
        terms["amp"] = weights.amp * float(np.mean(obs["amp"]))

    if not terms:
        # Backward compatibility with minimal dicts
        body = _l2(obs["body"], targets["body"])
        obj = _l2(obs["object"], targets["object"])
        rel = _l2(obs["relative"], targets["relative"])
        contact = _l2(obs["contact"], targets["contact"])
        terms = {
            "body_pos": weights.body_pos * np.exp(-scales.body_pos * body),
            "object_pos": weights.object_pos * np.exp(-scales.object_pos * obj),
            "relative_pos": weights.relative_pos * np.exp(-scales.relative_pos * rel),
            "contact": weights.contact * np.exp(-scales.contact * contact),
            "regularization": -weights.regularization * reg,
        }
    return terms


def compute_total_reward(terms: Dict[str, float]) -> float:
    return float(sum(terms.values()))
