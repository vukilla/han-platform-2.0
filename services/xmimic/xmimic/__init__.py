"""XMimic training pipeline modules.

This package includes optional GPU/simulation/training stacks (torch, Isaac, etc). To keep
CPU-side utilities importable in lightweight environments (docs generation, schema tests),
we avoid importing heavy dependencies at package import time and instead expose top-level
names via lazy module attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # envs/
    "BaseEnv": (".envs", "BaseEnv"),
    "CargoPickupEnv": (".envs", "CargoPickupEnv"),
    "EnvConfig": (".envs", "EnvConfig"),
    # isaac/
    "CargoPickupIsaacEnv": (".isaac", "CargoPickupIsaacEnv"),
    "IsaacEnvConfig": (".isaac", "IsaacEnvConfig"),
    # rewards/
    "RewardWeights": (".rewards", "RewardWeights"),
    "RewardScales": (".rewards", "RewardScales"),
    "RewardTerm": (".rewards", "RewardTerm"),
    "HumanoidNetworkRewardConfig": (".rewards", "HumanoidNetworkRewardConfig"),
    "load_reward_config": (".rewards", "load_reward_config"),
    "compute_reward_terms": (".rewards", "compute_reward_terms"),
    "compute_total_reward": (".rewards", "compute_total_reward"),
    # obs/
    "build_nep_observation": (".obs", "build_nep_observation"),
    "build_mocap_observation": (".obs", "build_mocap_observation"),
    # obs_pipeline/
    "ObservationPipeline": (".obs_pipeline", "ObservationPipeline"),
    "ObservationConfig": (".obs_pipeline", "ObservationConfig"),
    "ObsField": (".obs_pipeline", "ObsField"),
    "default_obs_config": (".obs_pipeline", "default_obs_config"),
    # generalization/
    "GeneralizationConfig": (".generalization", "GeneralizationConfig"),
    "disturbed_initialization": (".generalization", "disturbed_initialization"),
    "interaction_termination": (".generalization", "interaction_termination"),
    "contact_relative_error": (".generalization", "contact_relative_error"),
    "should_apply_external_force": (".generalization", "should_apply_external_force"),
    "sample_external_force": (".generalization", "sample_external_force"),
    "domain_randomization": (".generalization", "domain_randomization"),
    # eval/
    "success_rate": (".eval", "success_rate"),
    "generalization_success_rate": (".eval", "generalization_success_rate"),
    "object_tracking_error": (".eval", "object_tracking_error"),
    "key_body_tracking_error": (".eval", "key_body_tracking_error"),
    "sample_generalization": (".eval", "sample_generalization"),
    "build_eval_report": (".eval", "build_eval_report"),
    "save_eval_report": (".eval", "save_eval_report"),
    # train/ (heavy, torch)
    "TeacherConfig": (".train", "TeacherConfig"),
    "train_teacher": (".train", "train_teacher"),
    "StudentConfig": (".train", "StudentConfig"),
    "distill_student": (".train", "distill_student"),
    "train_student_ppo": (".train", "train_student_ppo"),
    "PPOConfig": (".train", "PPOConfig"),
    "train_ppo": (".train", "train_ppo"),
    "load_config": (".train", "load_config"),
    "save_config": (".train", "save_config"),
    "save_checkpoint": (".train", "save_checkpoint"),
    "load_checkpoint": (".train", "load_checkpoint"),
    "MultiSkillSampler": (".train", "MultiSkillSampler"),
    "SkillSample": (".train", "SkillSample"),
    "ppo_config_from_physhoi": (".train.physhoi_adapter", "ppo_config_from_physhoi"),
    "distill_student_with_physhoi": (".train.physhoi_adapter", "distill_student_with_physhoi"),
    # mlops/ (heavy, mlflow)
    "log_metrics": (".mlops", "log_metrics"),
    # contacts/
    "compute_external_torques": (".contacts", "compute_external_torques"),
    "solve_contact_forces": (".contacts", "solve_contact_forces"),
    # amp/ (heavy, torch)
    "AMPDiscriminator": (".amp", "AMPDiscriminator"),
    "AMPBatch": (".amp", "AMPBatch"),
    "amp_discriminator_loss": (".amp", "amp_discriminator_loss"),
    "amp_reward_from_logits": (".amp", "amp_reward_from_logits"),
    "compute_amp_reward": (".amp", "compute_amp_reward"),
    # controllers/
    "PDController": (".controllers", "PDController"),
    "PDGains": (".controllers", "PDGains"),
    "ActionSemantics": (".controllers", "ActionSemantics"),
    # robot_spec/
    "RobotSpec": (".robot_spec", "RobotSpec"),
    "load_robot_spec": (".robot_spec", "load_robot_spec"),
    # deploy/ (heavy, torch)
    "RuntimeConfig": (".deploy", "RuntimeConfig"),
    "RuntimeLoop": (".deploy", "RuntimeLoop"),
    "TorchPolicy": (".deploy", "TorchPolicy"),
    "MocapFrame": (".deploy", "MocapFrame"),
    "MocapDropout": (".deploy", "MocapDropout"),
    "transform_to_robot_frame": (".deploy", "transform_to_robot_frame"),
    "SafetyConfig": (".deploy", "SafetyConfig"),
    "SafetyStatus": (".deploy", "SafetyStatus"),
    "apply_torque_limits": (".deploy", "apply_torque_limits"),
    "safety_check": (".deploy", "safety_check"),
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

