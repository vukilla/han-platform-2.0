# XMimic Reward (Exact Terms)

The unified reward is implemented as exponential tracking terms plus regularization:

- Body position/rotation/velocity
- Object position/rotation
- Relative body–object motion
- Contact graph imitation
- Action regularization
- AMP (optional discriminator reward hook)

Humanoid Network-style gamma/lambda configuration lives in:

- `services/xmimic/xmimic/rewards/unified.py` (`HumanoidNetworkRewardConfig`, `RewardTerm`, `load_reward_config`)
- `services/xmimic/configs/humanoid_network_reward.yaml` (default values)

Use `load_reward_config` and pass the config into `compute_reward_terms(..., config=...)` to align
reward computation with the Eq.5–12 exponential form. Update the YAML values to match the paper
once the exact coefficients are confirmed.

AMP details live in `docs/AMP.md`. If you populate `obs["amp"]`, the reward code will include it
via `amp_weight`.
