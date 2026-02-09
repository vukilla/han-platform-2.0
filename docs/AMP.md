# AMP (Adversarial Motion Prior)

This repo includes a minimal AMP discriminator module and reward hook to mirror the HumanX/XMimic use of motion priors.

## What is implemented
- `xmimic/amp.py` provides:
  - `AMPDiscriminator` (simple MLP)
  - `amp_discriminator_loss` (BCE on expert vs policy samples)
  - `compute_amp_reward` (sigmoid of logits → reward signal)
- Reward plumbing:
  - `RewardWeights.amp` and `HumanXRewardConfig.amp_weight`
  - `compute_reward_terms` accepts `obs["amp"]` and adds it to the total reward
  - YAML config supports `amp_weight`

## How to use (minimal)
1. Build AMP features from state observations.
   - Common features: joint positions/velocities, root velocity, key body positions.
2. Train the discriminator each PPO epoch using:
   - `expert` features from reference clips
   - `policy` features from the current rollout
3. Add AMP reward to the environment step info or observation:
   - `obs["amp"] = compute_amp_reward(discriminator, features)`

## Integration note
The current PPO/teacher loops are intentionally lightweight. AMP reward is hooked at the reward-computation layer; wiring the discriminator training loop to your main PPO is the next step once the environment provides AMP features.
