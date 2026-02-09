# XMimic Technical Spec

## Two-stage training
1. Teacher policy per skill pattern with PPO and privileged observations.
2. Student policy distilled from teachers with PPO + BC loss.

## Deployment modes
- NEP: proprioception only, no object observations.
- MoCap: object pose observed with simulated dropout.

## Unified interaction imitation reward
- Body imitation + AMP term
- Object tracking
- Relative body-object motion
- Contact graph imitation
- Regularization

## Generalization-first training
- Disturbed Initialization (DI)
- Interaction Termination (IT) when contact-frame error exceeds threshold
- Domain Randomization (DR) across physical params, perception noise, and external forces

## Environments
MVP task: Cargo pickup
Later tasks: badminton hitting, basketball catch-shot

## Evaluation outputs
- SR: success rate on original trajectories
- GSR: generalization success on augmented distribution
- Eo: object tracking error
- Eh: key-body tracking error
