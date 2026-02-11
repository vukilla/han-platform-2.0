# PRD - Humanoid Network

## Problem
Physical interaction data is scarce, expensive, and fragmented. Existing datasets are difficult to unify across robots and tasks, which blocks scalable policy training.

## Approach
Use HumanX-style compilation to turn monocular human videos into standardized, physically plausible humanoid–object interaction trajectories (XGen). Augment those trajectories to expand coverage and use a unified imitation reward in a teacher–student PPO pipeline (XMimic) to avoid per-task reward engineering.

## MVP Scope
- Video upload → annotation → XGen dataset → preview → optional XMimic training → eval report → export
- Manual assist for object initialization and (optionally) SMPL motion input
- Single MVP task: cargo pickup
- Points-based incentives (no tokenization yet)

## Out of Scope (MVP)
- Fully automated pose/object estimation
- Multi-robot deployment
- On-chain staking/slashing
- Teleop capture and simulator gameplay ingestion

## Success Criteria
- One end-to-end demo completes in under 30 minutes on local stack
- Dataset includes >= 10 clips with previews and contact graphs
- XMimic produces a checkpoint with SR/GSR metrics (even if low)

## Key Risks
- Pose estimation quality and retargeting fidelity
- Simulation instability during non-contact synthesis
- Poor generalization without sufficient augmentation

## Acceptance Criteria
A new engineer can implement the system without guessing the architecture, data model, or pipeline stages.
