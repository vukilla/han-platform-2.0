# Isaac Lab Migration Assessment

## Current state
PhysHOI uses Isaac Gym (legacy) with SMPL‑X humanoid assets and BallPlay motions.

## Migration Pros
- Active maintenance and support in Isaac Lab (successor to Isaac Gym).
- Better integration with modern robot assets (Unitree G1/H1).
- Improved tooling for sensors, domain randomization, and logging.

## Migration Cons
- Asset conversion effort (MJCF → USD / Isaac Lab config).
- Training scripts and reward code need adaptation.
- Potential changes in sim fidelity and performance characteristics.

## Recommendation
Keep Isaac Gym for immediate replication and benchmarking (PhysHOI parity). Start a parallel Isaac Lab branch for long‑term support once baseline reproduction is stable.
