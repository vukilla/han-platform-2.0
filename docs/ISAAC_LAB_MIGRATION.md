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
Given the current constraint (Windows GPU PC without WSL/virtualization), treat **Isaac Lab as the primary path**.

Plan:
1. Bring up Isaac Sim + Isaac Lab on Windows and prove we can run a headless sim + a trivial task.
2. Add an adapter layer in `services/xmimic` that maps Isaac Lab env signals into the HumanX Table IV observation schema.
3. Port the first end-to-end task (Cargo Pickup) into Isaac Lab assets/configs.
4. Keep PhysHOI/Isaac Gym as optional reference material only (Linux-only), not as a dependency for the main platform.
