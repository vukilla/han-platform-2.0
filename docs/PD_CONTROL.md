# PD Control Interface

Humanoid Network requires action semantics parity between simulation and deployment. This repo now includes a PD controller module
that converts policy actions into torques, matching a 100 Hz policy → 1000 Hz PD loop.

## Code
- `services/xmimic/xmimic/controllers/pd.py`
  - `PDController` + `PDGains` + `ActionSemantics`

## Action semantics
- `mode="target"`: action is joint target position.
- `mode="delta"`: action is delta from current position (`q + action * scale`).

## Isaac Gym integration (stub)
`CargoPickupIsaacEnv` now instantiates a PD controller and applies joint efforts each step when available.
This is a lightweight bridge; full per-joint gains and real robot parity should be calibrated once the
URDF and DOF ordering are finalized.

## Next steps
- Bind real URDF joint order + per-joint gains.
- Mirror the same PD gains in deployment runtime.
- Add safety limits + watchdog (see deployment/safety issues).
