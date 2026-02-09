# MoCap Integration

This module provides a minimal MoCap ingestion + dropout handling scaffold that mirrors training-time
object-observation dropout.

## Files
- `services/xmimic/xmimic/deploy/mocap.py`

## Features
- `MocapFrame`: position/orientation + mask
- `MocapDropout`: stochastic dropout with optional hold-last behavior
- `transform_to_robot_frame`: placeholder for rigid transforms into robot base frame

## Next steps
- Replace placeholder transform with proper quaternion math.
- Bind to a real MoCap client (OptiTrack / Vicon / PhaseSpace).
- Ensure dropout masks match `build_mocap_observation` usage.
