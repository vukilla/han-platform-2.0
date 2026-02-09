# Deployment Runtime Loop

A lightweight runtime loop is provided for real-time deployment parity with simulation.

## Files
- `services/xmimic/xmimic/deploy/runtime.py`

## Features
- 100 Hz policy loop with 1000 Hz PD sub-steps.
- Uses `ObservationPipeline` for normalized + history-based observations.
- Uses `PDController` to convert actions → torques with matched semantics.

## Usage (skeleton)
Provide two callables:
- `sensor_fn() -> {"q": ..., "qd": ..., ...}`
- `actuator_fn(torque)`

Then build:
```
config = RuntimeConfig(action_dim=24)
obs_pipeline = ObservationPipeline(default_obs_config())
loop = RuntimeLoop(config, obs_pipeline, sensor_fn, actuator_fn, policy_fn)
loop.run(max_steps=1000)
```

This is a scaffold; swap in the real robot I/O and policy loader when available.
