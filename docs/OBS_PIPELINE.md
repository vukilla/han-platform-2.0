# Observation Pipeline

This module defines a fixed observation ordering, running mean/variance normalization,
and history buffers to preserve deployment parity.

## Module
- `services/xmimic/xmimic/obs_pipeline.py`

## Key pieces
- `ObservationConfig`: ordered fields + history length
- `RunningNorm`: online mean/variance
- `HistoryBuffer`: stacked recent frames
- `ObservationPipeline`: builds normalized + stacked observation vectors

## Usage
```python
from xmimic.obs_pipeline import ObservationPipeline, default_obs_config

config = default_obs_config()
pipeline = ObservationPipeline(config)

obs = pipeline.build({
    "base_ang_vel": base_ang_vel,
    "projected_gravity": gravity,
    "dof_pos": dof_pos,
    "dof_vel": dof_vel,
    "dof_pos_error": dof_pos_error,
    "prev_action": prev_action,
    "object_pos": obj_pos,
    "object_rot": obj_rot,
    "object_vel": obj_vel,
    "object_ang_vel": obj_ang_vel,
})
```

Adjust field sizes to match your robot (Table IV).
