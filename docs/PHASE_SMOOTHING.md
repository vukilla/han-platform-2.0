# Phase Transition Smoothing

`xgen.interaction.smooth_phase_transitions` blends body/object/velocity series
around contact ↔ non-contact boundaries to avoid impulses.

## Usage
```python
from xgen.interaction import smooth_phase_transitions

smoothed = smooth_phase_transitions(
    phases=phases,
    series={
        "robot_qpos": robot_qpos,
        "robot_qvel": robot_qvel,
        "object_pose": object_pose,
        "object_vel": object_vel,
    },
    window=5,
)
```

The smoothing window is symmetric around the boundary (default 5 frames).
