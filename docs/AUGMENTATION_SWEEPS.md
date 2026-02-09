# Augmentation Sweeps

Generates parametric augmentation sweeps by varying object size and velocity noise,
then re-running non-contact simulation.

Module: `services/xgen/xgen/augment/augmentations.py`

## Usage
```python
from xgen.augment import sweep_augmentations

augs = sweep_augmentations(object_pose, contact_indices, fps, scales=[0.8, 1.0, 1.2], velocity_noises=[0.0, 0.05])
```
