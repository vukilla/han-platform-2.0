# PhysHOI Training Adapter

This adapter reads PhysHOI training config YAML and maps it into `PPOConfig`.

Module: `services/xmimic/xmimic/train/physhoi_adapter.py`

## Usage
```python
from xmimic.train.physhoi_adapter import ppo_config_from_physhoi, distill_student_with_physhoi

ppo_cfg = ppo_config_from_physhoi("external/physhoi/physhoi/data/cfg/train/rlg/physhoi.yaml")

metrics = distill_student_with_physhoi(
    env,
    obs_dim,
    action_dim,
    physhoi_cfg_path="external/physhoi/physhoi/data/cfg/train/rlg/physhoi.yaml",
    checkpoint_path="output/student_phys_hoI.pth",
)
```
