# PhysHOI → Policy Registry Adapter

Registers a PhysHOI checkpoint into the platform policy registry by creating an imported XMimic job and policy, then uploading a manifest.

Module: `apps/api/app/core/physhoi_policy.py`

## Usage
```python
from app.core.physhoi_policy import register_physhoi_policy

policy_id, manifest_uri = register_physhoi_policy(
    db,
    dataset_id="...",
    checkpoint_uri="s3://bucket/path/PhysHOI.pth",
    task="toss",
    cfg_env="external/physhoi/physhoi/data/cfg/physhoi.yaml",
    cfg_train="external/physhoi/physhoi/data/cfg/train/rlg/physhoi.yaml",
    frames_scale=1.0,
)
```
