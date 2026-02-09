# PhysHOI Dataset Ingest

Converts PhysHOI motion `.pt` files into XGen clip NPZ format.

Module: `services/xgen/xgen/ingest/physhoi.py`

## Usage
```python
from xgen.ingest import convert_physhoi_motion_to_clip

convert_physhoi_motion_to_clip(
    "external/physhoi/physhoi/data/motions/BallPlay/toss.pt",
    "outputs/physhoi/toss_clip.npz",
    metadata={"source": "physhoi", "task": "toss"},
)
```
