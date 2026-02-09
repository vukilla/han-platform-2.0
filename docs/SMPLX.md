# SMPL → SMPL-X Conversion

This repo includes a lightweight SMPL → SMPL‑X conversion utility for PhysHOI compatibility.

## Module
- `services/xgen/xgen/pose/smplx_convert.py`

## What it does
- Loads SMPL params from NPZ (global_orient/body_pose/betas/transl)
- Fits SMPL‑X parameters by matching joints with gradient descent
- Writes a SMPL‑X NPZ with hands/jaw/eyes/expression (zeros by default)

## Usage
```python
from pathlib import Path
from xgen.pose.smplx_convert import convert_smpl_to_smplx

output = convert_smpl_to_smplx(
    smpl_npz=Path("input_smpl.npz"),
    output_path=Path("output_smplx.npz"),
    model_dir=Path("/path/to/body_models"),
    num_iters=15,
    device="cpu",
)
```

## Requirements
- `smplx` package
- SMPL/SMPL‑X model files in `model_dir`

## Resample + smooth
Resample SMPL‑X sequences to a target control rate (e.g., 100 Hz), with optional
root translation + yaw smoothing:
```
PYTHONPATH=services/xgen python services/xgen/scripts/smplx_resample.py \
  --input /path/output_smplx.npz \
  --output /path/output_smplx_100hz.npz \
  --source-fps 30 \
  --target-fps 100 \
  --smooth-window 5
```
