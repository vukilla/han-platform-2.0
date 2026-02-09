# CPU Pipeline (Prep Before GPU Runs)

This runbook prepares inputs on the laptop, so GPU runs can focus on Isaac Gym/PhysHOI.

## 1) GVHMR → SMPL-X
Assumes:
- GVHMR repo clone: `external/gvhmr`
- GVHMR checkpoints staged under either:
  - `external/gvhmr/inputs/checkpoints/...`
  - `external/humanoid-projects/GVHMR/inputs/checkpoints/...` (preferred staging)

Optional override:
```
export GVHMR_ROOT=/path/to/GVHMR
```

## 2) SMPL-X → PhysHOI motion
Use the CLI added in `services/xgen/scripts/physhoi_export.py` to export `.pt`.

## 3) Motion pack YAML
Use `services/xgen/scripts/physhoi_pack.py` to generate `motions.yaml` for PhysHOI.

## Optional: resample SMPL‑X to control rate
If GVHMR outputs at ~30 Hz, resample to 100 Hz before PhysHOI export:
```
PYTHONPATH=services/xgen python services/xgen/scripts/smplx_resample.py \
  --input /path/<video_stem>_gvhmr_smplx.npz \
  --output /path/<video_stem>_gvhmr_smplx_100hz.npz \
  --source-fps 30 \
  --target-fps 100 \
  --smooth-window 5
```

## Orchestration Script
```
./scripts/run_end_to_end_cpu.sh /path/video.mp4 /path/smplx/models /tmp/humanx_out
```
Outputs:
- `<video_stem>_gvhmr_smplx.npz`
- `motion_physhoi.pt`
- `motions.yaml`

Copy these to the GPU PC and run PhysHOI with `--motion_file /path/to/motions.yaml`.
