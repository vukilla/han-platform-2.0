# GVHMR Integration

## Repo setup
Clone the official GVHMR repo under `external/gvhmr`.

GVHMR expects checkpoints under `inputs/checkpoints/` inside that repo. In this project we often stage
large checkpoints under:
- `external/humanoid-projects/GVHMR/inputs/checkpoints`

The wrapper `xgen.pose.gvhmr_pose` will symlink staged checkpoints into `external/gvhmr/inputs/checkpoints`
when the staged directory exists.

You can also set `GVHMR_ROOT` to point to your GVHMR repo clone.

```
git clone https://github.com/zju3dv/GVHMR external/gvhmr
```

Follow GVHMR's `docs/INSTALL.md` to install dependencies and download checkpoints.

## Required SMPL-X Model File (Licensed)

GVHMR expects the SMPL-X body model file:
- `inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`

This file is **licensed** and is not distributed with this repository or with the GVHMR checkpoints.
You must download it from the official SMPL-X model distribution and place it at:
- `external/humanoid-projects/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`

The platform's GVHMR wrapper will symlink/copy staged checkpoints into:
- `external/gvhmr/inputs/checkpoints/...`

If this file is missing, GVHMR pose extraction will fail.

### Windows-native setup (recommended)
This repo supports running GVHMR on the **Windows GPU worker** (Isaac Sim + Isaac Lab Python):

1. Bootstrap Isaac Lab (once):
   - `scripts\\windows\\bootstrap_isaaclab.ps1 -IsaacSimPath C:\\isaacsim`
2. Bootstrap GVHMR (once):
   - `scripts\\windows\\bootstrap_gvhmr.ps1`
   - Optional: download the two direct-link checkpoints:
     - `scripts\\windows\\bootstrap_gvhmr.ps1 -DownloadLightCheckpoints`
   - Optional: best-effort download of the remaining checkpoints from GVHMR's public Google Drive folder:
     - `scripts\\windows\\bootstrap_gvhmr.ps1 -TryDownloadHeavyCheckpoints`
3. Manually place the remaining checkpoints under:
   - `external\\humanoid-projects\\GVHMR\\inputs\\checkpoints\\`

The platform wrapper will symlink/copy staged checkpoints into:
- `external/gvhmr/inputs/checkpoints`

## Usage
In the platform, GVHMR is executed inside the XGen job stage `ESTIMATE_POSE` when you set:
- `params_json.requires_gpu=true`
- `params_json.pose_estimator="gvhmr"`

Internally, the Windows GPU worker runs:
`python external/gvhmr/tools/demo/demo.py --video <video> --output_root <output_root> -s --skip_render`

The wrapper writes:
- `<output_root>/<video_stem>/hmr4d_results.pt` (GVHMR output)
- `<output_dir>/<video_stem>_gvhmr_smplx.npz` (SMPL-X params)
- `<output_dir>/<video_stem>_gvhmr_meta.json`

And uploads to object storage (MinIO/S3):
- `demos/<demo_id>/poses/<xgen_job_id>/gvhmr_smplx.npz`
- `demos/<demo_id>/poses/<xgen_job_id>/gvhmr_meta.json`

### Fallback behavior (platform)
If GVHMR fails (commonly due to missing `SMPLX_NEUTRAL.npz`), the platform can fall back to a
**placeholder SMPL-X NPZ** so the web app can still complete the end-to-end flow for debugging
and UX validation.

To force a hard failure instead (recommended once GVHMR is fully configured), set:
- `params_json.fail_on_pose_error=true`

## Environment
Optional env var:
```
export GVHMR_ROOT=/path/to/GVHMR
```
