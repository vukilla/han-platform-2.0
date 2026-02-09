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

## Usage
`services/xgen/xgen/pose/gvhmr_pose.py` calls:
```
python tools/demo/demo.py --video <video> --output_root <output_root> -s
```

The wrapper writes:
- `<output_root>/<video_stem>/hmr4d_results.pt` (GVHMR output)
- `<output_dir>/<video_stem>_gvhmr_smplx.npz` (SMPL-X params)
- `<output_dir>/<video_stem>_gvhmr_meta.json`

## Environment
Optional env var:
```
export GVHMR_ROOT=/path/to/GVHMR
```
