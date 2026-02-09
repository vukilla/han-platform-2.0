# PhysHOI Base Repo

Legacy note:
- PhysHOI uses Isaac Gym (Linux-focused) and is kept here as optional reference material.
- The current supported GPU path for this project is Windows-native Isaac Sim + Isaac Lab.

## Clone
```
git clone https://github.com/wyhuai/PhysHOI external/physhoi
```

If you keep a large asset+checkpoint staging clone, use:
- `external/humanoid-projects/PhysHOI`

## Install
1. Install Isaac Gym from NVIDIA.
2. Install requirements:
```
pip install -r external/physhoi/requirements.txt
```

## Pretrained checkpoints
Download the PhysHOI models from the repo’s Google Drive link and place them under:
```
external/physhoi/physhoi/data/models/<task>/nn/PhysHOI.pth
```

If you downloaded the zip to `external/humanoid-projects/PhysHOI/physhoi_checkpoints.zip`, install into the repo with:
```
./scripts/install_physhoi_checkpoints.sh
```

## Inference (example)
```
python external/physhoi/physhoi/run.py --test --task PhysHOI_BallPlay \
  --num_envs 16 \
  --cfg_env external/physhoi/physhoi/data/cfg/physhoi.yaml \
  --cfg_train external/physhoi/physhoi/data/cfg/train/rlg/physhoi.yaml \
  --motion_file external/physhoi/physhoi/data/motions/BallPlay/toss.pt \
  --checkpoint external/physhoi/physhoi/data/models/toss/nn/PhysHOI.pth
```

Wrapper script (auto-detects `external/humanoid-projects/PhysHOI` when present):
```
./scripts/run_physhoi_inference.sh <motion_file> <checkpoint> [num_envs] [task] [extra args...]
```

Headless example:
```
./scripts/run_physhoi_inference.sh <motion_file> <checkpoint> 16 PhysHOI_BallPlay --headless
```

## SMPL‑X motion conversion
Use the conversion utility to turn SMPL‑X NPZ into PhysHOI motion format:
```
python -c "from pathlib import Path; from xgen.pose.physhoi_motion import smplx_npz_to_physhoi_motion; smplx_npz_to_physhoi_motion(Path('motion_smplx.npz'), Path('motion_physhoi.pt'), Path('/path/to/body_models'))"
```

CLI (preferred):
```
PYTHONPATH=services/xgen python services/xgen/scripts/physhoi_export.py \
  --smplx-npz /path/to/motion_smplx.npz \
  --model-dir /path/to/smplx/models \
  --output /path/to/motion_physhoi.pt
```

Validate an existing PhysHOI motion file:
```
PYTHONPATH=services/xgen python services/xgen/scripts/physhoi_export.py \
  --output /path/to/motion_physhoi.pt \
  --validate-only
```
