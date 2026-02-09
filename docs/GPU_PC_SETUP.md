# GPU PC Setup (RTX 5090) for HumanX Runs

Goal: run the GPU-blocked parts (Isaac Gym / PhysHOI / GVHMR) on the Windows gaming PC, while keeping the laptop as the "control plane" (web/api/db/redis/minio).

This doc assumes you will run Linux on the PC for stability. If you must use WSL2, expect more friction with Isaac Gym.

## 0) Recommended Layout

Keep everything under the repo to avoid path confusion:
- Clone repo on the PC (same repo): `han-platform-2.0`
- External repos + checkpoints are rehydrated under `external/` (not committed to GitHub).

Fastest path:
```
scripts/gpu/bootstrap.sh
```

## 1) OS + Driver

Recommended:
- Ubuntu 22.04 (native install or dual boot).

Minimum checks:
- `nvidia-smi` works.
- Driver is recent enough for RTX 5090 (Blackwell).

## 2) Base Packages

Install:
- `git`
- `python3` (for tooling) and Miniconda (recommended)
- build essentials: `build-essential`, `cmake`, `ninja-build`

## 3) Clone Repo

Clone on PC:
```
git clone <your-repo-url> han-platform-2.0
cd han-platform-2.0
```

Alternative (no GitHub yet): if you copied `han-platform-2.0.bundle` to the PC:
```
git clone han-platform-2.0.bundle han-platform-2.0
cd han-platform-2.0
```

## 4) Python Envs

PhysHOI commonly expects Python 3.9.

Create env (conda recommended):
```
conda create -n physhoi python=3.9 -y
conda activate physhoi
```

Install PhysHOI deps:
```
pip install -r external/physhoi/requirements.txt
```

## 5) Isaac Gym (Preview 4)

Download Isaac Gym Preview 4 from NVIDIA and extract it somewhere on the PC.

Typical location:
- `external/humanoid-projects/isaacgym/`

Then follow PhysHOI README expectations (Isaac Gym python package must be importable).

Sanity check:
```
python -c "from isaacgym import gymapi; print('isaacgym ok')"
```

If Isaac Gym fails due to RTX 5090 compute capability / CUDA compatibility:
- Pivot to Isaac Lab (supported path) and treat PhysHOI as a reference baseline only.

## 6) Checkpoints

If you cloned from GitHub, you will not have checkpoints yet (by design).

Use the bootstrap script to download/place what it can and prompt for the license-gated Isaac Gym archive:
```
scripts/gpu/bootstrap.sh
```

If you already staged checkpoints locally (e.g., from laptop transfer), place them under:
- `external/gvhmr/inputs/checkpoints/...`
- `external/humanoid-projects/PhysHOI/physhoi_checkpoints.zip`

## 7) Install PhysHOI Checkpoints Into Repo Clone

From the PC repo root:
```
./scripts/install_physhoi_checkpoints.sh
```

This populates:
- `external/physhoi/physhoi/data/models/<task>/nn/PhysHOI.pth`

## 8) Run PhysHOI Baseline Inference (Unblocks HAN-142)

Example:
```
./scripts/run_physhoi_inference.sh \
  external/physhoi/physhoi/data/motions/BallPlay/toss.pt \
  external/physhoi/physhoi/data/models/toss/nn/PhysHOI.pth \
  16 PhysHOI_BallPlay --headless
```

Capture:
- full terminal logs
- `nvidia-smi` output during run
- any generated videos/renders

## 9) Run GVHMR Demo (Unblocks HAN-150 Prereq)

After GVHMR deps are installed in its env, run something like:
```
cd external/gvhmr
python tools/demo.py --video_path /path/to/sample.mp4 --ckpt inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
```

## 10) Remote Dev (Optional but Recommended)

Install SSH server on PC and use VS Code Remote-SSH.
