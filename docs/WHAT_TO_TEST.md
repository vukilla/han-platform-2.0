# What To Test (Golden Path)

This repo is currently wired so you can validate the full web-app + API + storage + job system end-to-end, even before real GVHMR/Isaac Lab training is online.

## 0) Start The Mac Control Plane

From your Mac:

```bash
cd /Users/robertvukosa/Downloads/Python/han-platform-2.0
docker compose -f infra/docker-compose.yml up -d --build
docker compose -f infra/docker-compose.yml exec api alembic upgrade head
```

Sanity checks:

1. Open `http://localhost:8000/health`
1. Confirm `status` is `ok` and `db/redis/s3` are `true`
1. Open `http://localhost:3000`

Optional one-command smoke tests:

```bash
# CPU-only golden path (upload -> XGen -> dataset)
/Users/robertvukosa/Downloads/Python/han-platform-2.0/scripts/smoke_local_e2e.sh

# Full golden path including GPU-queue XMimic (requires Windows GPU worker running)
/Users/robertvukosa/Downloads/Python/han-platform-2.0/scripts/smoke_e2e_with_gpu.sh

# REAL GPU golden path:
# - XGen on GPU queue with GVHMR pose extraction (video -> SMPL-X NPZ, uploaded to MinIO)
# - XMimic on GPU queue with Isaac Lab teacher PPO (exports .pt checkpoint, uploaded to MinIO)
#
# Requires:
# - Windows GPU worker running
# - GVHMR bootstrapped on Windows (see docs/GVHMR.md)
/Users/robertvukosa/Downloads/Python/han-platform-2.0/scripts/smoke_e2e_with_gpu_real.sh
```

Fastest “do it for me” launch:

1. On Mac:

```bash
/Users/robertvukosa/Downloads/Python/han-platform-2.0/scripts/mac/run_full_e2e.sh
```

2. When it prints your Mac IP, run the printed Windows command on the GPU PC. The Mac script will automatically continue once the GPU worker is detected.

Fastest REAL “do it for me” launch (GVHMR + Isaac Lab PPO):

1. On Mac:

```bash
# Optional: pass a video path. If omitted, the script uses:
# - /Users/robertvukosa/Desktop/delivery-man-...mp4 if it exists, else
# - assets/sample_videos/cargo_pickup_01.mp4
/Users/robertvukosa/Downloads/Python/han-platform-2.0/scripts/mac/run_full_e2e_real.sh
```

2. When it prints your Mac IP, run the printed Windows command on the GPU PC.
   - If GVHMR fails, follow `docs/GVHMR.md` to manually place the remaining checkpoints.

## 1) Web UI: Upload -> XGen -> Dataset

1. Open `http://localhost:3000/auth`
1. Enter any email, click `Send magic link` (it stores a local JWT)
1. Open `http://localhost:3000/demos/new`
1. Pick any `.mp4` (short is fine)
1. Leave defaults (ts/te, anchor type, robot/object)
1. Click `Start XGen`
1. Click the `View job <id>` link
1. Wait until the badge shows `COMPLETED`
1. Click `Open dataset` on the job page
1. On the dataset page:
1. Confirm clips are listed
1. Confirm the preview player loads (currently points at the original uploaded demo video)
1. Click `Download dataset` and confirm you get a `dataset.zip`

What “success” means here:

1. Video upload used a presigned URL and the object exists in MinIO.
1. XGen job progressed through all stages and completed.
1. Dataset row + clip rows exist in Postgres.
1. Clip `.npz` files and `dataset.zip` exist in object storage and download via presigned URL.

## 2) Web UI: XMimic Job -> Policy -> Eval Report (Requires GPU Worker)

This step requires a running GPU worker connected to the same Redis/MinIO as the Mac.

1. Open `http://localhost:3000/training`
1. Pick the dataset from the dropdown
1. Pick mode `NEP` or `MoCap`
1. Click `Start training`
1. Click the `View XMimic job` link and wait for `COMPLETED`
1. Open `http://localhost:3000/policies`
1. Confirm a policy exists
1. Click `Download checkpoint`
1. Click the latest eval run link to open `http://localhost:3000/eval/<id>`

What “success” means here (current state):

1. XMimic stages complete and logs are uploaded.
1. A policy record exists and exposes a downloadable checkpoint artifact:
   - Default: synthetic `checkpoint.json` (platform plumbing)
   - Real mode: Isaac Lab teacher PPO exports `checkpoint.pt` (use `scripts/smoke_e2e_with_gpu_real.sh`)
1. An eval run record exists and renders in the UI:
   - Default: placeholder metrics
   - Real mode: an eval report JSON is stored and linked from the policy (still not SR/GSR/Eo/Eh parity yet)

## 3) Admin: Quality Review -> Points

1. Open `http://localhost:3000/admin/quality`
1. Entity type `demo`
1. Paste the demo id (shown on the XGen job page)
1. Click `Fetch`
1. Click `Approve`
1. Open `http://localhost:3000/rewards` and confirm you have a positive points event

## Debugging Notes

- MinIO console: `http://localhost:9001`
  - Username: `minioadmin`
  - Password: `minioadmin`
- XGen/XMimic logs are exposed as presigned HTTP URLs from the job pages.
