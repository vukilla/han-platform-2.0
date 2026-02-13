# What To Test (Golden Path)

This repo is currently wired so you can validate the full web-app + API + storage + job system end-to-end, even before real GVHMR/Isaac Lab training is online.

## 0) Start The Mac Control Plane

From your Mac:

```bash
cd /Users/robertvukosa/Downloads/Python/han-platform
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
/Users/robertvukosa/Downloads/Python/han-platform/scripts/smoke_local_e2e.sh

# Full golden path including GPU-queue XMimic (requires a GPU worker)
/Users/robertvukosa/Downloads/Python/han-platform/scripts/smoke_e2e_with_gpu.sh

# REAL GPU golden path:
# - XGen on GPU queue with GVHMR pose extraction (video -> SMPL-X NPZ, uploaded to MinIO)
# - XMimic on GPU queue with Isaac Lab teacher PPO (exports .pt checkpoint, uploaded to MinIO)
#
# Requires:
# - Either Pegasus or Windows GPU worker
# - GVHMR bootstrapped on active worker (see docs/GVHMR.md)
/Users/robertvukosa/Downloads/Python/han-platform/scripts/smoke_e2e_with_gpu_real.sh
```

## 0.5) Web UI: Motion Recovery Only (Fastest)

This is the simplest “phone video -> 3D motion” UX loop.

1. Upload the licensed SMPL-X model file (`SMPLX_NEUTRAL.npz`) so GVHMR can run end-to-end:
1. Open `http://localhost:3000/studio`
1. If it shows `SMPL-X model missing`, upload the `.npz` file (see `docs/GVHMR.md` for where to download it).
1. Open `http://localhost:3000/studio`
1. Choose an `.mp4` (10-20s recommended).
1. Leave defaults:
1. Static camera: on
1. Quick preview: on (trims to first 12s to keep demos responsive)
1. Click `Run motion recovery`
1. You will be redirected to `/jobs/<id>` where you can watch stage progress and see a side-by-side preview.

Notes:
- `http://localhost:3000/gvhmr` is still available as a troubleshooting/setup page.

Fastest “do it for me” launch (GVHMR-only, Pegasus-first with Windows fallback):

```bash
PEGASUS_HOST=<pegasus_host> WINDOWS_GPU_IP=<windows_ip> /Users/robertvukosa/Downloads/Python/han-platform/scripts/mac/run_gvhmr_studio_ssh.sh
```

This will:
1. Start the Mac control-plane (docker compose)
1. Start/restart a pose worker over SSH (queue: `pose`), preferring Pegasus and falling back to Windows
1. Run the motion-recovery smoke test (upload -> GVHMR -> preview)
1. Print the `/jobs/<id>` URL to open in the browser

Fastest REAL “do it for me” launch (GVHMR + Isaac Lab PPO, Pegasus-first with Windows fallback):

1. On Mac:

```bash
# Optional: pass a video path. If omitted, the script uses:
# - /Users/robertvukosa/Desktop/delivery-man-...mp4 if it exists, else
# - assets/sample_videos/cargo_pickup_01.mp4
PEGASUS_HOST=<pegasus_host> WINDOWS_GPU_IP=<windows_ip> \
 /Users/robertvukosa/Downloads/Python/han-platform/scripts/mac/run_full_e2e_real_ssh.sh [optional_video_path]
```

2. If you are testing only Windows, omit `PEGASUS_HOST` and keep `WINDOWS_GPU_IP`:
   - `WINDOWS_GPU_IP=<windows_ip> /Users/robertvukosa/Downloads/Python/han-platform/scripts/mac/run_full_e2e_real_ssh.sh`

### Even easier: Pegasus + Windows preference via SSH launcher

If you have SSH access from your Mac to either endpoint, you can run a single command end-to-end:

```bash
WINDOWS_GPU_IP=<windows_ip> /Users/robertvukosa/Downloads/Python/han-platform/scripts/mac/run_full_e2e_real_ssh.sh
```

This will:
- start the Mac control-plane (docker compose)
- start/restart a GPU worker over SSH (Pegasus first when available)
- run the REAL smoke (GVHMR pose + Isaac Lab PPO checkpoint)

## 1) Web UI: Upload -> XGen -> Dataset

1. Open `http://localhost:3000/auth`
1. Enter any email, click `Send magic link` (it stores a local JWT)
1. Open `http://localhost:3000/demos/new`
1. Pick any `.mp4` (short is fine)
1. Leave defaults (advanced settings are optional)
1. Click `Generate dataset and policy`
1. Click the `View job <id>` link
1. Wait until the badge shows `COMPLETED`
1. Click `Open dataset` on the job page
1. On the dataset page:
1. Confirm clips are listed
1. Confirm the preview player loads (currently points at the original uploaded demo video)
1. Click `Download dataset` and confirm you get a `dataset.zip`

What “success” means here:

1. Video upload succeeded and the object exists in MinIO.
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
