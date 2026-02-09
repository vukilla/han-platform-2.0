# GPU PC Setup (Windows, Isaac Sim + Isaac Lab)

Goal: run the GPU-heavy parts (simulation, training, inference) on the Windows gaming PC **without WSL/virtualization**, while keeping the MacBook as the "control plane" (web/api/db/redis/minio).

If you can only do Windows on the PC, this is the supported path.

## 0) What Runs Where

MacBook (control plane):
- `docker compose -f infra/docker-compose.yml up --build`
- Postgres, Redis, MinIO, FastAPI API, CPU worker, Next.js web app

Windows gaming PC (GPU compute only):
- Isaac Sim + Isaac Lab (Windows native)
- GPU worker process (Celery `gpu` queue), pointing to the MacBook over LAN

## 1) One-Time Installs on the Windows PC

Install these (GUI installers are fine):
1. NVIDIA driver (latest)
2. Git for Windows
3. VS Code
4. Miniconda (optional; only needed if you want a separate conda env for RL frameworks)

Optional but recommended:
- Enable Windows Developer Mode (for symlinks). Isaac Lab prefers an `_isaac_sim` symlink.

## 2) Clone This Repo on the Windows PC

Choose a short path, for example `C:\\src`:
```powershell
mkdir C:\src
cd C:\src
git clone https://github.com/vukilla/han-platform-2.0.git
cd han-platform-2.0
```

## 3) Install Isaac Sim (Windows Native)

1. Download the Windows build of Isaac Sim from NVIDIA's official docs site.
2. Extract it to: `C:\isaacsim` (no spaces in the path).

You should end up with a file like:
- `C:\isaacsim\isaac-sim.bat`

## 4) Bootstrap Isaac Lab Inside This Repo

From the repo root in PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\bootstrap_isaaclab.ps1 -IsaacSimPath C:\isaacsim
```

This will:
- clone Isaac Lab into `external\isaaclab` (not committed to git)
- create `external\isaaclab\_isaac_sim` pointing at `C:\isaacsim`
- install Isaac Lab extensions using Isaac Sim's bundled Python (no conda required)
- run a quick import smoke test

## 5) Point the GPU Worker at the Mac Control Plane

On the Windows PC, set environment variables (replace `<MAC_LAN_IP>`):
```powershell
$env:REDIS_URL      = "redis://<MAC_LAN_IP>:6379/0"
$env:DATABASE_URL   = "postgresql+psycopg://han:han@<MAC_LAN_IP>:5432/han"
$env:S3_ENDPOINT    = "http://<MAC_LAN_IP>:9000"
$env:S3_ACCESS_KEY  = "minioadmin"
$env:S3_SECRET_KEY  = "minioadmin"
$env:S3_BUCKET      = "humanx-dev"
$env:CELERY_GPU_QUEUE = "gpu"
```

Then run the GPU worker (inside the Isaac Lab conda env):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_gpu_worker.ps1
```

## 6) Legacy (Linux/Isaac Gym/PhysHOI)

The older Isaac Gym/PhysHOI path is still documented under:
- `scripts/gpu/*`

It assumes Linux and is not the recommended approach for this project given the current constraints.
