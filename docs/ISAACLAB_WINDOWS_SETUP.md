# Isaac Sim + Isaac Lab (Windows) Setup

This is the Windows-native setup for the GPU PC. No WSL2, no virtualization.

## What You Get

- Windows PC runs GPU compute:
  - Isaac Sim (installed once)
  - Isaac Lab (cloned under `external/`, not committed)
  - A GPU worker process that can talk to the Mac control-plane over LAN

- MacBook runs the control-plane:
  - Postgres + Redis + MinIO + API + CPU worker + Web via Docker Compose

## MacBook: Start the Control Plane

From `/Users/robertvukosa/Downloads/Python/han-platform-2.0`:
```bash
docker compose -f infra/docker-compose.yml up --build
docker compose -f infra/docker-compose.yml exec api alembic upgrade head
```

Verify:
- Web: `http://localhost:3000`
- API: `http://localhost:8000/health`

## Windows PC: One-Time Installs (GUI)

1. Install latest NVIDIA driver
2. Install Git for Windows
3. Install VS Code
4. Install Miniconda (optional; only needed if you want a separate conda env for RL frameworks)

Optional but recommended:
- Enable Developer Mode so directory symlinks work:
  - Settings -> Privacy & security -> For developers -> Developer Mode (On)

## Windows PC: Clone Repo

Open PowerShell:
```powershell
mkdir C:\src
cd C:\src
git clone https://github.com/vukilla/han-platform-2.0.git
cd han-platform-2.0
code .
```

## Windows PC: Install Isaac Sim

1. Download Isaac Sim (Windows) from NVIDIA's official Isaac Sim installation docs.
2. Extract to `C:\isaacsim` (no spaces).

Expected:
- `C:\isaacsim\isaac-sim.bat` exists

## Windows PC: Bootstrap Isaac Lab (Automated)

From the repo root:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\bootstrap_isaaclab.ps1 -IsaacSimPath C:\isaacsim
```

If symlink creation fails:
- Re-run PowerShell as Administrator, or ensure Developer Mode is enabled.

## Windows PC: Connect GPU Worker to the Mac

1. Find the MacBook LAN IP address (example: `192.168.2.67`).
2. Start the GPU worker (recommended, only needs the IP):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_gpu_worker.ps1 -MacIp <MAC_LAN_IP>
```

Optional: detached mode (runs in background and writes logs):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start_gpu_worker_detached.ps1 -MacIp <MAC_LAN_IP>
```

Alternative (manual env vars, replace `<MAC_LAN_IP>`):
```powershell
$env:REDIS_URL      = "redis://<MAC_LAN_IP>:6379/0"
$env:DATABASE_URL   = "postgresql+psycopg://han:han@<MAC_LAN_IP>:5432/han"
$env:S3_ENDPOINT    = "http://<MAC_LAN_IP>:9000"
$env:S3_ACCESS_KEY  = "minioadmin"
$env:S3_SECRET_KEY  = "minioadmin"
$env:S3_BUCKET      = "humanx-dev"
```
Then:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_gpu_worker.ps1
```

## Troubleshooting

- `nvidia-smi` missing:
  - NVIDIA driver is not installed or GPU driver is broken.

- `conda` missing:
  - Miniconda not installed or not on PATH. Use the "Anaconda Prompt" that Miniconda installs.

- Symlink (`_isaac_sim`) creation fails:
  - Enable Developer Mode, or open PowerShell as Administrator.

- GPU worker cannot connect to Redis/MinIO:
  - Ensure Mac firewall allows inbound connections to ports `6379`, `9000`, `9001`.
  - Confirm both machines are on the same LAN and `<MAC_LAN_IP>` is reachable.
