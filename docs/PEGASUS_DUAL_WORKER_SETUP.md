# Pegasus + Windows Dual Worker Setup

This runbook adds a second worker host (Pegasus) so you can run motion recovery while your home gaming PC is off, while still keeping the Windows path available.

## Goal
- Keep current Windows worker path as fallback.
- Add a remote Linux worker path for `pose` queue jobs.
- Avoid breaking current local flow.

## Current limitation
- `isaaclab_teacher_ppo` is still Windows-only in `apps/api/app/worker.py`.
- Result: training jobs on the `gpu` queue still need the Windows GPU worker unless that backend is ported.

## Step 1, choose control-plane location
- Recommended: move API + Redis + Postgres + object storage to an always-on host.
- Temporary fallback: keep Mac as control-plane and point Pegasus worker back to Mac LAN IP, this still requires home network reachability.

## Step 2, prepare Pegasus repo and Python
On Pegasus:

```bash
git clone https://github.com/vukilla/han-platform.git ~/han-platform
cd ~/han-platform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r apps/api/requirements.worker.txt
```

If your worker will run GVHMR pose recovery, install the extra GVHMR dependencies in the same environment as well.

## Step 3, start Pegasus worker from Mac over SSH
From your Mac repo:

```bash
export PEGASUS_HOST=pegasus.dfki.de
export SSH_USER=rvuko
export SSH_KEY=~/.ssh/pegasus

# Point to your always-on control-plane endpoints.
export REDIS_URL=redis://<CONTROL_PLANE_HOST>:6379/0
export DATABASE_URL=postgresql+psycopg://han:han@<CONTROL_PLANE_HOST>:5432/han
export S3_ENDPOINT=http://<CONTROL_PLANE_HOST>:9000
export S3_ACCESS_KEY=minioadmin
export S3_SECRET_KEY=minioadmin
export S3_BUCKET=humanoid-network-dev

# Pose only by default.
export HAN_WORKER_QUEUES=pose

./scripts/mac/start_pegasus_worker_ssh.sh
```

Tail logs:

```bash
./scripts/mac/tail_pegasus_worker_logs_ssh.sh
```

Stop worker:

```bash
./scripts/mac/stop_pegasus_worker_ssh.sh
```

## Step 4, keep Windows worker as fallback
No change needed for your current Windows path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_gpu_worker.ps1 -MacIp <MAC_LAN_IP>
```

Or from Mac over SSH:

```bash
WINDOWS_GPU_IP=<windows_ip> ./scripts/mac/start_windows_gpu_worker_ssh.sh
```

## Step 5, validate failover
Run these checks before cutover:

1. Pegasus on, Windows off, run a `pose` job.
2. Pegasus off, Windows on, run the same `pose` job.
3. Both on, run multiple jobs and confirm stable queue processing.
4. Submit one training job, confirm it still routes to Windows `gpu` worker.
5. Verify worker status via `http://localhost:8000/ops/workers?timeout=2.0` (or your deployed API URL).

## Step 6, production hardening (recommended)
- Move control-plane services off local laptop.
- Use TLS for API and object storage.
- Replace default MinIO credentials.
- Restrict worker host ingress with firewall rules.
- Run worker as a managed service (systemd or supervisor), instead of ad-hoc nohup.
