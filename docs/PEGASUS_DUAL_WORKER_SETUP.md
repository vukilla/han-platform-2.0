# Pegasus + Windows Dual Worker Setup

This setup makes Pegasus the primary runtime path (private control-plane + pose worker), while keeping Windows available as fallback.

## Goal
- Primary: Pegasus private control-plane and Pegasus pose worker.
- Fallback 1: Mac control-plane when Pegasus control-plane is unavailable.
- Fallback 2: Windows worker for Windows-only workloads.

## Current limitation
- `isaaclab_teacher_ppo` is still Windows-only in `apps/api/app/worker.py`.
- Training jobs on the `gpu` queue still require the Windows GPU worker unless that backend is ported.

## Step 1, bootstrap Pegasus control-plane dependencies (one-time)
From your Mac repo:

```bash
export PEGASUS_HOST=<pegasus_ssh_host_or_alias>
export SSH_KEY=~/.ssh/<your_pegasus_key>
./scripts/mac/bootstrap_pegasus_control_plane_ssh.sh
```

This installs a user-space conda environment on Pegasus with:
- `postgresql`
- `redis-server`
- `minio-server`
- Python API dependencies

## Step 2, start Pegasus private control-plane
From your Mac repo:

```bash
export PEGASUS_HOST=<pegasus_ssh_host_or_alias>
export SSH_KEY=~/.ssh/<your_pegasus_key>
export SLURM_PARTITION=batch
export SLURM_TIME=12:00:00
./scripts/mac/start_pegasus_control_plane_ssh.sh
```

Status:

```bash
./scripts/mac/status_pegasus_control_plane_ssh.sh
```

Tail logs:

```bash
./scripts/mac/tail_pegasus_control_plane_logs_ssh.sh
```

Stop:

```bash
./scripts/mac/stop_pegasus_control_plane_ssh.sh
```

## Step 3, start Pegasus pose worker with automatic control-plane fallback
From your Mac repo:

```bash
export PEGASUS_HOST=<pegasus_ssh_host_or_alias>
export SSH_KEY=~/.ssh/<your_pegasus_key>
export HAN_WORKER_QUEUES=pose
export PEGASUS_LAUNCH_MODE=slurm
export SLURM_PARTITION=RTXA6000
export SLURM_TIME=12:00:00

# auto: use Pegasus control-plane endpoints if available, otherwise fallback to Mac LAN.
export CONTROL_PLANE_MODE=auto

./scripts/mac/start_pegasus_worker_ssh.sh
```

Mode behavior:
- `CONTROL_PLANE_MODE=pegasus`: requires Pegasus control-plane endpoints file, hard-fails if missing.
- `CONTROL_PLANE_MODE=mac`: always use Mac LAN endpoints.
- `CONTROL_PLANE_MODE=auto`: Pegasus first, then Mac fallback.

Worker logs:

```bash
./scripts/mac/tail_pegasus_worker_logs_ssh.sh
```

Stop worker:

```bash
./scripts/mac/stop_pegasus_worker_ssh.sh
```

## Step 4, keep Windows worker fallback
No change needed for your current Windows path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_gpu_worker.ps1 -MacIp <MAC_LAN_IP>
```

Or from Mac over SSH:

```bash
WINDOWS_GPU_IP=<windows_ip> ./scripts/mac/start_windows_gpu_worker_ssh.sh
```

## Step 5, validate failover
1. Pegasus control-plane + Pegasus pose worker on, Windows off:
   run a `pose` job.
2. Stop Pegasus pose worker, start Windows pose worker:
   run the same `pose` job.
3. Stop Pegasus control-plane and re-run Pegasus worker with `CONTROL_PLANE_MODE=auto`:
   verify Mac fallback is used.
4. Submit a training job:
   verify it still routes to Windows `gpu` worker.

## Step 6, hardening
- Replace default `minioadmin` credentials.
- Keep Pegasus control-plane private, no public Redis/Postgres ports.
- Use short-lived SSH sessions only for control operations.
