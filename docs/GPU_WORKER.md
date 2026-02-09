# GPU Worker Notes

## Overview
Run the GPU worker on the RTX 5090 PC as a standalone Python process that consumes Redis jobs and writes artifacts to MinIO.

## Prereqs
- Python 3.11
- Access to the Mac host running Redis + MinIO

## Environment
Set these variables on the Windows PC (PowerShell):

```powershell
$env:REDIS_URL      = "redis://<MAC_LAN_IP>:6379/0"
$env:DATABASE_URL   = "postgresql+psycopg://han:han@<MAC_LAN_IP>:5432/han"
$env:S3_ENDPOINT    = "http://<MAC_LAN_IP>:9000"
$env:S3_ACCESS_KEY  = "minioadmin"
$env:S3_SECRET_KEY  = "minioadmin"
$env:S3_BUCKET      = "humanx-dev"
$env:CELERY_GPU_QUEUE = "gpu"
$env:CELERY_MAX_QUEUE_GPU = "20"
$env:CELERY_PREFETCH_MULTIPLIER = "1"
$env:CELERY_TASK_ACKS_LATE = "true"
$env:CELERY_REJECT_ON_WORKER_LOST = "true"
```

## Run
From `apps/api`:

```powershell
pip install -r requirements.txt
celery -A app.worker.celery_app worker -l info -Q gpu -c 1
```

## Docker (GPU)
```
docker build -f apps/api/Dockerfile.gpu -t han-gpu-worker .
docker run --gpus all --env-file .env han-gpu-worker
```

## Notes
- Ensure macOS firewall allows inbound connections on ports 6379 and 9000.
- GPU-specific dependencies (Isaac Sim / Isaac Lab) should be installed on the PC only.
- Current `run_xmimic_job` task in `apps/api/app/worker.py` is still a placeholder state-machine; it will be wired to Isaac Lab training once the Windows Isaac Lab setup is stable.
