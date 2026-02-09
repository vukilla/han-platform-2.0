# GPU Worker Notes

## Overview
Run the GPU worker on the RTX 5090 PC as a standalone Python process that consumes Redis jobs and writes artifacts to MinIO.

## Prereqs
- Python 3.11
- Access to the Mac host running Redis + MinIO

## Environment
Set these variables on the PC:

```
export REDIS_URL=redis://<MAC_LAN_IP>:6379/0
export DATABASE_URL=postgresql+psycopg://han:han@<MAC_LAN_IP>:5432/han
export S3_ENDPOINT=http://<MAC_LAN_IP>:9000
export S3_ACCESS_KEY=minioadmin
export S3_SECRET_KEY=minioadmin
export S3_BUCKET=humanx-dev
export CELERY_GPU_QUEUE=gpu
export CELERY_MAX_QUEUE_GPU=20
export CELERY_PREFETCH_MULTIPLIER=1
export CELERY_TASK_ACKS_LATE=true
export CELERY_REJECT_ON_WORKER_LOST=true
```

## Run
From `apps/api`:

```
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
- GPU-specific dependencies (Isaac Gym, RL libs) should be installed on the PC only.
