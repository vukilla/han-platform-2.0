# Ops Runbooks

## Run locally
1. `cd infra`
2. `docker compose up --build`
3. Run Alembic migrations:
   - `docker compose exec api alembic upgrade head`

## Deploy (future)
- Build API and worker images
- Provision Postgres, Redis, S3-compatible storage
- Configure env vars for storage and DB
- Run migrations before traffic cutover

## Rerun failed jobs
- Set job status to QUEUED
- Re-enqueue Celery task for job ID

## Debug GPU worker
- Check GPU availability with `nvidia-smi`
- Tail worker logs
- Re-run a single job with verbose logging
