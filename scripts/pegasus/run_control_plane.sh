#!/usr/bin/env bash
set -euo pipefail

# Run the private control-plane stack on a Pegasus compute node:
# - Postgres
# - Redis
# - MinIO
# - FastAPI API
# - CPU Celery worker
#
# This script is intended to be launched inside a Slurm batch allocation.

ROOT_DIR="${HAN_CP_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STATE_ROOT="${HAN_CP_STATE_ROOT:-$ROOT_DIR/tmp/pegasus_control_plane}"
DATA_DIR="${HAN_CP_DATA_DIR:-$STATE_ROOT/data}"
LOG_DIR="${HAN_CP_LOG_DIR:-$STATE_ROOT/logs}"
RUN_DIR="${HAN_CP_RUN_DIR:-$STATE_ROOT/run}"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$RUN_DIR"
rm -f "$STATE_ROOT/endpoints.env"

PGDATA="${HAN_CP_PGDATA:-$DATA_DIR/postgres}"
REDIS_DIR="${HAN_CP_REDIS_DIR:-$DATA_DIR/redis}"
MINIO_DIR="${HAN_CP_MINIO_DIR:-$DATA_DIR/minio}"
mkdir -p "$REDIS_DIR" "$MINIO_DIR"

POSTGRES_PORT="${HAN_CP_POSTGRES_PORT:-15432}"
REDIS_PORT="${HAN_CP_REDIS_PORT:-16379}"
MINIO_PORT="${HAN_CP_MINIO_PORT:-19000}"
MINIO_CONSOLE_PORT="${HAN_CP_MINIO_CONSOLE_PORT:-19001}"
API_PORT="${HAN_CP_API_PORT:-18000}"

MINIO_ACCESS_KEY="${HAN_CP_MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${HAN_CP_MINIO_SECRET_KEY:-minioadmin}"
S3_BUCKET="${HAN_CP_S3_BUCKET:-humanoid-network-dev}"
S3_REGION="${HAN_CP_S3_REGION:-us-east-1}"
S3_SECURE="${HAN_CP_S3_SECURE:-false}"
PG_SUPERUSER="${HAN_CP_PG_SUPERUSER:-$(id -un)}"

CP_HOST="${HAN_CP_HOST_OVERRIDE:-$(hostname -f 2>/dev/null || hostname)}"

required_bins=(initdb pg_ctl psql pg_isready redis-server minio python celery uvicorn alembic)
for b in "${required_bins[@]}"; do
  if ! command -v "$b" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command in PATH: $b" >&2
    exit 1
  fi
done

if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
  initdb -D "$PGDATA" >"$LOG_DIR/postgres_initdb.log" 2>&1
fi

if ! grep -Eq "^[[:space:]]*listen_addresses[[:space:]]*=" "$PGDATA/postgresql.conf"; then
  echo "listen_addresses = '*'" >>"$PGDATA/postgresql.conf"
else
  sed -i.bak "s|^[[:space:]]*listen_addresses[[:space:]]*=.*|listen_addresses = '*'|g" "$PGDATA/postgresql.conf"
fi

if ! grep -Eq "^[[:space:]]*port[[:space:]]*=" "$PGDATA/postgresql.conf"; then
  echo "port = $POSTGRES_PORT" >>"$PGDATA/postgresql.conf"
else
  sed -i.bak "s|^[[:space:]]*port[[:space:]]*=.*|port = $POSTGRES_PORT|g" "$PGDATA/postgresql.conf"
fi

if ! grep -q "host all all 0.0.0.0/0 md5" "$PGDATA/pg_hba.conf"; then
  echo "host all all 0.0.0.0/0 md5" >>"$PGDATA/pg_hba.conf"
fi
if ! grep -q "host all all ::/0 md5" "$PGDATA/pg_hba.conf"; then
  echo "host all all ::/0 md5" >>"$PGDATA/pg_hba.conf"
fi

cleanup() {
  set +e
  if [[ -n "${CPU_WORKER_PID:-}" ]] && kill -0 "$CPU_WORKER_PID" >/dev/null 2>&1; then
    kill "$CPU_WORKER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${MINIO_PID:-}" ]] && kill -0 "$MINIO_PID" >/dev/null 2>&1; then
    kill "$MINIO_PID" >/dev/null 2>&1 || true
  fi
  if command -v redis-cli >/dev/null 2>&1; then
    redis-cli -h 127.0.0.1 -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true
  fi
  pg_ctl -D "$PGDATA" -m fast stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

pg_ctl -D "$PGDATA" -l "$LOG_DIR/postgres.log" -o "-p $POSTGRES_PORT -h 0.0.0.0" start >/dev/null

for _ in {1..60}; do
  if pg_isready -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$PG_SUPERUSER" -d postgres >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! pg_isready -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$PG_SUPERUSER" -d postgres >/dev/null 2>&1; then
  echo "[ERROR] Postgres failed to become ready." >&2
  exit 1
fi

if ! psql -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$PG_SUPERUSER" -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='han'" | grep -q 1; then
  psql -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$PG_SUPERUSER" -d postgres -c "CREATE ROLE han LOGIN PASSWORD 'han';" >/dev/null
else
  psql -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$PG_SUPERUSER" -d postgres -c "ALTER ROLE han WITH PASSWORD 'han';" >/dev/null
fi

if ! psql -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$PG_SUPERUSER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='han'" | grep -q 1; then
  psql -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$PG_SUPERUSER" -d postgres -c "CREATE DATABASE han OWNER han;" >/dev/null
fi

redis-server \
  --bind 0.0.0.0 \
  --protected-mode no \
  --port "$REDIS_PORT" \
  --appendonly yes \
  --dir "$REDIS_DIR" \
  --daemonize yes \
  --logfile "$LOG_DIR/redis.log"

export MINIO_ROOT_USER="$MINIO_ACCESS_KEY"
export MINIO_ROOT_PASSWORD="$MINIO_SECRET_KEY"
minio server "$MINIO_DIR" --address ":$MINIO_PORT" --console-address ":$MINIO_CONSOLE_PORT" >"$LOG_DIR/minio.log" 2>&1 &
MINIO_PID=$!

for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:${MINIO_PORT}/minio/health/live" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${MINIO_PORT}/minio/health/live" >/dev/null 2>&1; then
  echo "[ERROR] MinIO failed to become ready." >&2
  exit 1
fi

DATABASE_URL="postgresql+psycopg://han:han@127.0.0.1:${POSTGRES_PORT}/han"
REDIS_URL="redis://127.0.0.1:${REDIS_PORT}/0"
S3_ENDPOINT="http://127.0.0.1:${MINIO_PORT}"
S3_PUBLIC_ENDPOINT="http://${CP_HOST}:${MINIO_PORT}"
API_URL="http://${CP_HOST}:${API_PORT}"

export DATABASE_URL
export REDIS_URL
export S3_ENDPOINT
export S3_PUBLIC_ENDPOINT
export S3_ACCESS_KEY="$MINIO_ACCESS_KEY"
export S3_SECRET_KEY="$MINIO_SECRET_KEY"
export S3_BUCKET
export S3_REGION
export S3_SECURE

export CELERY_CPU_QUEUE="${CELERY_CPU_QUEUE:-cpu}"
export CELERY_GPU_QUEUE="${CELERY_GPU_QUEUE:-gpu}"
export CELERY_POSE_QUEUE="${CELERY_POSE_QUEUE:-pose}"
export CELERY_MAX_QUEUE_CPU="${CELERY_MAX_QUEUE_CPU:-100}"
export CELERY_MAX_QUEUE_GPU="${CELERY_MAX_QUEUE_GPU:-20}"
export CELERY_MAX_QUEUE_POSE="${CELERY_MAX_QUEUE_POSE:-50}"
export PYTHONPATH="$ROOT_DIR/apps/api${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT_DIR/apps/api"
alembic upgrade head >"$LOG_DIR/alembic.log" 2>&1

uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" >"$LOG_DIR/api.log" 2>&1 &
API_PID=$!

for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
  echo "[ERROR] API failed to become ready." >&2
  exit 1
fi

celery -A app.worker.celery_app worker -l info -Q cpu -P solo -c 1 >"$LOG_DIR/cpu_worker.log" 2>&1 &
CPU_WORKER_PID=$!

cat >"$STATE_ROOT/endpoints.env" <<EOF
PEGASUS_CONTROL_PLANE_HOST=${CP_HOST}
DATABASE_URL=postgresql+psycopg://han:han@${CP_HOST}:${POSTGRES_PORT}/han
REDIS_URL=redis://${CP_HOST}:${REDIS_PORT}/0
S3_ENDPOINT=http://${CP_HOST}:${MINIO_PORT}
S3_PUBLIC_ENDPOINT=http://${CP_HOST}:${MINIO_PORT}
S3_ACCESS_KEY=${MINIO_ACCESS_KEY}
S3_SECRET_KEY=${MINIO_SECRET_KEY}
S3_BUCKET=${S3_BUCKET}
S3_REGION=${S3_REGION}
S3_SECURE=${S3_SECURE}
API_URL=${API_URL}
POSTGRES_PORT=${POSTGRES_PORT}
REDIS_PORT=${REDIS_PORT}
MINIO_PORT=${MINIO_PORT}
MINIO_CONSOLE_PORT=${MINIO_CONSOLE_PORT}
API_PORT=${API_PORT}
EOF

echo "Pegasus control-plane started on host=${CP_HOST}" | tee "$LOG_DIR/status.log"
echo "API: ${API_URL}" | tee -a "$LOG_DIR/status.log"
echo "Redis: redis://${CP_HOST}:${REDIS_PORT}/0" | tee -a "$LOG_DIR/status.log"
echo "DB: postgresql+psycopg://han:han@${CP_HOST}:${POSTGRES_PORT}/han" | tee -a "$LOG_DIR/status.log"

while true; do
  sleep 5
  if ! kill -0 "$API_PID" >/dev/null 2>&1; then
    echo "[ERROR] API process exited." >&2
    exit 1
  fi
  if ! kill -0 "$CPU_WORKER_PID" >/dev/null 2>&1; then
    echo "[ERROR] CPU worker process exited." >&2
    exit 1
  fi
  if ! kill -0 "$MINIO_PID" >/dev/null 2>&1; then
    echo "[ERROR] MinIO process exited." >&2
    exit 1
  fi
done
