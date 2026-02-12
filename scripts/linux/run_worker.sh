#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
API_DIR="$ROOT_DIR/apps/api"

if [[ -n "${HAN_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$HAN_PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
QUEUES="${HAN_WORKER_QUEUES:-pose}"
POOL="${HAN_WORKER_POOL:-solo}"
CONCURRENCY="${HAN_WORKER_CONCURRENCY:-1}"

if [[ -z "${HAN_WORKER_ROLE:-}" ]]; then
  role="cpu"
  lower_queues="$(echo "$QUEUES" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower_queues" == *"gpu"* ]]; then
    role="gpu"
  elif [[ "$lower_queues" == *"pose"* ]]; then
    role="pose"
  fi
  export HAN_WORKER_ROLE="$role"
fi

required_env=(
  REDIS_URL
  DATABASE_URL
  S3_ENDPOINT
  S3_ACCESS_KEY
  S3_SECRET_KEY
  S3_BUCKET
)

for key in "${required_env[@]}"; do
  if [[ -z "${!key:-}" ]]; then
    echo "[ERROR] Missing required env var: ${key}" >&2
    exit 1
  fi
done

if [[ ! -d "$API_DIR" ]]; then
  echo "[ERROR] Missing API directory: $API_DIR" >&2
  exit 1
fi

if [[ "${INSTALL_REQS:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$API_DIR/requirements.worker.txt"
fi

echo "== han-platform Worker (Linux) =="
echo "Repo root:    $ROOT_DIR"
echo "Python:       $PYTHON_BIN"
echo "Queues:       $QUEUES"
echo "Role:         ${HAN_WORKER_ROLE}"
echo "Pool:         $POOL"
echo "Concurrency:  $CONCURRENCY"
echo ""

cd "$API_DIR"
exec "$PYTHON_BIN" -m celery -A app.worker.celery_app worker -l info -Q "$QUEUES" -P "$POOL" -c "$CONCURRENCY"
