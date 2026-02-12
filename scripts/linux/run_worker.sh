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

setup_gvhmr_demo_python() {
  # Keep heavy GVHMR runtime deps off NFS/home quota by using a node-local venv.
  local demo_py="${GVHMR_DEMO_PYTHON:-}"
  if [[ -z "$demo_py" ]]; then
    local demo_venv="${GVHMR_DEMO_VENV:-/tmp/${USER}/han-platform/gvhmr_demo_venv}"
    if [[ ! -x "$demo_venv/bin/python" ]]; then
      mkdir -p "$(dirname "$demo_venv")"
      python3 -m venv --system-site-packages "$demo_venv"
    fi
    demo_py="$demo_venv/bin/python"
    export GVHMR_DEMO_PYTHON="$demo_py"
  fi

  local marker="${GVHMR_DEMO_ENV_READY_MARKER:-$(dirname "$demo_py")/.han_gvhmr_ready}"
  if [[ "${GVHMR_FORCE_SETUP:-0}" == "1" ]]; then
    rm -f "$marker"
  fi
  if [[ ! -f "$marker" ]]; then
    echo "[Setup] Installing GVHMR demo runtime dependencies into: $(dirname "$demo_py")"
    "$demo_py" -m pip install -U pip wheel setuptools
    "$demo_py" -m pip install \
      "numpy==1.26.4" \
      "opencv-python-headless<4.12" \
      "pytorch-lightning==2.3.0" \
      "hydra-core==1.3.2" \
      hydra-zen rich tqdm einops "timm==0.9.12" yacs \
      ffmpeg-python scikit-image termcolor colorlog "imageio==2.34.1" "av==13.0.0" joblib \
      trimesh smplx "ultralytics==8.2.42"
    touch "$marker"
  fi
}

if [[ "${HAN_WORKER_ROLE}" == "pose" ]]; then
  setup_gvhmr_demo_python

  if [[ -z "${GVHMR_CHECKPOINTS_ROOT:-}" ]]; then
    preferred_root="/local/${USER}/han-platform/gvhmr_checkpoints"
    if mkdir -p "$preferred_root" >/dev/null 2>&1; then
      export GVHMR_CHECKPOINTS_ROOT="$preferred_root"
    else
      export GVHMR_CHECKPOINTS_ROOT="/tmp/${USER}/han-platform/gvhmr_checkpoints"
      mkdir -p "$GVHMR_CHECKPOINTS_ROOT"
    fi
  else
    mkdir -p "$GVHMR_CHECKPOINTS_ROOT"
  fi

  if [[ -z "${GVHMR_ROOT:-}" && -d "$ROOT_DIR/external/gvhmr" ]]; then
    export GVHMR_ROOT="$ROOT_DIR/external/gvhmr"
  fi

  if [[ "${GVHMR_AUTO_FETCH:-1}" == "1" ]]; then
    if command -v curl >/dev/null 2>&1; then
      HF_BASE="${GVHMR_HF_BASE:-https://huggingface.co/camenduru/GVHMR/resolve/main}"
      fetch_ckpt() {
        rel="$1"
        dst="$GVHMR_CHECKPOINTS_ROOT/$rel"
        if [[ -s "$dst" ]]; then
          return
        fi
        mkdir -p "$(dirname "$dst")"
        echo "[GVHMR] Downloading $rel ..."
        curl -fLsS --retry 5 --retry-delay 5 -C - -o "$dst" "$HF_BASE/$rel?download=true"
      }
      fetch_ckpt "gvhmr/gvhmr_siga24_release.ckpt"
      fetch_ckpt "vitpose/vitpose-h-multi-coco.pth"
      fetch_ckpt "hmr2/epoch=10-step=25000.ckpt"
      fetch_ckpt "yolo/yolov8x.pt"
      fetch_ckpt "dpvo/dpvo.pth"
    else
      echo "[WARN] curl not found, cannot auto-fetch GVHMR checkpoints."
    fi
  fi
fi

echo "== han-platform Worker (Linux) =="
echo "Repo root:    $ROOT_DIR"
echo "Python:       $PYTHON_BIN"
echo "Queues:       $QUEUES"
echo "Role:         ${HAN_WORKER_ROLE}"
echo "Pool:         $POOL"
echo "Concurrency:  $CONCURRENCY"
if [[ -n "${GVHMR_ROOT:-}" ]]; then
  echo "GVHMR root:   $GVHMR_ROOT"
fi
if [[ -n "${GVHMR_CHECKPOINTS_ROOT:-}" ]]; then
  echo "GVHMR ckpts:  $GVHMR_CHECKPOINTS_ROOT"
fi
if [[ -n "${GVHMR_DEMO_PYTHON:-}" ]]; then
  echo "GVHMR demo py:$GVHMR_DEMO_PYTHON"
fi
echo ""

cd "$API_DIR"
exec "$PYTHON_BIN" -m celery -A app.worker.celery_app worker -l info -Q "$QUEUES" -P "$POOL" -c "$CONCURRENCY"
