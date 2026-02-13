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
SOURCE="${HAN_WORKER_SOURCE:-}"

normalize_queues() {
  local queue_values="$1"
  local source="$2"
  local map_source="${source,,}"
  if [[ -z "$queue_values" || "$map_source" != "pegasus" && "$map_source" != "windows" ]]; then
    echo "$queue_values"
    return
  fi

  local mapped=()
  local raw_queue=()
  IFS="," read -r -a raw_queue <<< "$queue_values"
  for item in "${raw_queue[@]}"; do
    item="$(echo "$item" | tr '[:upper:]' '[:lower:]' | sed 's/^\\s*//;s/\\s*$//')"
    if [[ -z "$item" ]]; then
      continue
    fi

    if [[ "$item" == "pose" ]]; then
      mapped+=("pose_${map_source}")
      continue
    fi
    if [[ "$item" == "gpu" ]]; then
      mapped+=("gpu_${map_source}")
      continue
    fi
    mapped+=("$item")
  done
  if [[ ${#mapped[@]} -eq 0 ]]; then
    echo "$queue_values"
    return
  fi
  printf "%s" "$(IFS=","; echo "${mapped[*]}")"
}

QUEUES="$(normalize_queues "$QUEUES" "$SOURCE")"

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
  # GVHMR's official renderer depends on a specific Python/Torch/PyTorch3D combo.
  # The upstream GVHMR requirements ship a PyTorch3D wheel for:
  # - Python 3.10
  # - torch 2.3.0+cu121
  #
  # The Celery worker itself can run on a different Python, but we run the GVHMR demo
  # pipeline in this dedicated env and expose it as `GVHMR_DEMO_PYTHON`.
  local demo_py="${GVHMR_DEMO_PYTHON:-}"
  local demo_prefix="${GVHMR_DEMO_ENV_PREFIX:-}"

  if [[ -n "$demo_py" && -z "$demo_prefix" ]]; then
    demo_prefix="$(dirname "$(dirname "$demo_py")")"
    export GVHMR_DEMO_ENV_PREFIX="$demo_prefix"
  fi

  if [[ -z "$demo_py" ]]; then
    if [[ -z "$demo_prefix" ]]; then
      # Prefer node-local storage to avoid home/NFS quotas. Fall back to /tmp.
      preferred_prefix="/local/${USER}/han-platform/gvhmr_demo_env"
      if mkdir -p "$preferred_prefix" >/dev/null 2>&1; then
        demo_prefix="$preferred_prefix"
      else
        demo_prefix="/tmp/${USER}/han-platform/gvhmr_demo_env"
        mkdir -p "$demo_prefix"
      fi
    fi

    demo_py="$demo_prefix/bin/python"
    export GVHMR_DEMO_PYTHON="$demo_py"
    export GVHMR_DEMO_ENV_PREFIX="$demo_prefix"
  fi

  local marker="${GVHMR_DEMO_ENV_READY_MARKER:-${demo_prefix}/.han_gvhmr_ready}"
  if [[ "${GVHMR_FORCE_SETUP:-0}" == "1" ]]; then
    rm -f "$marker"
  fi

  if [[ ! -f "$marker" ]]; then
    conda_bin="$(command -v conda || true)"
    if [[ -z "$conda_bin" ]]; then
      echo "[ERROR] conda not found in PATH. Cannot set up GVHMR demo environment." >&2
      exit 1
    fi

    # Avoid relying on a potentially-shared, potentially-corrupted conda package cache.
    # Use a node-local cache directory by default (still overridable by the user).
    if [[ -z "${CONDA_PKGS_DIRS:-}" ]]; then
      pkgs_dir="/tmp/${USER}/han-platform/conda_pkgs"
      mkdir -p "$pkgs_dir"
      export CONDA_PKGS_DIRS="$pkgs_dir"
    fi

    # If an old venv exists at the same path, it will conflict with `conda create -p`.
    # Only remove it when explicitly forced.
    if [[ -d "$demo_prefix" && ! -x "$demo_py" ]]; then
      rm -rf "$demo_prefix"
      mkdir -p "$demo_prefix"
    fi

    if [[ ! -x "$demo_py" ]]; then
      echo "[Setup] Creating GVHMR demo conda env (py310) at: $demo_prefix"
      "$conda_bin" create -y -p "$demo_prefix" python=3.10
    fi

    echo "[Setup] Installing GVHMR demo runtime dependencies into: $demo_prefix"
    "$conda_bin" run -p "$demo_prefix" python -m pip install -U pip wheel setuptools

    # Use a platform-owned, minimal requirements set (avoids build-time deps like chumpy).
    "$conda_bin" run -p "$demo_prefix" python -m pip install -r "$ROOT_DIR/scripts/linux/requirements.gvhmr_demo.txt"

    # Sanity check, native rendering requires pytorch3d.renderer. Do not force a CUDA probe here,
    # `conda run` can sometimes interfere with GPU-related env vars on Slurm nodes.
    "$conda_bin" run -p "$demo_prefix" python -c "import pytorch3d.renderer"

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
