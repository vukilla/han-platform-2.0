#!/usr/bin/env bash
set -euo pipefail

# One-command REAL end-to-end run that:
# - starts the Mac control-plane (docker compose)
# - starts a GPU worker via SSH, preferring Pegasus if configured and falling back to Windows
# - runs the REAL smoke (GVHMR + Isaac Lab PPO)
#
# Usage:
#   WINDOWS_GPU_IP=192.168.2.77 ./scripts/mac/run_full_e2e_real_ssh.sh [/path/to/video.mp4]
#   PEGASUS_HOST=user@pegasus-host ./scripts/mac/run_full_e2e_real_ssh.sh [/path/to/video.mp4]
#
# Notes:
# - GVHMR requires licensed SMPL-X model files on the GPU PC (see docs/GVHMR.md).
# - If GVHMR fails, the platform can fall back to a placeholder pose unless you set `fail_on_pose_error=true`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEFAULT_DESKTOP_VIDEO="/Users/robertvukosa/Desktop/delivery-man-carry-a-goods-package-and-send-to-customer-at-home-for-impress-of-good-service.mp4"
DEFAULT_REPO_VIDEO="$ROOT_DIR/assets/sample_videos/cargo_pickup_01.mp4"

VIDEO_PATH="${1:-}"
if [[ -z "$VIDEO_PATH" ]]; then
  if [[ -f "$DEFAULT_DESKTOP_VIDEO" ]]; then
    VIDEO_PATH="$DEFAULT_DESKTOP_VIDEO"
  else
    VIDEO_PATH="$DEFAULT_REPO_VIDEO"
  fi
fi

WINDOWS_IP="${WINDOWS_GPU_IP:-}"
PEGASUS_HOST="${PEGASUS_HOST:-}"
if [[ -z "$WINDOWS_IP" && -z "$PEGASUS_HOST" ]]; then
  echo "Missing worker connection target. Set one of:" >&2
  echo "  PEGASUS_HOST=<host> $ROOT_DIR/scripts/mac/run_full_e2e_real_ssh.sh" >&2
  echo "  WINDOWS_GPU_IP=<windows_ip> $ROOT_DIR/scripts/mac/run_full_e2e_real_ssh.sh" >&2
  exit 1
fi

echo "== Full REAL E2E (Mac + GPU worker over SSH) =="
echo "Repo:    $ROOT_DIR"
echo "Video:   $VIDEO_PATH"
echo "Win GPU: ${WINDOWS_IP:-<not set>}"
echo "Pegasus: ${PEGASUS_HOST:-<not set>}"
if [[ -n "$PEGASUS_HOST" ]]; then
  echo "Worker preference: Pegasus first, Windows fallback."
else
  echo "Worker preference: Windows."
fi
echo ""

"$ROOT_DIR/scripts/mac/control_plane_up.sh"

echo ""
echo "-- Starting GPU worker (preferred: Pegasus, fallback: Windows) --"
start_epoch="$(date +%s)"
windows_started=0
if [[ -n "$PEGASUS_HOST" ]]; then
  HAN_WORKER_SOURCE="pegasus" \
  WORKER_QUEUES="gpu" \
  "$ROOT_DIR/scripts/mac/start_pegasus_worker_ssh.sh" "$PEGASUS_HOST"
elif [[ -n "$WINDOWS_IP" ]]; then
  REAL_WORKER=1 "$ROOT_DIR/scripts/mac/start_windows_gpu_worker_ssh.sh" "$WINDOWS_IP"
  windows_started=1
fi

echo ""
echo "-- Waiting for GPU worker (Celery) --"
deadline=$(( $(date +%s) + 600 ))
while true; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "[ERROR] No GPU Celery worker detected within 10 minutes."
    echo "Check Windows logs: gpu_worker.out.log and gpu_worker.err.log in the repo root."
    exit 1
  fi

  workers_json="$(curl -sS "http://localhost:8000/ops/workers?timeout=2.0" || true)"
  ok_pegasus="$(WORKERS_JSON="$workers_json" python -c 'import json,os; d=json.loads(os.environ.get("WORKERS_JSON") or "{}"); print("1" if d.get("ok") and d.get("has_gpu_queue_pegasus", False) else "0")')"
  ok_windows="$(WORKERS_JSON="$workers_json" python -c 'import json,os; d=json.loads(os.environ.get("WORKERS_JSON") or "{}"); print("1" if d.get("ok") and d.get("has_gpu_queue_windows", False) else "0")')"
  ok_legacy="$(WORKERS_JSON="$workers_json" python -c 'import json,os; d=json.loads(os.environ.get("WORKERS_JSON") or "{}"); print("1" if d.get("ok") and d.get("has_gpu_queue", False) else "0")')"

  if [[ "$ok_pegasus" == "1" ]] || [[ "$ok_windows" == "1" ]]; then
    echo "GPU worker detected."
    break
  fi

  if [[ -n "$WINDOWS_IP" && "$windows_started" -eq 0 ]]; then
    if (( now - start_epoch > 20 )); then
      if [[ -n "$PEGASUS_HOST" ]]; then
        echo "No Pegasus worker detected, starting Windows GPU worker fallback."
      else
        echo "Starting Windows GPU worker."
      fi
      REAL_WORKER=1 "$ROOT_DIR/scripts/mac/start_windows_gpu_worker_ssh.sh" "$WINDOWS_IP"
      windows_started=1
    fi
  fi

  if [[ "$ok_legacy" == "1" ]]; then
    echo "Legacy GPU worker detected, proceeding."
    break
  fi
  sleep 2
done

echo ""
"$ROOT_DIR/scripts/smoke_e2e_with_gpu_real.sh" "$VIDEO_PATH"
