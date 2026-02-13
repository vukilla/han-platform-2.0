#!/usr/bin/env bash
set -euo pipefail

# One-command GVHMR-only run that:
# - starts the Mac control-plane (docker compose)
# - starts a pose worker via SSH, preferring Pegasus if configured
# - falls back to the Windows pose worker if Pegasus is unavailable
# - runs the motion-recovery smoke test (upload -> GVHMR -> preview)
#
# Usage:
#   WINDOWS_GPU_IP=192.168.2.77 ./scripts/mac/run_gvhmr_studio_ssh.sh [/path/to/video.mp4]
#
# Notes:
# - GVHMR requires licensed SMPL-X model files. Upload SMPLX_NEUTRAL.npz once via:
#   http://localhost:3000/studio
#
# Optional env vars:
# - SSH_USER / SSH_KEY / WINDOWS_REPO / ISAACSIM_PATH (see scripts/mac/start_windows_gpu_worker_ssh.sh)

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
  echo "  PEGASUS_HOST=<host> $ROOT_DIR/scripts/mac/run_gvhmr_studio_ssh.sh" >&2
  echo "  WINDOWS_GPU_IP=<windows_ip> $ROOT_DIR/scripts/mac/run_gvhmr_studio_ssh.sh" >&2
  exit 1
fi

if [[ -n "$PEGASUS_HOST" ]]; then
  echo "Preferring Pegasus worker, with optional Windows fallback."
else
  echo "Pegasus not configured, using Windows worker."
fi

echo "== GVHMR Studio (Mac + pose worker over SSH) =="
echo "Repo:    $ROOT_DIR"
echo "Video:   $VIDEO_PATH"
echo "Win GPU: ${WINDOWS_IP:-<not set>}"
echo "Pegasus: ${PEGASUS_HOST:-<not set>}"
echo ""

"$ROOT_DIR/scripts/mac/control_plane_up.sh"

echo ""
echo "-- Starting pose worker (preferred: Pegasus, fallback: Windows) --"
start_epoch="$(date +%s)"
windows_started=0
if [[ -n "$PEGASUS_HOST" ]]; then
  HAN_WORKER_SOURCE="pegasus" \
  WORKER_QUEUES="pose" \
  "$ROOT_DIR/scripts/mac/start_pegasus_worker_ssh.sh" "$PEGASUS_HOST"
elif [[ -n "$WINDOWS_IP" ]]; then
  REAL_WORKER=1 SETUP_GVHMR=1 WORKER_SOURCE=windows WORKER_QUEUES=pose "$ROOT_DIR/scripts/mac/start_windows_gpu_worker_ssh.sh" "$WINDOWS_IP"
  windows_started=1
fi

echo ""
echo "-- Waiting for pose worker (Celery) --"
deadline=$(( $(date +%s) + 600 ))
while true; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "[ERROR] No pose Celery worker detected within 10 minutes."
    exit 1
  fi

  workers_json="$(curl -sS "http://localhost:8000/ops/workers?timeout=2.0" || true)"
  ok_pegasus="$(WORKERS_JSON="$workers_json" python -c 'import json,os; d=json.loads(os.environ.get("WORKERS_JSON") or "{}"); print("1" if d.get("ok") and d.get("has_pose_queue_pegasus", False) else "0")')"
  ok_windows="$(WORKERS_JSON="$workers_json" python -c 'import json,os; d=json.loads(os.environ.get("WORKERS_JSON") or "{}"); print("1" if d.get("ok") and d.get("has_pose_queue_windows", False) else "0")')"

  if [[ "$ok_pegasus" == "1" ]] || [[ "$ok_windows" == "1" ]]; then
    echo "Pose worker detected."
    break
  fi

  now=$(date +%s)
  if [[ -n "$WINDOWS_IP" && "$windows_started" -eq 0 ]]; then
    if (( now - start_epoch > 20 )); then
      if [[ -n "$PEGASUS_HOST" ]]; then
        echo "No Pegasus worker detected, starting Windows pose worker fallback."
      else
        echo "Starting Windows pose worker."
      fi
      REAL_WORKER=1 SETUP_GVHMR=1 WORKER_SOURCE=windows WORKER_QUEUES=pose "$ROOT_DIR/scripts/mac/start_windows_gpu_worker_ssh.sh" "$WINDOWS_IP"
      windows_started=1
    fi
  fi

  sleep 2
done

echo ""
"$ROOT_DIR/scripts/smoke_motion_recovery.sh" "$VIDEO_PATH"
