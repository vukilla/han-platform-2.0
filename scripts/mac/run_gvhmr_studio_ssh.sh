#!/usr/bin/env bash
set -euo pipefail

# One-command GVHMR-only run that:
# - starts the Mac control-plane (docker compose)
# - starts the Windows pose worker via SSH (queue: pose)
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
if [[ -z "$WINDOWS_IP" ]]; then
  echo "Missing WINDOWS_GPU_IP. Example:" >&2
  echo "  WINDOWS_GPU_IP=192.168.2.77 $ROOT_DIR/scripts/mac/run_gvhmr_studio_ssh.sh" >&2
  exit 1
fi

echo "== GVHMR Studio (Mac + Windows pose worker over SSH) =="
echo "Repo:    $ROOT_DIR"
echo "Video:   $VIDEO_PATH"
echo "Win GPU: $WINDOWS_IP"
echo ""

"$ROOT_DIR/scripts/mac/control_plane_up.sh"

echo ""
echo "-- Starting Windows pose worker (GVHMR) --"
# Use the REAL one-click worker so it can bootstrap GVHMR if needed. Consume only the pose queue.
REAL_WORKER=1 SETUP_GVHMR=1 WORKER_QUEUES=pose "$ROOT_DIR/scripts/mac/start_windows_gpu_worker_ssh.sh" "$WINDOWS_IP"

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
  ok="$(WORKERS_JSON="$workers_json" python - <<'PY'
import json,os
try:
  d=json.loads(os.environ.get("WORKERS_JSON") or "{}")
  print("1" if d.get("ok") and d.get("has_pose_queue") is True else "0")
except Exception:
  print("0")
PY
)"
  if [[ "$ok" == "1" ]]; then
    echo "Pose worker detected."
    break
  fi
  sleep 2
done

echo ""
"$ROOT_DIR/scripts/smoke_motion_recovery.sh" "$VIDEO_PATH"

