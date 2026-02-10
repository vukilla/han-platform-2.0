#!/usr/bin/env bash
set -euo pipefail

# Runs control-plane + waits for a GPU worker + runs the REAL E2E smoke:
# - GVHMR pose extraction on the Windows GPU worker (video -> SMPL-X NPZ in MinIO)
# - Isaac Lab PPO training on the Windows GPU worker (checkpoint.pt in MinIO)
#
# Usage:
#   ./scripts/mac/run_full_e2e_real.sh [/path/to/video.mp4]

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

echo "== Full REAL E2E (Mac + Windows GPU worker) =="
echo "Repo:  $ROOT_DIR"
echo "Video: $VIDEO_PATH"
echo ""

"$ROOT_DIR/scripts/mac/control_plane_up.sh"

echo ""
echo "-- Mac LAN IP (Wi-Fi) --"
MAC_IP="$(ipconfig getifaddr en0 || true)"
if [[ -z "$MAC_IP" ]]; then
  MAC_IP="$(ipconfig getifaddr en1 || true)"
fi
if [[ -z "$MAC_IP" ]]; then
  echo "[WARN] Could not determine LAN IP via en0/en1. Manually find it in System Settings -> Network."
else
  echo "$MAC_IP"
  echo ""
  echo "Windows command (REAL worker + GVHMR bootstrap):"
  echo "  powershell -ExecutionPolicy Bypass -File .\\scripts\\windows\\one_click_gpu_worker_real.ps1 -MacIp $MAC_IP -IsaacSimPath C:\\isaacsim -SetupGVHMR -DownloadLightCheckpoints"
  echo ""
  echo "If GVHMR still fails, you likely need to manually place these checkpoints on Windows:"
  echo "  external\\humanoid-projects\\GVHMR\\inputs\\checkpoints\\dpvo\\dpvo.pth"
  echo "  external\\humanoid-projects\\GVHMR\\inputs\\checkpoints\\vitpose\\vitpose-h-multi-coco.pth"
  echo "  external\\humanoid-projects\\GVHMR\\inputs\\checkpoints\\hmr2\\epoch=10-step=25000.ckpt"
fi

echo ""
echo "-- Waiting for GPU worker (Celery) --"
deadline=$(( $(date +%s) + 900 ))
last_print=0
while true; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "[ERROR] No GPU Celery worker detected within 15 minutes."
    echo "Start the Windows GPU worker, then re-run:"
    echo "  $ROOT_DIR/scripts/smoke_e2e_with_gpu_real.sh \"$VIDEO_PATH\""
    exit 1
  fi

  workers_json="$(curl -sS "http://localhost:8000/ops/workers?timeout=2.0" || true)"

  if (( now - last_print >= 10 )); then
    WORKERS_JSON="$workers_json" python - <<'PY' || true
import json,sys
import os
try:
  d=json.loads(os.environ.get("WORKERS_JSON") or "{}")
except Exception:
  print("Workers: (unavailable)")
  raise SystemExit(0)
names=d.get("worker_names") or []
has_gpu=bool(d.get("has_gpu_queue"))
print(f"Workers: {names} gpu_queue={has_gpu}")
PY
    last_print=$now
  fi

  ok="$(WORKERS_JSON="$workers_json" python - <<'PY'
import json,sys
import os
try:
  d=json.loads(os.environ.get("WORKERS_JSON") or "{}")
  print("1" if d.get("ok") else "0")
except Exception:
  print("0")
PY
)"
  if [[ "$ok" == "1" ]]; then
    has_gpu="$(WORKERS_JSON="$workers_json" python - <<'PY'
import json,sys
import os
d=json.loads(os.environ.get("WORKERS_JSON") or "{}")
print("1" if d.get("has_gpu_queue") is True else "0")
PY
)"
    if [[ "$has_gpu" == "1" ]]; then
      echo "GPU worker detected."
      break
    fi
  fi
  sleep 2
done

echo ""
"$ROOT_DIR/scripts/smoke_e2e_with_gpu_real.sh" "$VIDEO_PATH"
