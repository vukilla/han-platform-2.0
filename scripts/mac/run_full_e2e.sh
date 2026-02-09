#!/usr/bin/env bash
set -euo pipefail

# Runs control-plane + waits for a GPU worker + runs the full E2E smoke.
# You still need to start the Windows GPU worker once this prints your Mac IP.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "== Full E2E (Mac + Windows GPU worker) =="

"$ROOT_DIR/scripts/mac/control_plane_up.sh"

echo ""
echo "-- Waiting for GPU worker (Celery) --"
deadline=$(( $(date +%s) + 180 ))
while true; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "[ERROR] No Celery workers responded to ping within 3 minutes."
    echo "Start the Windows GPU worker, then re-run:"
    echo "  $ROOT_DIR/scripts/smoke_e2e_with_gpu.sh"
    exit 1
  fi

  workers_json="$(curl -sS http://localhost:8000/ops/workers || true)"
  ok="$(python - <<'PY'
import json,sys
try:
  d=json.load(sys.stdin)
  print("1" if d.get("ok") else "0")
except Exception:
  print("0")
PY
<<<"$workers_json")"
  if [[ "$ok" == "1" ]]; then
    # Any worker is enough to proceed, but we also check that at least one queue includes "gpu".
    has_gpu="$(python - <<'PY'
import json,sys
d=json.load(sys.stdin)
queues=d.get("active_queues") or {}
for _, qlist in queues.items():
  for q in qlist or []:
    if (q or {}).get("name") == "gpu":
      print("1")
      raise SystemExit(0)
print("0")
PY
<<<"$workers_json")"
    if [[ "$has_gpu" == "1" ]]; then
      echo "GPU worker detected."
      break
    fi
  fi
  sleep 2
done

echo ""
"$ROOT_DIR/scripts/smoke_e2e_with_gpu.sh"

