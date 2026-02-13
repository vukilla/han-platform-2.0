#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yml"

echo "== Control-plane (Mac) =="
echo "Repo: $ROOT_DIR"
echo ""

echo "-- Starting Docker Compose --"
docker compose -f "$COMPOSE_FILE" up -d --build --remove-orphans

echo "-- Running migrations --"
docker compose -f "$COMPOSE_FILE" exec -T api alembic upgrade head

echo "-- Health check --"
deadline=$(( $(date +%s) + 60 ))
while true; do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    break
  fi
  now=$(date +%s)
  if (( now > deadline )); then
    echo "[ERROR] API health endpoint did not become ready within 60s." >&2
    docker compose -f "$COMPOSE_FILE" ps >&2 || true
    exit 1
  fi
  sleep 1
done
curl -sS http://localhost:8000/health | python -m json.tool

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
  echo "Windows command to start GPU worker:"
  echo "  powershell -ExecutionPolicy Bypass -File .\\scripts\\windows\\one_click_gpu_worker.ps1 -MacIp $MAC_IP -IsaacSimPath C:\\isaacsim"
fi

echo ""
echo "Web: http://localhost:3000"
echo "API: http://localhost:8000/health"
