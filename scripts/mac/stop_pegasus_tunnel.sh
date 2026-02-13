#!/usr/bin/env bash
set -euo pipefail

# Stop the local Pegasus SSH tunnel started by start_pegasus_tunnel.sh.
#
# Usage:
#   ./scripts/mac/stop_pegasus_tunnel.sh
#
# Optional env vars:
# - PID_FILE (default: <repo>/tmp/pegasus_tunnel.pid)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID_FILE="${PID_FILE:-$ROOT_DIR/tmp/pegasus_tunnel.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No tunnel PID file found: $PID_FILE"
  exit 0
fi

pid="$(cat "$PID_FILE" || true)"
if [[ -z "$pid" ]]; then
  rm -f "$PID_FILE"
  echo "Removed empty PID file: $PID_FILE"
  exit 0
fi

if kill -0 "$pid" >/dev/null 2>&1; then
  kill "$pid" >/dev/null 2>&1 || true
  sleep 1
fi

rm -f "$PID_FILE"
echo "Stopped tunnel pid=$pid"

