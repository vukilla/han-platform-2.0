#!/usr/bin/env bash
set -euo pipefail

# Stop the Linux worker on Pegasus over SSH.
#
# Usage:
#   PEGASUS_HOST=pegasus.dfki.de ./scripts/mac/stop_pegasus_worker_ssh.sh
#   ./scripts/mac/stop_pegasus_worker_ssh.sh pegasus.dfki.de

PEGASUS_HOST="${PEGASUS_HOST:-${1:-}}"
if [[ -z "$PEGASUS_HOST" ]]; then
  echo "Missing Pegasus host." >&2
  echo "Usage: PEGASUS_HOST=<host> $0" >&2
  echo "   or: $0 <host>" >&2
  exit 1
fi

SSH_USER="${SSH_USER:-rvuko}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/pegasus}"
PEGASUS_REPO="${PEGASUS_REPO:-~/han-platform}"
PID_FILE="${PEGASUS_REPO}/tmp/pegasus_worker.pid"

echo "== Stop Pegasus Worker over SSH =="
echo "Pegasus host: $PEGASUS_HOST"
echo "SSH user:     $SSH_USER"
echo "PID file:     $PID_FILE"
echo ""

ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey "${SSH_USER}@${PEGASUS_HOST}" \
  "bash -s -- $(printf '%q ' "$PID_FILE")" <<'REMOTE'
set -euo pipefail

pid_file="$1"

if [[ ! -f "$pid_file" ]]; then
  echo "No PID file found, worker is already stopped."
  exit 0
fi

pid="$(cat "$pid_file" || true)"
if [[ -z "$pid" ]]; then
  rm -f "$pid_file"
  echo "PID file was empty, removed stale PID file."
  exit 0
fi

if kill -0 "$pid" >/dev/null 2>&1; then
  kill "$pid" || true
  sleep 1
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -9 "$pid" || true
  fi
  echo "Stopped worker PID=$pid"
else
  echo "Worker PID=$pid was not running."
fi

rm -f "$pid_file"
echo "Removed PID file: $pid_file"
REMOTE
