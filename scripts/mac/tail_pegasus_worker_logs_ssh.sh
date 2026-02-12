#!/usr/bin/env bash
set -euo pipefail

# Tail the Linux worker logs on Pegasus over SSH.
#
# Usage:
#   PEGASUS_HOST=pegasus.dfki.de ./scripts/mac/tail_pegasus_worker_logs_ssh.sh
#   ./scripts/mac/tail_pegasus_worker_logs_ssh.sh pegasus.dfki.de

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
OUT_LOG="${PEGASUS_REPO}/tmp/pegasus_worker.out.log"
ERR_LOG="${PEGASUS_REPO}/tmp/pegasus_worker.err.log"

echo "== Tail Pegasus Worker Logs over SSH =="
echo "Pegasus host: $PEGASUS_HOST"
echo "SSH user:     $SSH_USER"
echo "Out log:      $OUT_LOG"
echo "Err log:      $ERR_LOG"
echo ""

ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey "${SSH_USER}@${PEGASUS_HOST}" \
  "bash -s -- $(printf '%q ' "$OUT_LOG" "$ERR_LOG")" <<'REMOTE'
set -euo pipefail

out_log="$1"
err_log="$2"

touch "$out_log" "$err_log"
tail -n 120 -f "$out_log" "$err_log"
REMOTE
