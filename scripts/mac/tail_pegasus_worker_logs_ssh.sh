#!/usr/bin/env bash
set -euo pipefail

# Tail the Linux worker logs on Pegasus over SSH.
#
# Usage:
#   PEGASUS_HOST=<pegasus_host_or_alias> ./scripts/mac/tail_pegasus_worker_logs_ssh.sh
#   ./scripts/mac/tail_pegasus_worker_logs_ssh.sh <pegasus_host_or_alias>

PEGASUS_HOST="${PEGASUS_HOST:-${1:-}}"
if [[ -z "$PEGASUS_HOST" ]]; then
  echo "Missing Pegasus host." >&2
  echo "Usage: PEGASUS_HOST=<host> $0" >&2
  echo "   or: $0 <host>" >&2
  exit 1
fi

SSH_USER="${SSH_USER:-}"
SSH_KEY="${SSH_KEY:-}"
if [[ -z "$SSH_KEY" ]]; then
  for candidate in \
    "$HOME/.ssh/pegasus" \
    "$HOME/.ssh/pegasus_ssh" \
    "$HOME/.ssh/pegasus_key" \
    "$HOME/.ssh/id_ed25519" \
    "$HOME/.ssh/id_rsa"; do
    if [[ -f "$candidate" ]]; then
      SSH_KEY="$candidate"
      break
    fi
  done
fi
PEGASUS_REPO="${PEGASUS_REPO:-~/han-platform}"
OUT_LOG="${PEGASUS_REPO}/tmp/pegasus_worker.out.log"
ERR_LOG="${PEGASUS_REPO}/tmp/pegasus_worker.err.log"

echo "== Tail Pegasus Worker Logs over SSH =="
echo "Pegasus host: $PEGASUS_HOST"
if [[ -n "$SSH_USER" ]]; then
  echo "SSH user:     $SSH_USER"
else
  echo "SSH user:     <from ssh config>"
fi
echo "Out log:      $OUT_LOG"
echo "Err log:      $ERR_LOG"
echo ""

ssh_target="$PEGASUS_HOST"
if [[ -n "$SSH_USER" && "$ssh_target" != *@* ]]; then
  ssh_target="${SSH_USER}@${ssh_target}"
fi

ssh_cmd=(ssh)
if [[ -n "$SSH_KEY" ]]; then
  ssh_cmd+=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey)
fi

"${ssh_cmd[@]}" "$ssh_target" "bash -s -- $(printf '%q ' "$OUT_LOG" "$ERR_LOG")" <<'REMOTE'
set -euo pipefail

out_log="$1"
err_log="$2"

touch "$out_log" "$err_log"
tail -n 120 -f "$out_log" "$err_log"
REMOTE
