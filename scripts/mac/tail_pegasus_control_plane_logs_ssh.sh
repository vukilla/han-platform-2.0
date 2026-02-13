#!/usr/bin/env bash
set -euo pipefail

# Tail Pegasus control-plane logs.
#
# Usage:
#   PEGASUS_HOST=<pegasus_host_or_alias> ./scripts/mac/tail_pegasus_control_plane_logs_ssh.sh
#   ./scripts/mac/tail_pegasus_control_plane_logs_ssh.sh <pegasus_host_or_alias>

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

ssh_target="$PEGASUS_HOST"
if [[ -n "$SSH_USER" && "$ssh_target" != *@* ]]; then
  ssh_target="${SSH_USER}@${ssh_target}"
fi

ssh_cmd=(ssh)
if [[ -n "$SSH_KEY" ]]; then
  ssh_cmd+=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey)
fi

"${ssh_cmd[@]}" "$ssh_target" "bash -s -- $(printf '%q ' "$PEGASUS_REPO")" <<'REMOTE'
set -euo pipefail

repo="$1"
log_dir="$repo/tmp/pegasus_control_plane/logs"
mkdir -p "$log_dir"
touch "$log_dir/"{slurm.out.log,slurm.err.log,api.log,cpu_worker.log,redis.log,postgres.log,minio.log,status.log}
tail -n 120 -f \
  "$log_dir/slurm.out.log" \
  "$log_dir/slurm.err.log" \
  "$log_dir/status.log" \
  "$log_dir/api.log" \
  "$log_dir/cpu_worker.log" \
  "$log_dir/redis.log" \
  "$log_dir/postgres.log" \
  "$log_dir/minio.log"
REMOTE
