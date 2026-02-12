#!/usr/bin/env bash
set -euo pipefail

# Show Pegasus private control-plane status and endpoints.
#
# Usage:
#   PEGASUS_HOST=dfki ./scripts/mac/status_pegasus_control_plane_ssh.sh
#   ./scripts/mac/status_pegasus_control_plane_ssh.sh dfki

PEGASUS_HOST="${PEGASUS_HOST:-${1:-}}"
if [[ -z "$PEGASUS_HOST" ]]; then
  echo "Missing Pegasus host." >&2
  echo "Usage: PEGASUS_HOST=<host> $0" >&2
  echo "   or: $0 <host>" >&2
  exit 1
fi

SSH_USER="${SSH_USER:-}"
SSH_KEY="${SSH_KEY:-}"
if [[ -z "$SSH_KEY" && -f "$HOME/.ssh/dfki_pegasus" ]]; then
  SSH_KEY="$HOME/.ssh/dfki_pegasus"
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
state_root="$repo/tmp/pegasus_control_plane"
job_id_file="$state_root/job_id"
endpoints_file="$state_root/endpoints.env"

if [[ -f "$job_id_file" ]]; then
  job_id="$(cat "$job_id_file" || true)"
  echo "job_id=${job_id}"
  if [[ -n "$job_id" ]]; then
    squeue -j "$job_id" -o "%.18i %.9P %.10T %.12M %.30R" | sed -n '1,5p'
  fi
else
  echo "job_id=<none>"
fi

if [[ -f "$endpoints_file" ]]; then
  echo ""
  echo "endpoints:"
  cat "$endpoints_file"
else
  echo ""
  echo "endpoints=<missing>"
fi
REMOTE
