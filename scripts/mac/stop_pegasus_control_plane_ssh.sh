#!/usr/bin/env bash
set -euo pipefail

# Stop the Pegasus private control-plane Slurm job.
#
# Usage:
#   PEGASUS_HOST=dfki ./scripts/mac/stop_pegasus_control_plane_ssh.sh
#   ./scripts/mac/stop_pegasus_control_plane_ssh.sh dfki

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

echo "== Stop Pegasus Control Plane =="
echo "Pegasus host: $PEGASUS_HOST"
echo "Remote repo:  $PEGASUS_REPO"
echo ""

"${ssh_cmd[@]}" "$ssh_target" "bash -s -- $(printf '%q ' "$PEGASUS_REPO")" <<'REMOTE'
set -euo pipefail

repo="$1"
state_root="$repo/tmp/pegasus_control_plane"
job_id_file="$state_root/job_id"

if [[ ! -f "$job_id_file" ]]; then
  echo "No control-plane job ID file found, nothing to stop."
  exit 0
fi

job_id="$(cat "$job_id_file" || true)"
if [[ -z "$job_id" ]]; then
  rm -f "$job_id_file"
  echo "Job ID file was empty and has been removed."
  exit 0
fi

if scancel "$job_id" >/dev/null 2>&1; then
  echo "Cancelled control-plane Slurm job id=$job_id"
else
  echo "Slurm job id=$job_id was not active."
fi

rm -f "$job_id_file"
echo "Removed: $job_id_file"
REMOTE
