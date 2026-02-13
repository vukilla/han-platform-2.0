#!/usr/bin/env bash
set -euo pipefail

# Stop the Linux worker on Pegasus over SSH.
#
# Usage:
#   PEGASUS_HOST=<pegasus_host_or_alias> ./scripts/mac/stop_pegasus_worker_ssh.sh
#   ./scripts/mac/stop_pegasus_worker_ssh.sh <pegasus_host_or_alias>

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
PID_FILE="${PEGASUS_REPO}/tmp/pegasus_worker.pid"
JOB_ID_FILE="${PEGASUS_REPO}/tmp/pegasus_worker.slurm_job_id"

echo "== Stop Pegasus Worker over SSH =="
echo "Pegasus host: $PEGASUS_HOST"
if [[ -n "$SSH_USER" ]]; then
  echo "SSH user:     $SSH_USER"
else
  echo "SSH user:     <from ssh config>"
fi
echo "PID file:     $PID_FILE"
echo "Job ID file:  $JOB_ID_FILE"
echo ""

ssh_target="$PEGASUS_HOST"
if [[ -n "$SSH_USER" && "$ssh_target" != *@* ]]; then
  ssh_target="${SSH_USER}@${ssh_target}"
fi

ssh_cmd=(ssh)
if [[ -n "$SSH_KEY" ]]; then
  ssh_cmd+=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey)
fi

"${ssh_cmd[@]}" "$ssh_target" "bash -s -- $(printf '%q ' "$PID_FILE" "$JOB_ID_FILE")" <<'REMOTE'
set -euo pipefail

pid_file="$1"
job_id_file="$2"

if [[ -f "$job_id_file" ]]; then
  job_id="$(cat "$job_id_file" || true)"
  if [[ -n "$job_id" ]]; then
    if command -v scancel >/dev/null 2>&1; then
      if scancel "$job_id" >/dev/null 2>&1; then
        echo "Cancelled Slurm job id=$job_id"
      else
        echo "Slurm job id=$job_id was not active."
      fi
    else
      echo "scancel not found on remote host, cannot cancel job id=$job_id"
    fi
  fi
  rm -f "$job_id_file"
fi

if [[ ! -f "$pid_file" ]]; then
  echo "No PID file found."
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
