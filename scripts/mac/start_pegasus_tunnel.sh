#!/usr/bin/env bash
set -euo pipefail

# Start local SSH port-forwards to a running Pegasus control-plane.
#
# Usage:
#   PEGASUS_HOST=<pegasus_host_or_alias> ./scripts/mac/start_pegasus_tunnel.sh
#   ./scripts/mac/start_pegasus_tunnel.sh <pegasus_host_or_alias>
#
# This reads the control-plane endpoints file on Pegasus and forwards:
# - API      (remote 18000) -> localhost:8000
# - MinIO    (remote 19000) -> localhost:9000
# - MinIO UI (remote 19001) -> localhost:9001
# - Redis    (remote 16379) -> localhost:6379
# - Postgres (remote 15432) -> localhost:5432
#
# Optional env vars:
# - PEGASUS_REPO (default: ~/han-platform)
# - CP_ENDPOINTS_FILE (default: $PEGASUS_REPO/tmp/pegasus_control_plane/endpoints.env)
# - SSH_USER / SSH_KEY (optional)
# - PID_FILE (default: <repo>/tmp/pegasus_tunnel.pid)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
CP_ENDPOINTS_FILE="${CP_ENDPOINTS_FILE:-$PEGASUS_REPO/tmp/pegasus_control_plane/endpoints.env}"
PID_FILE="${PID_FILE:-$ROOT_DIR/tmp/pegasus_tunnel.pid}"

mkdir -p "$(dirname "$PID_FILE")"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
    echo "Pegasus tunnel already running (pid=$old_pid)."
    exit 0
  fi
  rm -f "$PID_FILE"
fi

ssh_target="$PEGASUS_HOST"
if [[ -n "$SSH_USER" && "$ssh_target" != *@* ]]; then
  ssh_target="${SSH_USER}@${ssh_target}"
fi

ssh_cmd=(ssh)
if [[ -n "$SSH_KEY" ]]; then
  ssh_cmd+=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey)
fi

env_text="$("${ssh_cmd[@]}" "$ssh_target" "cat $CP_ENDPOINTS_FILE" 2>/dev/null || true)"
if [[ -z "$env_text" ]]; then
  echo "[ERROR] Could not read Pegasus endpoints file: $CP_ENDPOINTS_FILE" >&2
  echo "Start the control-plane first: ./scripts/mac/start_pegasus_control_plane_ssh.sh" >&2
  exit 1
fi

get_val() {
  key="$1"
  printf '%s\n' "$env_text" | sed -n "s/^${key}=//p" | tail -n 1
}

control_host="$(get_val PEGASUS_CONTROL_PLANE_HOST)"
api_port="$(get_val API_PORT)"
minio_port="$(get_val MINIO_PORT)"
minio_console_port="$(get_val MINIO_CONSOLE_PORT)"
redis_port="$(get_val REDIS_PORT)"
postgres_port="$(get_val POSTGRES_PORT)"

api_url="$(get_val API_URL)"
s3_endpoint="$(get_val S3_ENDPOINT)"
redis_url="$(get_val REDIS_URL)"
db_url="$(get_val DATABASE_URL)"

if [[ -z "$control_host" ]]; then
  # Fall back to parsing host from API_URL.
  control_host="$(printf '%s' "$api_url" | sed -E 's#^https?://##' | cut -d: -f1)"
fi

if [[ -z "$api_port" && -n "$api_url" ]]; then
  api_port="$(printf '%s' "$api_url" | sed -E 's#^https?://##' | rev | cut -d: -f1 | rev)"
fi
if [[ -z "$minio_port" && -n "$s3_endpoint" ]]; then
  minio_port="$(printf '%s' "$s3_endpoint" | sed -E 's#^https?://##' | rev | cut -d: -f1 | rev)"
fi
if [[ -z "$redis_port" && -n "$redis_url" ]]; then
  redis_port="$(printf '%s' "$redis_url" | sed -E 's#^redis://##' | cut -d/ -f1 | rev | cut -d: -f1 | rev)"
fi
if [[ -z "$postgres_port" && -n "$db_url" ]]; then
  postgres_port="$(printf '%s' "$db_url" | rev | cut -d@ -f1 | rev | cut -d/ -f1 | rev | cut -d: -f1 | rev)"
fi

api_port="${api_port:-18000}"
minio_port="${minio_port:-19000}"
minio_console_port="${minio_console_port:-19001}"
redis_port="${redis_port:-16379}"
postgres_port="${postgres_port:-15432}"

echo "== Pegasus Tunnel =="
echo "Pegasus host:   $PEGASUS_HOST"
echo "Control-plane:  $control_host"
echo "Endpoints file: $CP_ENDPOINTS_FILE"
echo ""
echo "Forwarding:"
echo "  API      $control_host:$api_port -> localhost:8000"
echo "  MinIO    $control_host:$minio_port -> localhost:9000"
echo "  MinIO UI $control_host:$minio_console_port -> localhost:9001"
echo "  Redis    $control_host:$redis_port -> localhost:6379"
echo "  Postgres $control_host:$postgres_port -> localhost:5432"
echo ""

set +e
"${ssh_cmd[@]}" -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "8000:${control_host}:${api_port}" \
  -L "9000:${control_host}:${minio_port}" \
  -L "9001:${control_host}:${minio_console_port}" \
  -L "6379:${control_host}:${redis_port}" \
  -L "5432:${control_host}:${postgres_port}" \
  "$ssh_target" >/dev/null 2>&1 &
pid="$!"
set -e

sleep 1
if ! kill -0 "$pid" >/dev/null 2>&1; then
  echo "[ERROR] SSH tunnel failed to start." >&2
  exit 1
fi

echo "$pid" >"$PID_FILE"
echo "Tunnel PID: $pid"
echo "PID file:   $PID_FILE"
echo ""
echo "API health: http://localhost:8000/health"

