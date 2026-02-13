#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Pegasus dependencies for running a private control-plane without Docker.
#
# Usage:
#   PEGASUS_HOST=<pegasus_host_or_alias> ./scripts/mac/bootstrap_pegasus_control_plane_ssh.sh
#   ./scripts/mac/bootstrap_pegasus_control_plane_ssh.sh <pegasus_host_or_alias>

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
CONDA_PREFIX_PATH="${CONDA_PREFIX_PATH:-$PEGASUS_REPO/.conda-controlplane}"

ssh_target="$PEGASUS_HOST"
if [[ -n "$SSH_USER" && "$ssh_target" != *@* ]]; then
  ssh_target="${SSH_USER}@${ssh_target}"
fi

ssh_cmd=(ssh)
if [[ -n "$SSH_KEY" ]]; then
  ssh_cmd+=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey)
fi

echo "== Bootstrap Pegasus Control Plane =="
echo "Pegasus host:      $PEGASUS_HOST"
echo "Remote repo:       $PEGASUS_REPO"
echo "Conda env prefix:  $CONDA_PREFIX_PATH"
echo ""

"${ssh_cmd[@]}" "$ssh_target" "bash -s -- $(printf '%q ' "$PEGASUS_REPO" "$CONDA_PREFIX_PATH")" <<'REMOTE'
set -euo pipefail

repo="$1"
conda_prefix="$2"

if [[ ! -d "$repo/.git" ]]; then
  git clone https://github.com/vukilla/han-platform.git "$repo"
fi
git -C "$repo" fetch --all --prune
git -C "$repo" checkout codex/pegasus-dual-worker
git -C "$repo" pull --ff-only

conda_root="/netscratch/${USER}/miniconda3"
if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
  echo "[ERROR] Miniconda not found at $conda_root" >&2
  exit 1
fi
source "$conda_root/etc/profile.d/conda.sh"

if [[ ! -x "$conda_prefix/bin/python" ]]; then
  conda create -y -p "$conda_prefix" -c conda-forge \
    python=3.11 \
    redis-server \
    postgresql \
    minio-server
fi

python_bin="$conda_prefix/bin/python"
"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install -r "$repo/apps/api/requirements.txt"

echo "Bootstrap complete."
echo "Python: $python_bin"
echo "redis-server: $("$conda_prefix/bin/redis-server" --version | head -n 1)"
echo "psql: $("$conda_prefix/bin/psql" --version)"
echo "minio: $("$conda_prefix/bin/minio" --version | head -n 1)"
REMOTE
