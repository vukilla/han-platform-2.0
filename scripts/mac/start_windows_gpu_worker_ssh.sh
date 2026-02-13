#!/usr/bin/env bash
set -euo pipefail

# Start (or restart) the Windows GPU worker via SSH so the Mac control-plane can run GPU-queued jobs.
#
# Requires:
# - Windows OpenSSH server enabled
# - SSH key present on Mac (default: ~/.ssh/han_windows)
#
# Usage:
#   WINDOWS_GPU_IP=192.168.2.77 ./scripts/mac/start_windows_gpu_worker_ssh.sh
#   ./scripts/mac/start_windows_gpu_worker_ssh.sh 192.168.2.77
#
# Optional env vars:
# - SSH_USER (default: rvuko)
# - SSH_KEY (default: ~/.ssh/han_windows)
# - WINDOWS_REPO (default: C:\Users\<user>\OneDrive\Documents\GitHub\han-platform)
# - ISAACSIM_PATH (default: C:\isaacsim)
# - REAL_WORKER=1 (use one_click_gpu_worker_real.ps1)
# - SETUP_GVHMR=1 (only when REAL_WORKER=1)
# - DOWNLOAD_LIGHT=1 (only when REAL_WORKER=1)
# - WORKER_QUEUES (optional: "pose", "gpu", or "pose,gpu")
# - WORKER_SOURCE (default: windows)
# - MAC_LAN_IP (override autodetected)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WINDOWS_IP="${WINDOWS_GPU_IP:-${1:-}}"
if [[ -z "${WINDOWS_IP}" ]]; then
  echo "Missing Windows IP." >&2
  echo "Usage: WINDOWS_GPU_IP=<windows_ip> $0" >&2
  echo "   or: $0 <windows_ip>" >&2
  exit 1
fi

MAC_IP="${MAC_LAN_IP:-$(ipconfig getifaddr en0 || true)}"
if [[ -z "$MAC_IP" ]]; then
  MAC_IP="$(ipconfig getifaddr en1 || true)"
fi
if [[ -z "$MAC_IP" ]]; then
  echo "Could not determine Mac LAN IP (en0/en1). Set MAC_LAN_IP manually." >&2
  exit 1
fi

SSH_USER="${SSH_USER:-rvuko}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/han_windows}"
WINDOWS_REPO="${WINDOWS_REPO:-C:\\Users\\${SSH_USER}\\OneDrive\\Documents\\GitHub\\han-platform}"
ISAACSIM_PATH="${ISAACSIM_PATH:-C:\\isaacsim}"

REAL_WORKER="${REAL_WORKER:-0}"
SETUP_GVHMR="${SETUP_GVHMR:-0}"
DOWNLOAD_LIGHT="${DOWNLOAD_LIGHT:-0}"
WORKER_QUEUES="${WORKER_QUEUES:-}"
WORKER_SOURCE="${WORKER_SOURCE:-windows}"

if [[ "$REAL_WORKER" == "1" ]]; then
  PS1_REL="scripts\\windows\\one_click_gpu_worker_real.ps1"
  EXTRA_ARGS=""
  if [[ "$SETUP_GVHMR" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} -SetupGVHMR"
  fi
  if [[ "$DOWNLOAD_LIGHT" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} -DownloadLightCheckpoints"
  fi
else
  PS1_REL="scripts\\windows\\one_click_gpu_worker.ps1"
  EXTRA_ARGS=""
fi

PS1_PATH="${WINDOWS_REPO}\\${PS1_REL}"

echo "== Start Windows GPU worker over SSH =="
echo "Repo:       $ROOT_DIR"
echo "Windows IP: $WINDOWS_IP"
echo "Mac IP:     $MAC_IP"
echo "SSH user:   $SSH_USER"
echo "PS1:        $PS1_PATH"
if [[ -n "$WORKER_QUEUES" ]]; then
  echo "Queues:     $WORKER_QUEUES"
fi
echo "Source:     $WORKER_SOURCE"
echo ""

QUEUES_ARG=""
if [[ -n "$WORKER_QUEUES" ]]; then
  QUEUES_ARG=" -Queues '${WORKER_QUEUES}'"
fi

  REMOTE_CMD="& { git -C '${WINDOWS_REPO}' pull | Out-Host; & '${PS1_PATH}' -MacIp '${MAC_IP}' -IsaacSimPath '${ISAACSIM_PATH}' -WorkerSource '${WORKER_SOURCE}'${QUEUES_ARG}${EXTRA_ARGS} }"

ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey "${SSH_USER}@${WINDOWS_IP}" \
  "powershell -NoProfile -ExecutionPolicy Bypass -Command \"$REMOTE_CMD\""
