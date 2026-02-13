#!/usr/bin/env bash
set -euo pipefail

# Start a private control-plane on Pegasus using a Slurm CPU allocation.
#
# Usage:
#   PEGASUS_HOST=dfki ./scripts/mac/start_pegasus_control_plane_ssh.sh
#   ./scripts/mac/start_pegasus_control_plane_ssh.sh dfki
#
# Optional env vars:
# - SSH_USER
# - SSH_KEY (default: ~/.ssh/dfki_pegasus if present)
# - PEGASUS_REPO (default: ~/han-platform)
# - CONDA_PREFIX_PATH (default: ~/han-platform/.conda-controlplane)
# - PULL_REPO (default: 1)
# - SLURM_PARTITION (default: batch)
# - SLURM_TIME (default: 12:00:00)
# - SLURM_CPUS_PER_TASK (default: 2)
# - SLURM_MEM (default: 8G)
# - HAN_CP_POSTGRES_PORT (default: 15432)
# - HAN_CP_REDIS_PORT (default: 16379)
# - HAN_CP_MINIO_PORT (default: 19000)
# - HAN_CP_MINIO_CONSOLE_PORT (default: 19001)
# - HAN_CP_API_PORT (default: 18000)
# - HAN_CP_MINIO_ACCESS_KEY (default: minioadmin)
# - HAN_CP_MINIO_SECRET_KEY (default: minioadmin)
# - HAN_CP_S3_BUCKET (default: humanoid-network-dev)
# - HAN_CP_S3_PUBLIC_ENDPOINT (optional, browser-facing endpoint for presigned URLs)
# - HAN_CP_STATE_ROOT (optional, for example /local/$USER/han_cp_state to avoid home quota)

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
CONDA_PREFIX_PATH="${CONDA_PREFIX_PATH:-$PEGASUS_REPO/.conda-controlplane}"
PULL_REPO="${PULL_REPO:-1}"

SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-2}"
SLURM_MEM="${SLURM_MEM:-8G}"

HAN_CP_POSTGRES_PORT="${HAN_CP_POSTGRES_PORT:-15432}"
HAN_CP_REDIS_PORT="${HAN_CP_REDIS_PORT:-16379}"
HAN_CP_MINIO_PORT="${HAN_CP_MINIO_PORT:-19000}"
HAN_CP_MINIO_CONSOLE_PORT="${HAN_CP_MINIO_CONSOLE_PORT:-19001}"
HAN_CP_API_PORT="${HAN_CP_API_PORT:-18000}"
HAN_CP_MINIO_ACCESS_KEY="${HAN_CP_MINIO_ACCESS_KEY:-minioadmin}"
HAN_CP_MINIO_SECRET_KEY="${HAN_CP_MINIO_SECRET_KEY:-minioadmin}"
HAN_CP_S3_BUCKET="${HAN_CP_S3_BUCKET:-humanoid-network-dev}"
# Clients typically access the Pegasus control-plane via SSH port-forwarding.
# Default to localhost so presigned URLs work without requiring cluster DNS on the client machine.
HAN_CP_S3_PUBLIC_ENDPOINT="${HAN_CP_S3_PUBLIC_ENDPOINT:-http://127.0.0.1:9000}"
HAN_CP_STATE_ROOT="${HAN_CP_STATE_ROOT:-}"

ssh_target="$PEGASUS_HOST"
if [[ -n "$SSH_USER" && "$ssh_target" != *@* ]]; then
  ssh_target="${SSH_USER}@${ssh_target}"
fi

ssh_cmd=(ssh)
if [[ -n "$SSH_KEY" ]]; then
  ssh_cmd+=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey)
fi

echo "== Start Pegasus Control Plane over SSH =="
echo "Pegasus host:     $PEGASUS_HOST"
echo "Remote repo:      $PEGASUS_REPO"
echo "Conda env:        $CONDA_PREFIX_PATH"
echo "Partition:        $SLURM_PARTITION"
echo "Walltime:         $SLURM_TIME"
echo "API port:         $HAN_CP_API_PORT"
echo "Redis port:       $HAN_CP_REDIS_PORT"
echo "Postgres port:    $HAN_CP_POSTGRES_PORT"
echo "MinIO port:       $HAN_CP_MINIO_PORT"
echo ""

"${ssh_cmd[@]}" "$ssh_target" \
  "bash -s -- $(printf '%q ' "$PEGASUS_REPO" "$CONDA_PREFIX_PATH" "$PULL_REPO" "$SLURM_PARTITION" "$SLURM_TIME" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM" "$HAN_CP_POSTGRES_PORT" "$HAN_CP_REDIS_PORT" "$HAN_CP_MINIO_PORT" "$HAN_CP_MINIO_CONSOLE_PORT" "$HAN_CP_API_PORT" "$HAN_CP_MINIO_ACCESS_KEY" "$HAN_CP_MINIO_SECRET_KEY" "$HAN_CP_S3_BUCKET" "$HAN_CP_S3_PUBLIC_ENDPOINT" "$HAN_CP_STATE_ROOT")" <<'REMOTE'
set -euo pipefail

repo="$1"
conda_prefix="$2"
pull_repo="$3"
slurm_partition="$4"
slurm_time="$5"
slurm_cpus="$6"
slurm_mem="$7"
pg_port="$8"
redis_port="$9"
minio_port="${10}"
minio_console_port="${11}"
api_port="${12}"
minio_access_key="${13}"
minio_secret_key="${14}"
s3_bucket="${15}"
s3_public_endpoint="${16}"
cp_state_root="${17}"

state_root="$repo/tmp/pegasus_control_plane"
mkdir -p "$state_root/logs"
job_id_file="$state_root/job_id"
sbatch_script="$state_root/run_control_plane.sbatch"

if [[ ! -d "$repo/.git" ]]; then
  echo "[ERROR] Repo missing at $repo. Run bootstrap script first." >&2
  exit 1
fi

if [[ "$pull_repo" == "1" ]]; then
  git -C "$repo" pull --ff-only
fi

if [[ ! -x "$conda_prefix/bin/python" ]]; then
  echo "[ERROR] Missing conda environment at $conda_prefix" >&2
  echo "Run scripts/mac/bootstrap_pegasus_control_plane_ssh.sh first." >&2
  exit 1
fi

if [[ -f "$job_id_file" ]]; then
  old_job_id="$(cat "$job_id_file" || true)"
  if [[ -n "$old_job_id" ]]; then
    scancel "$old_job_id" >/dev/null 2>&1 || true
    sleep 1
  fi
  rm -f "$job_id_file"
fi

cat >"$sbatch_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="$conda_prefix/bin:\$PATH"
export HAN_CP_ROOT_DIR="$repo"
export HAN_CP_POSTGRES_PORT="$pg_port"
export HAN_CP_REDIS_PORT="$redis_port"
export HAN_CP_MINIO_PORT="$minio_port"
export HAN_CP_MINIO_CONSOLE_PORT="$minio_console_port"
export HAN_CP_API_PORT="$api_port"
export HAN_CP_MINIO_ACCESS_KEY="$minio_access_key"
export HAN_CP_MINIO_SECRET_KEY="$minio_secret_key"
export HAN_CP_S3_BUCKET="$s3_bucket"
export HAN_CP_S3_PUBLIC_ENDPOINT="$s3_public_endpoint"
export HAN_CP_STATE_ROOT="$cp_state_root"
bash "$repo/scripts/pegasus/run_control_plane.sh"
EOF
chmod +x "$sbatch_script"

job_id="$(sbatch --parsable \
  --job-name han-control-plane \
  --partition "$slurm_partition" \
  --time "$slurm_time" \
  --cpus-per-task "$slurm_cpus" \
  --mem "$slurm_mem" \
  --output "$state_root/logs/slurm.out.log" \
  --error "$state_root/logs/slurm.err.log" \
  "$sbatch_script")"
echo "$job_id" >"$job_id_file"

deadline=$(( $(date +%s) + 240 ))
state=""
node=""
while true; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "[ERROR] Timed out waiting for control-plane job to start." >&2
    exit 1
  fi
  state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)"
  node="$(squeue -h -j "$job_id" -o '%N' 2>/dev/null || true)"
  if [[ "$state" == "RUNNING" && -n "$node" && "$node" != "(null)" ]]; then
    break
  fi
  if [[ -z "$state" ]]; then
    sacct_state="$(sacct -j "$job_id" --format=State -n -P 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    if [[ -n "$sacct_state" && "$sacct_state" != "RUNNING" && "$sacct_state" != "PENDING" ]]; then
      echo "[ERROR] Control-plane job failed early (state=$sacct_state)." >&2
      exit 1
    fi
  fi
  sleep 2
done

if [[ -z "$s3_public_endpoint" ]]; then
  s3_public_endpoint="http://$node:$minio_port"
fi

cat >"$state_root/endpoints.env" <<EOF
PEGASUS_CONTROL_PLANE_HOST=$node
DATABASE_URL=postgresql+psycopg://han:han@$node:$pg_port/han
REDIS_URL=redis://$node:$redis_port/0
S3_ENDPOINT=http://$node:$minio_port
S3_PUBLIC_ENDPOINT=$s3_public_endpoint
S3_ACCESS_KEY=$minio_access_key
S3_SECRET_KEY=$minio_secret_key
S3_BUCKET=$s3_bucket
S3_REGION=us-east-1
S3_SECURE=false
API_URL=http://$node:$api_port
EOF

echo "Submitted control-plane job id=$job_id"
echo "State: $state"
echo "Node: $node"
echo "API_URL=http://$node:$api_port"
echo "Endpoints file: $state_root/endpoints.env"
REMOTE
