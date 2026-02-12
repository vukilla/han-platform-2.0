#!/usr/bin/env bash
set -euo pipefail

# Start a Linux worker on Pegasus over SSH.
#
# Usage:
#   PEGASUS_HOST=dfki ./scripts/mac/start_pegasus_worker_ssh.sh
#   ./scripts/mac/start_pegasus_worker_ssh.sh dfki
#
# Optional env vars:
# - SSH_USER (optional, if already configured in ~/.ssh/config leave empty)
# - SSH_KEY (default: ~/.ssh/dfki_pegasus if present)
# - PEGASUS_REPO (default: ~/han-platform)
# - PULL_REPO (default: 1)
# - INSTALL_REQS (default: 0)
# - PEGASUS_LAUNCH_MODE (default: auto, options: auto|slurm|local)
# - HAN_PYTHON_BIN (default: python3)
# - HAN_WORKER_QUEUES (default: pose)
# - HAN_WORKER_POOL (default: solo)
# - HAN_WORKER_CONCURRENCY (default: 1)
# - SLURM_PARTITION (default: RTXA6000)
# - SLURM_TIME (default: 12:00:00)
# - SLURM_GRES (default: gpu:1)
# - SLURM_CPUS_PER_TASK (optional)
# - SLURM_MEM (optional, for example 32G)
# - REDIS_URL / DATABASE_URL / S3_ENDPOINT / S3_ACCESS_KEY / S3_SECRET_KEY / S3_BUCKET
# - S3_REGION (default: us-east-1)
# - S3_SECURE (default: false)
# - GVHMR_NATIVE_RENDER / GVHMR_RENDER_DEVICE / GVHMR_RENDER_INCAM / GVHMR_RENDER_EXTRA_VIEWS

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
if [[ -z "$SSH_KEY" && -f "$HOME/.ssh/dfki_pegasus" ]]; then
  SSH_KEY="$HOME/.ssh/dfki_pegasus"
fi
PEGASUS_REPO="${PEGASUS_REPO:-~/han-platform}"
PULL_REPO="${PULL_REPO:-1}"
INSTALL_REQS="${INSTALL_REQS:-0}"
PEGASUS_LAUNCH_MODE="${PEGASUS_LAUNCH_MODE:-auto}"

HAN_PYTHON_BIN="${HAN_PYTHON_BIN:-}"
HAN_WORKER_QUEUES="${HAN_WORKER_QUEUES:-pose}"
HAN_WORKER_POOL="${HAN_WORKER_POOL:-solo}"
HAN_WORKER_CONCURRENCY="${HAN_WORKER_CONCURRENCY:-1}"

SLURM_PARTITION="${SLURM_PARTITION:-RTXA6000}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_GRES="${SLURM_GRES:-gpu:1}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-}"
SLURM_MEM="${SLURM_MEM:-}"

if [[ -z "${REDIS_URL:-}" || -z "${DATABASE_URL:-}" || -z "${S3_ENDPOINT:-}" ]]; then
  mac_ip="${MAC_LAN_IP:-$(ipconfig getifaddr en0 || true)}"
  if [[ -z "$mac_ip" ]]; then
    mac_ip="$(ipconfig getifaddr en1 || true)"
  fi
  if [[ -n "$mac_ip" ]]; then
    REDIS_URL="${REDIS_URL:-redis://${mac_ip}:6379/0}"
    DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://han:han@${mac_ip}:5432/han}"
    S3_ENDPOINT="${S3_ENDPOINT:-http://${mac_ip}:9000}"
  fi
fi

REDIS_URL="${REDIS_URL:-}"
DATABASE_URL="${DATABASE_URL:-}"
S3_ENDPOINT="${S3_ENDPOINT:-}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"
S3_BUCKET="${S3_BUCKET:-humanoid-network-dev}"
S3_REGION="${S3_REGION:-us-east-1}"
S3_SECURE="${S3_SECURE:-false}"

GVHMR_NATIVE_RENDER="${GVHMR_NATIVE_RENDER:-1}"
GVHMR_RENDER_DEVICE="${GVHMR_RENDER_DEVICE:-cuda}"
GVHMR_RENDER_INCAM="${GVHMR_RENDER_INCAM:-1}"
GVHMR_RENDER_EXTRA_VIEWS="${GVHMR_RENDER_EXTRA_VIEWS:-1}"

for key in REDIS_URL DATABASE_URL S3_ENDPOINT S3_ACCESS_KEY S3_SECRET_KEY S3_BUCKET; do
  if [[ -z "${!key:-}" ]]; then
    echo "[ERROR] Missing required env var: $key" >&2
    exit 1
  fi
done

PID_FILE="${PEGASUS_REPO}/tmp/pegasus_worker.pid"
JOB_ID_FILE="${PEGASUS_REPO}/tmp/pegasus_worker.slurm_job_id"
OUT_LOG="${PEGASUS_REPO}/tmp/pegasus_worker.out.log"
ERR_LOG="${PEGASUS_REPO}/tmp/pegasus_worker.err.log"

echo "== Start Pegasus Worker over SSH =="
echo "Repo:         $ROOT_DIR"
echo "Pegasus host: $PEGASUS_HOST"
if [[ -n "$SSH_USER" ]]; then
  echo "SSH user:     $SSH_USER"
else
  echo "SSH user:     <from ssh config>"
fi
if [[ -n "$SSH_KEY" ]]; then
  echo "SSH key:      $SSH_KEY"
else
  echo "SSH key:      <from ssh config/agent>"
fi
echo "Remote repo:  $PEGASUS_REPO"
echo "Queues:       $HAN_WORKER_QUEUES"
if [[ -n "$HAN_PYTHON_BIN" ]]; then
  echo "Python bin:   $HAN_PYTHON_BIN"
else
  echo "Python bin:   <auto (repo .venv preferred)>"
fi
echo "Launch mode:  $PEGASUS_LAUNCH_MODE"
if [[ "$PEGASUS_LAUNCH_MODE" != "local" ]]; then
  echo "Partition:    $SLURM_PARTITION"
  echo "Walltime:     $SLURM_TIME"
fi
echo "Log out:      $OUT_LOG"
echo "Log err:      $ERR_LOG"
echo ""

ssh_target="$PEGASUS_HOST"
if [[ -n "$SSH_USER" && "$ssh_target" != *@* ]]; then
  ssh_target="${SSH_USER}@${ssh_target}"
fi

ssh_cmd=(ssh)
if [[ -n "$SSH_KEY" ]]; then
  ssh_cmd+=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey)
fi

"${ssh_cmd[@]}" "$ssh_target" \
  "bash -s -- $(printf '%q ' "$PEGASUS_REPO" "$PULL_REPO" "$INSTALL_REQS" "$PEGASUS_LAUNCH_MODE" "$SLURM_PARTITION" "$SLURM_TIME" "$SLURM_GRES" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM" "$HAN_PYTHON_BIN" "$HAN_WORKER_QUEUES" "$HAN_WORKER_POOL" "$HAN_WORKER_CONCURRENCY" "$REDIS_URL" "$DATABASE_URL" "$S3_ENDPOINT" "$S3_ACCESS_KEY" "$S3_SECRET_KEY" "$S3_BUCKET" "$S3_REGION" "$S3_SECURE" "$GVHMR_NATIVE_RENDER" "$GVHMR_RENDER_DEVICE" "$GVHMR_RENDER_INCAM" "$GVHMR_RENDER_EXTRA_VIEWS" "$PID_FILE" "$JOB_ID_FILE" "$OUT_LOG" "$ERR_LOG")" <<'REMOTE'
set -euo pipefail

remote_repo="$1"
pull_repo="$2"
install_reqs="$3"
launch_mode="$4"
slurm_partition="$5"
slurm_time="$6"
slurm_gres="$7"
slurm_cpus_per_task="$8"
slurm_mem="$9"
python_bin="${10}"
worker_queues="${11}"
worker_pool="${12}"
worker_concurrency="${13}"
redis_url="${14}"
database_url="${15}"
s3_endpoint="${16}"
s3_access_key="${17}"
s3_secret_key="${18}"
s3_bucket="${19}"
s3_region="${20}"
s3_secure="${21}"
gvhmr_native_render="${22}"
gvhmr_render_device="${23}"
gvhmr_render_incam="${24}"
gvhmr_render_extra_views="${25}"
pid_file="${26}"
job_id_file="${27}"
out_log="${28}"
err_log="${29}"

mkdir -p "$remote_repo/tmp"

if [[ "$pull_repo" == "1" ]]; then
  git -C "$remote_repo" pull --ff-only
fi

if [[ -f "$job_id_file" ]]; then
  old_job_id="$(cat "$job_id_file" || true)"
  if [[ -n "$old_job_id" ]] && command -v scancel >/dev/null 2>&1; then
    scancel "$old_job_id" >/dev/null 2>&1 || true
  fi
  rm -f "$job_id_file"
fi

if [[ -f "$pid_file" ]]; then
  old_pid="$(cat "$pid_file" || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
    kill "$old_pid" >/dev/null 2>&1 || true
    sleep 1
  fi
  rm -f "$pid_file"
fi

cd "$remote_repo"
export REDIS_URL="$redis_url"
export DATABASE_URL="$database_url"
export S3_ENDPOINT="$s3_endpoint"
export S3_ACCESS_KEY="$s3_access_key"
export S3_SECRET_KEY="$s3_secret_key"
export S3_BUCKET="$s3_bucket"
export S3_REGION="$s3_region"
export S3_SECURE="$s3_secure"
export HAN_PYTHON_BIN="$python_bin"
export HAN_WORKER_QUEUES="$worker_queues"
export HAN_WORKER_POOL="$worker_pool"
export HAN_WORKER_CONCURRENCY="$worker_concurrency"
export INSTALL_REQS="$install_reqs"
export GVHMR_NATIVE_RENDER="$gvhmr_native_render"
export GVHMR_RENDER_DEVICE="$gvhmr_render_device"
export GVHMR_RENDER_INCAM="$gvhmr_render_incam"
export GVHMR_RENDER_EXTRA_VIEWS="$gvhmr_render_extra_views"

if [[ "$launch_mode" == "auto" ]]; then
  if command -v sbatch >/dev/null 2>&1; then
    launch_mode="slurm"
  else
    launch_mode="local"
  fi
fi

if [[ "$launch_mode" == "slurm" ]]; then
  env_file="$remote_repo/tmp/pegasus_worker.env.sh"
  batch_file="$remote_repo/tmp/pegasus_worker.sbatch"

  sq() {
    printf "%s" "$1" | sed "s/'/'\"'\"'/g"
  }

  cat >"$env_file" <<EOF
export REDIS_URL='$(sq "$redis_url")'
export DATABASE_URL='$(sq "$database_url")'
export S3_ENDPOINT='$(sq "$s3_endpoint")'
export S3_ACCESS_KEY='$(sq "$s3_access_key")'
export S3_SECRET_KEY='$(sq "$s3_secret_key")'
export S3_BUCKET='$(sq "$s3_bucket")'
export S3_REGION='$(sq "$s3_region")'
export S3_SECURE='$(sq "$s3_secure")'
export HAN_PYTHON_BIN='$(sq "$python_bin")'
export HAN_WORKER_QUEUES='$(sq "$worker_queues")'
export HAN_WORKER_POOL='$(sq "$worker_pool")'
export HAN_WORKER_CONCURRENCY='$(sq "$worker_concurrency")'
export INSTALL_REQS='$(sq "$install_reqs")'
export GVHMR_NATIVE_RENDER='$(sq "$gvhmr_native_render")'
export GVHMR_RENDER_DEVICE='$(sq "$gvhmr_render_device")'
export GVHMR_RENDER_INCAM='$(sq "$gvhmr_render_incam")'
export GVHMR_RENDER_EXTRA_VIEWS='$(sq "$gvhmr_render_extra_views")'
EOF

  cat >"$batch_file" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$1"
bash "$2/scripts/linux/run_worker.sh"
EOF
  chmod +x "$batch_file"

  sbatch_args=(
    --parsable
    --job-name "han-worker-${worker_queues//,/+}"
    --partition "$slurm_partition"
    --time "$slurm_time"
    --gres "$slurm_gres"
    --output "$out_log"
    --error "$err_log"
  )

  if [[ -n "$slurm_cpus_per_task" ]]; then
    sbatch_args+=(--cpus-per-task "$slurm_cpus_per_task")
  fi
  if [[ -n "$slurm_mem" ]]; then
    sbatch_args+=(--mem "$slurm_mem")
  fi

  job_id="$(sbatch "${sbatch_args[@]}" "$batch_file" "$env_file" "$remote_repo")"
  echo "$job_id" > "$job_id_file"
  sleep 2
  state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)"
  if [[ -z "$state" ]]; then
    echo "[WARN] Submitted Slurm job $job_id but could not read state (it may have started/finished quickly)."
  else
    echo "Submitted Slurm worker job id=$job_id state=$state"
  fi
  echo "Job ID file: $job_id_file"
  echo "Out log:     $out_log"
  echo "Err log:     $err_log"
else
  nohup bash "$remote_repo/scripts/linux/run_worker.sh" >"$out_log" 2>"$err_log" < /dev/null &
  new_pid="$!"
  echo "$new_pid" > "$pid_file"
  sleep 2

  if ! kill -0 "$new_pid" >/dev/null 2>&1; then
    echo "[ERROR] Worker failed to start. Inspect logs:"
    echo "  $out_log"
    echo "  $err_log"
    exit 1
  fi

  echo "Started Pegasus worker PID=$new_pid"
  echo "PID file: $pid_file"
  echo "Out log:  $out_log"
  echo "Err log:  $err_log"
fi
REMOTE
