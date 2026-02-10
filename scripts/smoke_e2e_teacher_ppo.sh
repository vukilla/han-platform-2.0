#!/usr/bin/env bash
set -euo pipefail

# Smoke test: placeholder XGen (CPU) + real Isaac Lab teacher PPO (GPU, Windows worker).
#
# Requires:
# - Mac control-plane stack running (docker compose)
# - Windows GPU worker running (Celery queue=gpu via scripts/windows/one_click_gpu_worker.ps1)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

API_URL="${API_URL:-http://localhost:8000}"
WEB_URL="${WEB_URL:-http://localhost:3000}"
EMAIL="${EMAIL:-smoke@example.com}"
NAME="${NAME:-Smoke User}"
MODE="${MODE:-mocap}" # nep | mocap
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

VIDEO_PATH="${1:-$ROOT_DIR/assets/sample_videos/cargo_pickup_01.mp4}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found: $VIDEO_PATH" >&2
  exit 1
fi

json_get() {
  local path="$1"
  python -c 'import json,sys
data=json.load(sys.stdin)
cur=data
for part in sys.argv[1].split("."):
  if isinstance(cur, list):
    cur = cur[int(part)]
  else:
    cur = cur[part]
print(cur)
' "$path"
}

echo "== Login =="
LOGIN_JSON="$(curl -sS -X POST "$API_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"name\":\"$NAME\"}")"
TOKEN="$(printf '%s' "$LOGIN_JSON" | json_get 'token')"
AUTH_HEADER="Authorization: Bearer $TOKEN"

echo "== Create project =="
PROJECT_JSON="$(curl -sS -X POST "$API_URL/projects" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d "{\"name\":\"Smoke Project (teacher PPO)\",\"description\":\"placeholder xgen + isaaclab ppo\"}")"
PROJECT_ID="$(printf '%s' "$PROJECT_JSON" | json_get 'id')"

echo "== Create demo =="
DEMO_JSON="$(curl -sS -X POST "$API_URL/demos" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d "{\"project_id\":\"$PROJECT_ID\",\"robot_model\":\"unitree-g1\",\"object_id\":\"cargo_box\"}")"
DEMO_ID="$(printf '%s' "$DEMO_JSON" | json_get 'id')"

echo "== Get upload URL =="
UPLOAD_JSON="$(curl -sS -X POST "$API_URL/demos/$DEMO_ID/upload-url")"
UPLOAD_URL="$(printf '%s' "$UPLOAD_JSON" | json_get 'upload_url')"
VIDEO_URI="$(printf '%s' "$UPLOAD_JSON" | json_get 'video_uri')"

echo "== Upload video =="
curl -sS -X PUT "$UPLOAD_URL" \
  -H "Content-Type: video/mp4" \
  --data-binary "@$VIDEO_PATH" >/dev/null

echo "== Save annotations =="
curl -sS -X POST "$API_URL/demos/$DEMO_ID/annotations" \
  -H "Content-Type: application/json" \
  -d '{"ts_contact_start":0.5,"ts_contact_end":8.0,"anchor_type":"palms_midpoint","key_bodies":["left_hand","right_hand"]}' >/dev/null

echo "== Start XGen job (CPU, placeholder pose) =="
XGEN_JSON="$(curl -sS -X POST "$API_URL/demos/$DEMO_ID/xgen/run" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d "{\"params_json\":{\"video_uri\":\"$VIDEO_URI\",\"placeholder_pose\":true,\"clip_count\":2,\"frames\":30,\"nq\":12,\"contact_dim\":4}}")"
XGEN_JOB_ID="$(printf '%s' "$XGEN_JSON" | json_get 'id')"

echo "== Poll XGen job =="
start_ts="$(python -c 'import time; print(int(time.time()))')"
while true; do
  JOB_JSON="$(curl -sS "$API_URL/xgen/jobs/$XGEN_JOB_ID")"
  STATUS="$(printf '%s' "$JOB_JSON" | json_get 'status')"
  if [[ "$STATUS" == "COMPLETED" ]]; then
    break
  fi
  if [[ "$STATUS" == "FAILED" ]]; then
    echo "XGen failed: $(printf '%s' "$JOB_JSON" | json_get 'error')" >&2
    echo "Check logs: $WEB_URL/jobs/$XGEN_JOB_ID" >&2
    exit 1
  fi
  now_ts="$(python -c 'import time; print(int(time.time()))')"
  if (( now_ts - start_ts > TIMEOUT_SECONDS )); then
    echo "Timed out waiting for XGen. Check: $WEB_URL/jobs/$XGEN_JOB_ID" >&2
    exit 1
  fi
  sleep 1
done

DATASET_ID="$(printf '%s' "$JOB_JSON" | json_get 'params_json.dataset_id')"
if [[ -z "$DATASET_ID" ]]; then
  echo "Failed to locate dataset_id in XGen job params_json" >&2
  exit 1
fi

echo "== Start XMimic job (GPU, Isaac Lab teacher PPO) =="
XMIMIC_JSON="$(curl -sS -X POST "$API_URL/datasets/$DATASET_ID/xmimic/run" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d "{\"mode\":\"$MODE\",\"params_json\":{\"backend\":\"isaaclab_teacher_ppo\",\"env_task\":\"cargo_pickup_v0\",\"isaaclab_task\":\"cargo_pickup_franka\",\"num_envs\":8,\"updates\":2,\"rollout_steps\":64}}")"
XMIMIC_JOB_ID="$(printf '%s' "$XMIMIC_JSON" | json_get 'id')"

echo "== Poll XMimic job =="
start_ts="$(python -c 'import time; print(int(time.time()))')"
while true; do
  JOB_JSON="$(curl -sS "$API_URL/xmimic/jobs/$XMIMIC_JOB_ID")"
  STATUS="$(printf '%s' "$JOB_JSON" | json_get 'status')"
  if [[ "$STATUS" == "COMPLETED" ]]; then
    break
  fi
  if [[ "$STATUS" == "FAILED" ]]; then
    echo "XMimic failed: $(printf '%s' "$JOB_JSON" | json_get 'error')" >&2
    echo "Check logs: $WEB_URL/xmimic/$XMIMIC_JOB_ID" >&2
    exit 1
  fi
  now_ts="$(python -c 'import time; print(int(time.time()))')"
  if (( now_ts - start_ts > TIMEOUT_SECONDS )); then
    echo "Timed out waiting for XMimic. Ensure the Windows GPU worker is running." >&2
    echo "Check: $WEB_URL/xmimic/$XMIMIC_JOB_ID" >&2
    exit 1
  fi
  sleep 5
done

POLICIES_JSON="$(curl -sS "$API_URL/policies?xmimic_job_id=$XMIMIC_JOB_ID")"
POLICY_ID="$(printf '%s' "$POLICIES_JSON" | json_get '0.id')"
CHECKPOINT_URL="$(printf '%s' "$POLICIES_JSON" | json_get '0.checkpoint_uri')"

EVALS_JSON="$(curl -sS "$API_URL/eval?policy_id=$POLICY_ID")"
EVAL_ID="$(printf '%s' "$EVALS_JSON" | json_get '0.id')"

cat <<EOF

OK (Teacher PPO end-to-end)
- XGen job:            $WEB_URL/jobs/$XGEN_JOB_ID
- Dataset:             $WEB_URL/datasets/$DATASET_ID
- XMimic job:          $WEB_URL/xmimic/$XMIMIC_JOB_ID
- Policies:            $WEB_URL/policies
- Eval report:         $WEB_URL/eval/$EVAL_ID
- Checkpoint download: $CHECKPOINT_URL

EOF

