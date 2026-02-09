#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

API_URL="${API_URL:-http://localhost:8000}"
WEB_URL="${WEB_URL:-http://localhost:3000}"
EMAIL="${EMAIL:-smoke@example.com}"
NAME="${NAME:-Smoke User}"

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
  -d "{\"name\":\"Smoke Project\",\"description\":\"local e2e\"}")"
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

echo "== Start XGen job (CPU queue) =="
XGEN_JSON="$(curl -sS -X POST "$API_URL/demos/$DEMO_ID/xgen/run" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d "{\"params_json\":{\"video_uri\":\"$VIDEO_URI\",\"placeholder_pose\":true,\"clip_count\":3,\"frames\":40,\"nq\":12,\"contact_dim\":4}}")"
JOB_ID="$(printf '%s' "$XGEN_JSON" | json_get 'id')"

echo "== Poll XGen job =="
while true; do
  JOB_JSON="$(curl -sS "$API_URL/xgen/jobs/$JOB_ID")"
  STATUS="$(printf '%s' "$JOB_JSON" | json_get 'status')"
  if [[ "$STATUS" == "COMPLETED" ]]; then
    break
  fi
  if [[ "$STATUS" == "FAILED" ]]; then
    echo "XGen failed: $(printf '%s' "$JOB_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("error"))')" >&2
    exit 1
  fi
  sleep 1
done

DATASET_ID="$(printf '%s' "$JOB_JSON" | python -c 'import json,sys; d=json.load(sys.stdin); print((d.get("params_json") or {}).get("dataset_id",""))')"
if [[ -z "$DATASET_ID" ]]; then
  echo "Failed to locate dataset_id in XGen job params_json" >&2
  exit 1
fi

DOWNLOAD_JSON="$(curl -sS "$API_URL/datasets/$DATASET_ID/download-url")"
DOWNLOAD_URL="$(printf '%s' "$DOWNLOAD_JSON" | json_get 'download_url')"

cat <<EOF

OK
- Web demo wizard: $WEB_URL/demos/new
- XGen job:          $WEB_URL/jobs/$JOB_ID
- Dataset:           $WEB_URL/datasets/$DATASET_ID
- Dataset download:  $DOWNLOAD_URL

Next:
- Start XMimic from $WEB_URL/training (requires GPU worker)
- View policies at  $WEB_URL/policies
EOF
