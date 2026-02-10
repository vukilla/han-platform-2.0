#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

API_URL="${API_URL:-http://localhost:8000}"
WEB_URL="${WEB_URL:-http://localhost:3000}"
EMAIL="${EMAIL:-smoke@example.com}"
NAME="${NAME:-Smoke User}"

DEFAULT_VIDEO="$ROOT_DIR/assets/sample_videos/cargo_pickup_01.mp4"
if [[ -f "/Users/robertvukosa/Desktop/delivery-man-carry-a-goods-package-and-send-to-customer-at-home-for-impress-of-good-service.mp4" ]]; then
  DEFAULT_VIDEO="/Users/robertvukosa/Desktop/delivery-man-carry-a-goods-package-and-send-to-customer-at-home-for-impress-of-good-service.mp4"
fi

VIDEO_PATH="${1:-$DEFAULT_VIDEO}"

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

echo "== Check SMPL-X model (one-time setup) =="
SMPLX_JSON="$(curl -sS "$API_URL/admin/gvhmr/smplx-model" -H "$AUTH_HEADER")"
SMPLX_EXISTS="$(printf '%s' "$SMPLX_JSON" | json_get 'exists')"
if [[ "$SMPLX_EXISTS" != "True" && "$SMPLX_EXISTS" != "true" ]]; then
  cat <<EOF >&2

SMPL-X model missing.

Open $WEB_URL/studio and upload SMPLX_NEUTRAL.npz (one-time setup),
or upload via API: POST $API_URL/admin/gvhmr/smplx-model

EOF
  exit 1
fi

echo "== Create project =="
PROJECT_NAME="Smoke Motion Recovery $(date +%Y%m%d-%H%M%S)"
PROJECT_JSON="$(curl -sS -X POST "$API_URL/projects" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d "{\"name\":\"$PROJECT_NAME\",\"description\":\"motion recovery smoke\"}")"
PROJECT_ID="$(printf '%s' "$PROJECT_JSON" | json_get 'id')"

echo "== Create demo =="
DEMO_JSON="$(curl -sS -X POST "$API_URL/demos" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d "{\"project_id\":\"$PROJECT_ID\",\"robot_model\":\"human\",\"object_id\":\"none\"}")"
DEMO_ID="$(printf '%s' "$DEMO_JSON" | json_get 'id')"

echo "== Get upload URL =="
UPLOAD_JSON="$(curl -sS -X POST "$API_URL/demos/$DEMO_ID/upload-url")"
UPLOAD_URL="$(printf '%s' "$UPLOAD_JSON" | json_get 'upload_url')"
VIDEO_URI="$(printf '%s' "$UPLOAD_JSON" | json_get 'video_uri')"

echo "== Upload video =="
curl -sS -X PUT "$UPLOAD_URL" \
  -H "Content-Type: video/mp4" \
  --data-binary "@$VIDEO_PATH" >/dev/null

echo "== Start motion recovery job (pose queue) =="
JOB_JSON="$(curl -sS -X POST "$API_URL/demos/$DEMO_ID/xgen/run" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d "{\"params_json\":{\"video_uri\":\"$VIDEO_URI\",\"requires_gpu\":true,\"only_pose\":true,\"pose_estimator\":\"gvhmr\",\"gvhmr_static_cam\":true,\"gvhmr_max_seconds\":12,\"fail_on_pose_error\":true}}")"
JOB_ID="$(printf '%s' "$JOB_JSON" | json_get 'id')"

echo "== Poll job =="
while true; do
  JOB_JSON="$(curl -sS "$API_URL/xgen/jobs/$JOB_ID")"
  STATUS="$(printf '%s' "$JOB_JSON" | json_get 'status')"
  echo "  status=$STATUS"
  if [[ "$STATUS" == "COMPLETED" ]]; then
    break
  fi
  if [[ "$STATUS" == "FAILED" ]]; then
    echo "Job failed: $(printf '%s' "$JOB_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("error"))')" >&2
    exit 1
  fi
  sleep 2
done

POSE_OK="$(printf '%s' "$JOB_JSON" | python -c 'import json,sys; d=json.load(sys.stdin); print((d.get("params_json") or {}).get("pose_ok"))')"
PREVIEW_URL="$(printf '%s' "$JOB_JSON" | python -c 'import json,sys; d=json.load(sys.stdin); print((d.get("params_json") or {}).get("pose_preview_mp4_uri",""))')"

echo "== Verify preview size (best effort) =="
if command -v ffprobe >/dev/null 2>&1 && [[ -n "$PREVIEW_URL" ]]; then
  ORIG_WH="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$VIDEO_PATH" 2>/dev/null || true)"
  PREV_WH="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$PREVIEW_URL" 2>/dev/null || true)"
  echo "  original=$ORIG_WH"
  echo "  preview =$PREV_WH"
fi

cat <<EOF

OK
- Studio:    $WEB_URL/studio
- Job page:  $WEB_URL/jobs/$JOB_ID
- pose_ok:   $POSE_OK
- preview:   $PREVIEW_URL

EOF
