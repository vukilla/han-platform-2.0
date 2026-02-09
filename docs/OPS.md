# Ops Monitoring

## Health check
`GET /health` returns DB/Redis/S3 connectivity plus queue depths.

## Failed jobs dashboard
`GET /ops/jobs/failed` returns failed/retrying XGen and XMimic jobs for admin UI.

## Alerting (optional)
Set `ALERT_WEBHOOK_URL` to receive JSON alerts when jobs enter `RETRYING` or `FAILED`.
