$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$isaacLabBat = Join-Path $repoRoot "external\\isaaclab\\isaaclab.bat"
$apiDir = Join-Path $repoRoot "apps\\api"

Write-Host "== han-platform-2.0 GPU Worker (Windows) ==" -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot"
Write-Host ""

foreach ($k in @("REDIS_URL", "DATABASE_URL", "S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET")) {
  if (-not $env:$k) {
    throw "Missing required env var: $k"
  }
}

if (-not (Test-Path $isaacLabBat)) {
  throw "Missing Isaac Lab bootstrap. Expected: $isaacLabBat`nRun scripts\\windows\\bootstrap_isaaclab.ps1 first."
}
if (-not (Test-Path $apiDir)) {
  throw "API dir not found: $apiDir"
}

Push-Location $apiDir
try {
  Write-Host "-- Installing API requirements into Isaac Sim python --" -ForegroundColor Cyan
  cmd /c "`"$isaacLabBat`" -p -m pip install -r requirements.txt"

  Write-Host "-- Starting Celery GPU worker (queue=gpu) --" -ForegroundColor Cyan
  Write-Host "Stop with Ctrl+C."
  cmd /c "`"$isaacLabBat`" -p -m celery -A app.worker.celery_app worker -l info -Q gpu -c 1"
} finally {
  Pop-Location
}

