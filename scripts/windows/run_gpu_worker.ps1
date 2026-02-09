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

Write-Host "-- USD Python (pxr) Path Setup --" -ForegroundColor Cyan
$isaacSimLink = Join-Path $repoRoot "external\\isaaclab\\_isaac_sim"
try {
  if (Test-Path $isaacSimLink) {
    $target = (Get-Item $isaacSimLink).Target
    $isaacSimRoot = if ($target -is [System.Array]) { $target[0] } else { $target }
    $extCacheDir = Join-Path $isaacSimRoot "extscache"
    if (Test-Path $extCacheDir) {
      $usdLib = Get-ChildItem -Path $extCacheDir -Directory -Filter "omni.usd.libs-*" | Sort-Object Name -Descending | Select-Object -First 1
      if ($usdLib) {
        if ($env:PYTHONPATH) { $env:PYTHONPATH = "$env:PYTHONPATH;$($usdLib.FullName)" } else { $env:PYTHONPATH = "$($usdLib.FullName)" }
        $usdBin = Join-Path $usdLib.FullName "bin"
        if (Test-Path $usdBin) { $env:PATH = "$env:PATH;$usdBin" }
      }
    }
  }
} catch {
  Write-Host "[WARN] Unable to auto-configure pxr import path: $($_.Exception.Message)" -ForegroundColor Yellow
}

Push-Location $apiDir
try {
  Write-Host "-- Installing API requirements into Isaac Sim python --" -ForegroundColor Cyan
  $reqFile = "requirements.txt"
  if (Test-Path "requirements.worker.txt") {
    $reqFile = "requirements.worker.txt"
  }
  cmd /c "`"$isaacLabBat`" -p -m pip install -r $reqFile"

  Write-Host "-- Starting Celery GPU worker (queue=gpu) --" -ForegroundColor Cyan
  Write-Host "Stop with Ctrl+C."
  cmd /c "`"$isaacLabBat`" -p -m celery -A app.worker.celery_app worker -l info -Q gpu -c 1"
} finally {
  Pop-Location
}

