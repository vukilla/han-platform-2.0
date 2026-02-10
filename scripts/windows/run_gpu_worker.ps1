param(
  [string]$MacIp = "",
  [string]$Queues = "gpu"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$isaacLabBat = Join-Path $repoRoot "external\\isaaclab\\isaaclab.bat"
$apiDir = Join-Path $repoRoot "apps\\api"
$workerReq = Join-Path $apiDir "requirements.worker.txt"

Write-Host "== han-platform-2.0 GPU Worker (Windows) ==" -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot"
Write-Host ""

if (-not $Queues) {
  $Queues = "gpu"
}
$Queues = $Queues.Trim()
if (-not $Queues) {
  $Queues = "gpu"
}

if ($MacIp) {
  $env:REDIS_URL = "redis://${MacIp}:6379/0"
  $env:DATABASE_URL = "postgresql+psycopg://han:han@${MacIp}:5432/han"
  $env:S3_ENDPOINT = "http://${MacIp}:9000"
  $env:S3_ACCESS_KEY = "minioadmin"
  $env:S3_SECRET_KEY = "minioadmin"
  $env:S3_BUCKET = "humanx-dev"
}

$env:HAN_WORKER_QUEUES = $Queues

# Heartbeat role is used by the Mac control-plane to detect which worker is online.
# If we are only consuming the `pose` queue (GVHMR-only), advertise role=pose.
$queueTokens = @()
foreach ($part in ($Queues -split ",")) {
  $t = ($part -as [string]).Trim().ToLower()
  if ($t) { $queueTokens += $t }
}
$role = "cpu"
if ($queueTokens -contains "gpu") {
  $role = "gpu"
} elseif ($queueTokens -contains "pose") {
  $role = "pose"
} elseif ($queueTokens.Count -gt 0) {
  $role = $queueTokens[0]
}
$env:HAN_WORKER_ROLE = $role

# GVHMR native mesh rendering:
# - When enabled, the GVHMR demo will produce `2_global.mp4` (and related videos) which the platform
#   can use as the 3D preview.
# - Default to CPU rendering for now on Blackwell GPUs (e.g. RTX 5090) because most prebuilt
#   PyTorch3D CUDA wheels lack sm_120 kernels; GVHMR will still render correctly on CPU.
if (-not $env:GVHMR_NATIVE_RENDER) {
  $env:GVHMR_NATIVE_RENDER = "1"
}
if (-not $env:GVHMR_RENDER_DEVICE) {
  $env:GVHMR_RENDER_DEVICE = "cpu"
}

foreach ($k in @("REDIS_URL", "DATABASE_URL", "S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET")) {
  $val = ""
  try {
    $val = (Get-Item -Path ("Env:" + $k) -ErrorAction Stop).Value
  } catch {
    $val = ""
  }
  if (-not $val) {
    throw "Missing required env var: $k"
  }
}

if (-not (Test-Path $isaacLabBat)) {
  throw "Missing Isaac Lab bootstrap. Expected: $isaacLabBat`nRun scripts\\windows\\bootstrap_isaaclab.ps1 first."
}
if (-not (Test-Path $apiDir)) {
  throw "API dir not found: $apiDir"
}

function Invoke-CmdChecked([string]$command) {
  cmd /c $command
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed (exit=$LASTEXITCODE): $command"
  }
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
  $reqFile = "requirements.txt"
  if (Test-Path $workerReq) {
    $reqFile = "requirements.worker.txt"
  }

  Write-Host "-- Installing worker requirements into Isaac Sim python --" -ForegroundColor Cyan
  Invoke-CmdChecked "`"$isaacLabBat`" -p -m pip install -r $reqFile"

  Write-Host "-- Starting Celery GPU worker (queue=$Queues) --" -ForegroundColor Cyan
  Write-Host "Stop with Ctrl+C. (Windows requires -P solo.)"
  Invoke-CmdChecked "`"$isaacLabBat`" -p -m celery -A app.worker.celery_app worker -l info -Q $Queues -P solo -c 1"
} finally {
  Pop-Location
}
