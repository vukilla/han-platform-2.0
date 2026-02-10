param(
  [Parameter(Mandatory = $true)]
  [string]$MacIp,

  [string]$IsaacSimPath = "C:\\isaacsim",

  # Also bootstrap GVHMR + its Windows deps so XGen can run `pose_estimator=gvhmr`.
  [switch]$SetupGVHMR,

  # Optionally download the two direct-link checkpoints (gvhmr_siga24_release.ckpt + yolov8x.pt).
  [switch]$DownloadLightCheckpoints,

  # Best-effort attempt to download the remaining heavy checkpoints from GVHMR's public Google Drive folder.
  [switch]$TryDownloadHeavyCheckpoints,

  # Celery queues to consume. For GVHMR-only runs, use "pose".
  # For full training/inference, use "gpu" (or "gpu,pose" to consume both).
  [string]$Queues = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$oneClick = Join-Path $repoRoot "scripts\\windows\\one_click_gpu_worker.ps1"
$bootstrapGvhmr = Join-Path $repoRoot "scripts\\windows\\bootstrap_gvhmr.ps1"

Write-Host "== One-click REAL GPU worker (Windows) ==" -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot"
Write-Host "Mac IP:    $MacIp"
Write-Host "Isaac Sim: $IsaacSimPath"
Write-Host ""

if (-not (Test-Path $oneClick)) {
  throw "Missing script: $oneClick"
}

if ($SetupGVHMR) {
  if (-not (Test-Path $bootstrapGvhmr)) {
    throw "Missing script: $bootstrapGvhmr"
  }
  Write-Host "-- Bootstrapping GVHMR (Windows) --" -ForegroundColor Cyan
  if ($DownloadLightCheckpoints) {
    if ($TryDownloadHeavyCheckpoints) {
      powershell -NoProfile -ExecutionPolicy Bypass -File $bootstrapGvhmr -DownloadLightCheckpoints -TryDownloadHeavyCheckpoints
    } else {
      powershell -NoProfile -ExecutionPolicy Bypass -File $bootstrapGvhmr -DownloadLightCheckpoints
    }
  } else {
    if ($TryDownloadHeavyCheckpoints) {
      powershell -NoProfile -ExecutionPolicy Bypass -File $bootstrapGvhmr -TryDownloadHeavyCheckpoints
    } else {
      powershell -NoProfile -ExecutionPolicy Bypass -File $bootstrapGvhmr
    }
  }
  Write-Host ""
}

if (-not $Queues) {
  # If we're setting up GVHMR, default to the pose queue so the worker doesn't crash-loop
  # on unrelated training jobs while the user is focusing on pose previews.
  $Queues = if ($SetupGVHMR) { "pose" } else { "gpu" }
}
$Queues = $Queues.Trim()
if (-not $Queues) {
  $Queues = if ($SetupGVHMR) { "pose" } else { "gpu" }
}

Write-Host "-- Bootstrapping + starting GPU worker --" -ForegroundColor Cyan
powershell -NoProfile -ExecutionPolicy Bypass -File $oneClick -MacIp $MacIp -IsaacSimPath $IsaacSimPath -Queues $Queues

Write-Host ""
Write-Host "Next (Mac): run the REAL smoke test:" -ForegroundColor Green
Write-Host "  ./scripts/smoke_e2e_with_gpu_real.sh"
