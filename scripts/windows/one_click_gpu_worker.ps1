param(
  [Parameter(Mandatory = $true)]
  [string]$MacIp,

  [string]$IsaacSimPath = "C:\\isaacsim"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$bootstrap = Join-Path $repoRoot "scripts\\windows\\bootstrap_isaaclab.ps1"
$startDetached = Join-Path $repoRoot "scripts\\windows\\start_gpu_worker_detached.ps1"
$isaacLabBat = Join-Path $repoRoot "external\\isaaclab\\isaaclab.bat"

Write-Host "== One-click GPU worker (Windows) ==" -ForegroundColor Cyan
Write-Host "Repo root:     $repoRoot"
Write-Host "Mac IP:        $MacIp"
Write-Host "Isaac Sim:     $IsaacSimPath"
Write-Host ""

if (-not $MacIp) {
  throw "MacIp is required."
}
if (-not (Test-Path $startDetached)) {
  throw "Missing script: $startDetached"
}

if (-not (Test-Path $isaacLabBat)) {
  if (-not (Test-Path $bootstrap)) {
    throw "Missing script: $bootstrap"
  }
  Write-Host "-- Bootstrapping Isaac Lab (first run) --" -ForegroundColor Cyan
  powershell -NoProfile -ExecutionPolicy Bypass -File $bootstrap -IsaacSimPath $IsaacSimPath
} else {
  Write-Host "-- Isaac Lab already bootstrapped (skipping) --" -ForegroundColor Cyan
}

Write-Host "-- Starting GPU worker (detached) --" -ForegroundColor Cyan
powershell -NoProfile -ExecutionPolicy Bypass -File $startDetached -MacIp $MacIp

Write-Host ""
Write-Host "Next (Mac): run the full smoke test:" -ForegroundColor Green
Write-Host "  ./scripts/smoke_e2e_with_gpu.sh"

