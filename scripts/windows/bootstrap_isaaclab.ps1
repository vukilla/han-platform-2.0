param(
  [Parameter(Mandatory = $true)]
  [string]$IsaacSimPath,

  [string]$IsaacLabRepo = "https://github.com/isaac-sim/IsaacLab.git",
  [string]$IsaacLabRef = "main",

  [ValidateSet("none", "all")]
  [string]$RlFrameworks = "none"
)

$ErrorActionPreference = "Stop"

function Assert-Command($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $name"
  }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$externalDir = Join-Path $repoRoot "external"
$isaacLabDir = Join-Path $externalDir "isaaclab"

Write-Host "== Isaac Lab Bootstrap (Windows) ==" -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot"
Write-Host "Isaac Sim: $IsaacSimPath"
Write-Host "Isaac Lab: $isaacLabDir"
Write-Host ""

if (-not (Test-Path $IsaacSimPath)) {
  throw "Isaac Sim path does not exist: $IsaacSimPath"
}
if (-not (Test-Path (Join-Path $IsaacSimPath "isaac-sim.bat"))) {
  Write-Host "[WARN] Isaac Sim path exists but isaac-sim.bat not found. Expected: $(Join-Path $IsaacSimPath 'isaac-sim.bat')" -ForegroundColor Yellow
}

Assert-Command git

New-Item -ItemType Directory -Force -Path $externalDir | Out-Null

if (-not (Test-Path $isaacLabDir)) {
  Write-Host "-- Cloning Isaac Lab --" -ForegroundColor Cyan
  git clone --depth 1 --branch $IsaacLabRef $IsaacLabRepo $isaacLabDir
} else {
  Write-Host "-- Isaac Lab already present (skipping clone) --" -ForegroundColor Cyan
}

$symlinkPath = Join-Path $isaacLabDir "_isaac_sim"
if (-not (Test-Path $symlinkPath)) {
  Write-Host "-- Creating _isaac_sim link --" -ForegroundColor Cyan
  Write-Host "Link: $symlinkPath -> $IsaacSimPath"
  try {
    cmd /c "mklink /D `"$symlinkPath`" `"$IsaacSimPath`""
  } catch {
    Write-Host "[ERROR] Failed to create symlink. Fix options:" -ForegroundColor Red
    Write-Host "1) Enable Developer Mode (Settings -> For developers -> Developer Mode)" -ForegroundColor Red
    Write-Host "2) Re-run PowerShell as Administrator" -ForegroundColor Red
    throw
  }
} else {
  Write-Host "-- _isaac_sim already exists (skipping) --" -ForegroundColor Cyan
}

$isaacLabBat = Join-Path $isaacLabDir "isaaclab.bat"
if (-not (Test-Path $isaacLabBat)) {
  throw "isaaclab.bat not found at: $isaacLabBat"
}

Write-Host "-- Installing Isaac Lab extensions (this can take a while) --" -ForegroundColor Cyan
Write-Host "Command: $isaacLabBat -i $RlFrameworks"
cmd /c "`"$isaacLabBat`" -i $RlFrameworks"

Write-Host "-- Smoke Test: Imports --" -ForegroundColor Cyan
cmd /c "`"$isaacLabBat`" -p -c \"import sys; print(sys.version); import isaacsim; print('isaacsim ok'); import isaaclab_rl; print('isaaclab_rl ok'); import isaaclab_tasks; print('isaaclab_tasks ok')\""

Write-Host ""
Write-Host "OK: Isaac Lab bootstrapped." -ForegroundColor Green
Write-Host "Next: run scripts\\windows\\run_gpu_worker.ps1 after setting <MAC_LAN_IP> env vars."

