param(
  [Parameter(Mandatory = $true)]
  [string]$IsaacSimPath,

  [string]$IsaacLabRepo = "https://github.com/isaac-sim/IsaacLab.git",
  [string]$IsaacLabRef = "main",

  [ValidateSet("none", "all")]
  [string]$RlFrameworks = "none",

  # Isaac Lab's mimic extension pulls in Jupyter labextensions and can hit Windows path-length limits.
  # Keep it off by default. Enable only after confirming long paths are enabled and the repo path is short.
  [switch]$IncludeMimic
)

$ErrorActionPreference = "Stop"

function Assert-Command($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $name"
  }
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string[]]$Args
  )
  & $Exe @Args
  if ($LASTEXITCODE -ne 0) {
    $joined = ($Args -join " ")
    throw "Command failed (exit=$LASTEXITCODE): $Exe $joined"
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

Write-Host "-- Path Length Check (Windows) --" -ForegroundColor Cyan
try {
  $lp = (Get-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" -Name "LongPathsEnabled" -ErrorAction Stop).LongPathsEnabled
  if ($lp -ne 1) {
    Write-Host "[WARN] Windows long paths are NOT enabled (LongPathsEnabled=$lp)." -ForegroundColor Yellow
    Write-Host "If installs fail with path errors, enable long paths or move this repo to a shorter path like C:\\src\\han-platform-2.0." -ForegroundColor Yellow
  }
} catch {
  Write-Host "[WARN] Could not read LongPathsEnabled from registry." -ForegroundColor Yellow
}
if ($repoRoot.Path -match "OneDrive") {
  Write-Host "[WARN] Repo is under OneDrive. This increases path length and can break installs." -ForegroundColor Yellow
  Write-Host "Recommended: clone to C:\\src\\han-platform-2.0 for stability." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "-- USD Python (pxr) Path Setup --" -ForegroundColor Cyan
# Isaac Sim ships the USD (pxr) Python package as an extension under extscache. When running plain
# `python.bat` (used by isaaclab.bat -p), pxr is not on PYTHONPATH/PATH by default.
$extCacheDir = Join-Path $IsaacSimPath "extscache"
if (Test-Path $extCacheDir) {
  $usdLib = Get-ChildItem -Path $extCacheDir -Directory -Filter "omni.usd.libs-*" | Sort-Object Name -Descending | Select-Object -First 1
  if ($usdLib) {
    if ($env:PYTHONPATH) { $env:PYTHONPATH = "$env:PYTHONPATH;$($usdLib.FullName)" } else { $env:PYTHONPATH = "$($usdLib.FullName)" }
    $usdBin = Join-Path $usdLib.FullName "bin"
    if (Test-Path $usdBin) { $env:PATH = "$env:PATH;$usdBin" }
    Write-Host "Added to PYTHONPATH: $($usdLib.FullName)"
    if (Test-Path $usdBin) { Write-Host "Added to PATH: $usdBin" }
  } else {
    Write-Host "[WARN] omni.usd.libs-* not found under $extCacheDir. pxr imports may fail until Isaac Sim has populated extensions." -ForegroundColor Yellow
  }
} else {
  Write-Host "[WARN] Isaac Sim extscache not found at: $extCacheDir. pxr imports may fail." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "-- Upgrading pip/setuptools/wheel in Isaac Sim Python --" -ForegroundColor Cyan
# Newer setuptools versions have removed pkg_resources, which breaks some source builds (for example flatdict).
# Pin to a known-good setuptools that still ships pkg_resources.
Invoke-Checked $isaacLabBat @("-p", "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools==81.0.0")

Write-Host "-- Ensuring PyTorch CUDA build (cu128) --" -ForegroundColor Cyan
Invoke-Checked $isaacLabBat @(
  "-p", "-m", "pip", "install",
  "--index-url", "https://download.pytorch.org/whl/cu128",
  "torch==2.7.0", "torchvision==0.22.0"
)

Write-Host "-- Build Helper Deps (avoid PEP517 build isolation pitfalls) --" -ForegroundColor Cyan
# flatdict publishes an sdist; without --no-build-isolation pip may create a build env with a setuptools that
# doesn't include pkg_resources, causing the install to fail.
Invoke-Checked $isaacLabBat @("-p", "-m", "pip", "install", "--no-build-isolation", "flatdict==4.0.1")

Write-Host "-- Installing Isaac Lab Python packages (minimal set) --" -ForegroundColor Cyan
$pkgs = @("isaaclab", "isaaclab_assets", "isaaclab_contrib", "isaaclab_tasks", "isaaclab_rl")
if ($IncludeMimic) {
  $pkgs += "isaaclab_mimic"
}
foreach ($pkg in $pkgs) {
  $pkgPath = Join-Path $isaacLabDir ("source\\" + $pkg)
  $setupPy = Join-Path $pkgPath "setup.py"
  if (-not (Test-Path $setupPy)) {
    Write-Host "[WARN] Skipping $pkg (no setup.py at $setupPy)" -ForegroundColor Yellow
    continue
  }
  if ($pkg -eq "isaaclab_rl" -and $RlFrameworks -eq "all") {
    # Install RL frameworks as extras only when requested.
    $spec = "$pkgPath" + "[all]"
    Write-Host "Installing: $spec"
    Invoke-Checked $isaacLabBat @("-p", "-m", "pip", "install", "--no-build-isolation", "--editable", $spec)
  } else {
    Write-Host "Installing: $pkgPath"
    Invoke-Checked $isaacLabBat @("-p", "-m", "pip", "install", "--no-build-isolation", "--editable", $pkgPath)
  }
}

Write-Host "-- Smoke Test: Imports --" -ForegroundColor Cyan
$code = "import sys; print(sys.version); import isaacsim; print('isaacsim ok'); import isaaclab; print('isaaclab ok'); import isaaclab_rl; print('isaaclab_rl ok'); import isaaclab_tasks; print('isaaclab_tasks ok')"
Invoke-Checked $isaacLabBat @("-p", "-c", $code)

Write-Host ""
Write-Host "OK: Isaac Lab bootstrapped." -ForegroundColor Green
Write-Host "Next: run scripts\\windows\\run_gpu_worker.ps1 after setting <MAC_LAN_IP> env vars."
