param(
  [string]$IsaacSimPath = $env:ISAACSIM_PATH
)

$ErrorActionPreference = "Stop"

Write-Host "== han-platform-2.0 Windows Preflight ==" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format o)"
Write-Host "PWD:  $(Get-Location)"
Write-Host ""

Write-Host "-- OS / PowerShell --"
Write-Host "OS:   $([System.Environment]::OSVersion.VersionString)"
Write-Host "PS:   $($PSVersionTable.PSVersion)"
Write-Host ""

Write-Host "-- GPU / Driver --"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  try {
    nvidia-smi
  } catch {
    Write-Host "[WARN] nvidia-smi exists but failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
} else {
  Write-Host "[WARN] nvidia-smi not found. Install/repair NVIDIA drivers." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "-- Tools --"
foreach ($cmd in @("git", "conda", "python", "pip")) {
  if (Get-Command $cmd -ErrorAction SilentlyContinue) {
    $path = (Get-Command $cmd).Source
    Write-Host "$cmd: OK ($path)"
  } else {
    Write-Host "$cmd: MISSING"
  }
}
Write-Host ""

Write-Host "-- Disk --"
try {
  $drive = Get-PSDrive -Name C -ErrorAction Stop
  $freeGB = [math]::Round($drive.Free / 1GB, 1)
  $usedGB = [math]::Round(($drive.Used) / 1GB, 1)
  Write-Host "C: free=${freeGB}GB used=${usedGB}GB"
} catch {
  Write-Host "[WARN] Unable to read C: drive info." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "-- Isaac Sim Path --"
if (-not $IsaacSimPath) {
  Write-Host "[INFO] ISAACSIM_PATH not set." -ForegroundColor Yellow
} else {
  Write-Host "ISAACSIM_PATH=$IsaacSimPath"
}

if ($IsaacSimPath -and (Test-Path $IsaacSimPath)) {
  $simBat = Join-Path $IsaacSimPath "isaac-sim.bat"
  if (Test-Path $simBat) {
    Write-Host "Found: $simBat"
  } else {
    Write-Host "[WARN] Isaac Sim path exists but isaac-sim.bat not found: $simBat" -ForegroundColor Yellow
  }
} else {
  Write-Host "[INFO] Isaac Sim directory not found. Expected something like C:\\isaacsim." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "-- Next Steps --" -ForegroundColor Cyan
Write-Host "1) Install Isaac Sim to C:\\isaacsim (or pass -IsaacSimPath)."
Write-Host "2) (Optional) Install Miniconda if you want a separate conda env for RL frameworks."
Write-Host "3) Run: scripts\\windows\\bootstrap_isaaclab.ps1 -IsaacSimPath C:\\isaacsim"
