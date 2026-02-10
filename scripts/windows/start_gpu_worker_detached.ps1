param(
  [Parameter(Mandatory = $true)]
  [string]$MacIp
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$runScript = Join-Path $repoRoot "scripts\\windows\\run_gpu_worker.ps1"
$outLog = Join-Path $repoRoot "gpu_worker.out.log"
$errLog = Join-Path $repoRoot "gpu_worker.err.log"

if (-not $MacIp) {
  throw "MacIp is required."
}
if (-not (Test-Path $runScript)) {
  throw "Missing script: $runScript"
}

Write-Host "== Start GPU Worker (detached) ==" -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot"
Write-Host "Mac IP:    $MacIp"
Write-Host "Stdout:    $outLog"
Write-Host "Stderr:    $errLog"
Write-Host ""

# NOTE:
# When invoked over Windows OpenSSH, child processes may get terminated when the SSH session exits
# (Job object semantics). Using `cmd.exe /c start ...` is more reliable at truly detaching.
#
# We do redirection at the CMD level so we still capture errors if PowerShell fails to parse/launch.
$cmdLine = @(
  "start `"han-gpu-worker`" /min cmd.exe /c",
  "`"powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runScript`" -MacIp `"$MacIp`" 1>> `"$outLog`" 2>> `"$errLog`"`""
) -join " "
cmd.exe /c $cmdLine | Out-Null

Write-Host "Started. Tail logs with:" -ForegroundColor Green
Write-Host "  Get-Content -Path `"$outLog`" -Wait"
Write-Host "  Get-Content -Path `"$errLog`" -Wait"
