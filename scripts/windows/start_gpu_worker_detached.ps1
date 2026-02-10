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
# (Job object semantics). `cmd.exe /c start ...` is *sometimes* reliable, but can hang under SSH on
# some systems. A per-user Scheduled Task is the most reliable way to start a long-running worker
# and return control to the caller immediately.

$taskName = "han-gpu-worker"

function Start-WorkerScheduledTask() {
  $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runScript`" -MacIp `"$MacIp`" 1>> `"$outLog`" 2>> `"$errLog`""

  # Create/update a per-user scheduled task that runs only when the user is logged on.
  # This avoids storing a password and avoids OpenSSH job-object teardown killing the worker.
  $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $cmd"
  $trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddYears(10))
  $principal = $null
  foreach ($uid in @(
    "$env:COMPUTERNAME\\$env:USERNAME",
    "$env:USERDOMAIN\\$env:USERNAME",
    "$env:USERNAME"
  )) {
    if ($uid) {
      try {
        $principal = New-ScheduledTaskPrincipal -UserId $uid -LogonType Interactive -RunLevel Limited
        break
      } catch {
        $principal = $null
      }
    }
  }
  if (-not $principal) {
    throw "Unable to resolve current user for Scheduled Task principal."
  }
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden

  try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
  } catch {
    # ignore
  }

  Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
  Start-ScheduledTask -TaskName $taskName
}

try {
  Start-WorkerScheduledTask
} catch {
  Write-Host "[WARN] Scheduled Task start failed ($($_.Exception.Message)). Falling back to cmd.exe start." -ForegroundColor Yellow
  # Fallback:
  # Use `cmd.exe /c start ...` and do redirection at the CMD level so we still capture errors.
  $cmdLine = @(
    "start `"han-gpu-worker`" /min cmd.exe /c",
    "`"powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runScript`" -MacIp `"$MacIp`" 1>> `"$outLog`" 2>> `"$errLog`"`""
  ) -join " "
  cmd.exe /c $cmdLine | Out-Null
}

Write-Host "Started. Tail logs with:" -ForegroundColor Green
Write-Host "  Get-Content -Path `"$outLog`" -Wait"
Write-Host "  Get-Content -Path `"$errLog`" -Wait"
