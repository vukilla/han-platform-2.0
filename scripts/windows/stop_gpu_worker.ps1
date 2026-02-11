param()

$ErrorActionPreference = "Stop"

# Stops any running Celery worker processes launched for han-platform on Windows.
# We match by command line to avoid killing unrelated Python processes.

Write-Host "== Stop GPU Worker (Windows) ==" -ForegroundColor Cyan

$pattern = "celery -A app.worker.celery_app worker"
$procs = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and $_.CommandLine.Contains($pattern)
}

if (-not $procs) {
  Write-Host "No matching Celery worker processes found." -ForegroundColor Green
  exit 0
}

foreach ($p in $procs) {
  try {
    Write-Host "Stopping PID $($p.ProcessId): $($p.CommandLine)" -ForegroundColor Yellow
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
  } catch {
    Write-Host "[WARN] Failed to stop PID $($p.ProcessId): $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

Write-Host "Done." -ForegroundColor Green

