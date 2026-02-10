param(
  [string]$GvhmrRepo = "https://github.com/zju3dv/GVHMR.git",
  # Pinned commit known to work with our patch (`--skip_render` and optional deps).
  [string]$GvhmrRef = "088caff492aa38c2d82cea363b78a3c65a83118f",
  # Downloads ONLY the two files we can fetch from direct URLs:
  # - gvhmr_siga24_release.ckpt (HuggingFace)
  # - yolov8x.pt (Ultralytics release)
  # The remaining checkpoints are still required but may need manual download due to Google Drive UX/licensing.
  [switch]$DownloadLightCheckpoints,
  # Best-effort attempt to download the remaining heavy checkpoints from the public Google Drive folder.
  #
  # NOTE: This may still fail if Google requires interactive sign-in or license acknowledgement.
  # If it fails, use the manual steps described at the end of this script.
  [switch]$TryDownloadHeavyCheckpoints
)

$ErrorActionPreference = "Stop"

function Assert-Command($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $name"
  }
}

function Invoke-CmdChecked([string]$command) {
  cmd /c $command
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed (exit=$LASTEXITCODE): $command"
  }
}

function Download-File([string]$Url, [string]$OutFile) {
  $outDir = Split-Path -Parent $OutFile
  if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  }
  Write-Host "Downloading:" -ForegroundColor Cyan
  Write-Host "  $Url"
  Write-Host "To:" -ForegroundColor Cyan
  Write-Host "  $OutFile"
  Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$externalDir = Join-Path $repoRoot "external"
$gvhmrDir = Join-Path $externalDir "gvhmr"
$isaacLabBat = Join-Path $repoRoot "external\\isaaclab\\isaaclab.bat"
$patchFile = Join-Path $repoRoot "scripts\\gvhmr\\patches\\gvhmr_windows_compat.patch"
$stubSrc = Join-Path $repoRoot "scripts\\gvhmr\\pytorch3d"
$stubDst = Join-Path $gvhmrDir "pytorch3d"
$reqFile = Join-Path $repoRoot "scripts\\windows\\requirements_gvhmr_windows.txt"
$stagedCkptRoot = Join-Path $repoRoot "external\\humanoid-projects\\GVHMR\\inputs\\checkpoints"

Write-Host "== GVHMR Bootstrap (Windows) ==" -ForegroundColor Cyan
Write-Host "Repo root:   $repoRoot"
Write-Host "GVHMR dir:   $gvhmrDir"
Write-Host "Isaac Lab:   $isaacLabBat"
Write-Host "Patch file:  $patchFile"
Write-Host "Stub src:    $stubSrc"
Write-Host "Staged ckpt: $stagedCkptRoot"
Write-Host ""

Assert-Command git

New-Item -ItemType Directory -Force -Path $externalDir | Out-Null

if (-not (Test-Path $isaacLabBat)) {
  throw "Isaac Lab not bootstrapped. Expected: $isaacLabBat`nRun scripts\\windows\\bootstrap_isaaclab.ps1 first."
}
if (-not (Test-Path $patchFile)) {
  throw "Missing patch file: $patchFile"
}
if (-not (Test-Path $stubSrc)) {
  throw "Missing pytorch3d stub folder: $stubSrc"
}
if (-not (Test-Path $reqFile)) {
  throw "Missing requirements file: $reqFile"
}

if (-not (Test-Path $gvhmrDir)) {
  Write-Host "-- Cloning GVHMR --" -ForegroundColor Cyan
  git clone $GvhmrRepo $gvhmrDir
} else {
  Write-Host "-- GVHMR already present (skipping clone) --" -ForegroundColor Cyan
}

if ($GvhmrRef) {
  Write-Host "-- Checkout GVHMR ref: $GvhmrRef --" -ForegroundColor Cyan
  Push-Location $gvhmrDir
  try {
    git fetch --all --tags
    git checkout $GvhmrRef
  } finally {
    Pop-Location
  }
}

Write-Host "-- Applying Windows compatibility patch (idempotent) --" -ForegroundColor Cyan
# Reset the working tree so we can re-apply the patch even if the repo was patched previously.
# This keeps the bootstrap deterministic (we pin a ref and always apply our patch).
Invoke-CmdChecked "git -C `"$gvhmrDir`" reset --hard"
Invoke-CmdChecked "git -C `"$gvhmrDir`" apply `"$patchFile`""

Write-Host "-- Installing pytorch3d stub into GVHMR repo --" -ForegroundColor Cyan
if (Test-Path $stubDst) {
  Remove-Item -Recurse -Force $stubDst
}
Copy-Item -Path $stubSrc -Destination $stubDst -Recurse -Force

Write-Host "-- Installing GVHMR runtime deps into Isaac Sim python --" -ForegroundColor Cyan
Invoke-CmdChecked "`"$isaacLabBat`" -p -m pip install -r `"$reqFile`""

Write-Host ""
Write-Host "-- Checkpoints --" -ForegroundColor Cyan
Write-Host "GVHMR expects checkpoints under:" -ForegroundColor Cyan
Write-Host "  $stagedCkptRoot"
Write-Host "and the runner will symlink/copy into:" -ForegroundColor Cyan
Write-Host "  $gvhmrDir\\inputs\\checkpoints"
Write-Host ""

New-Item -ItemType Directory -Force -Path (Join-Path $stagedCkptRoot "gvhmr") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stagedCkptRoot "yolo") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stagedCkptRoot "vitpose") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stagedCkptRoot "hmr2") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stagedCkptRoot "dpvo") | Out-Null

if ($DownloadLightCheckpoints) {
  $gvhmrCkpt = Join-Path $stagedCkptRoot "gvhmr\\gvhmr_siga24_release.ckpt"
  if (-not (Test-Path $gvhmrCkpt)) {
    Download-File "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt?download=true" $gvhmrCkpt
  } else {
    Write-Host "OK: Found $gvhmrCkpt" -ForegroundColor Green
  }

  $yoloCkpt = Join-Path $stagedCkptRoot "yolo\\yolov8x.pt"
  if (-not (Test-Path $yoloCkpt)) {
    # Public Ultralytics release artifact. If this URL ever changes, download yolov8x.pt from Ultralytics and place it here.
    Download-File "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt" $yoloCkpt
  } else {
    Write-Host "OK: Found $yoloCkpt" -ForegroundColor Green
  }
}

if ($TryDownloadHeavyCheckpoints) {
  Write-Host ""
  Write-Host "-- Heavy checkpoint download (best-effort) --" -ForegroundColor Cyan
  Write-Host "Attempting to download remaining checkpoints from GVHMR's Google Drive folder." -ForegroundColor Cyan
  Write-Host "If this fails, download manually and place files under:" -ForegroundColor Yellow
  Write-Host "  $stagedCkptRoot" -ForegroundColor Yellow
  Write-Host ""

  $folderUrl = "https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
  try {
    Invoke-CmdChecked "`"$isaacLabBat`" -p -m pip install gdown==5.2.0"
  } catch {
    # Retry without pin.
    Invoke-CmdChecked "`"$isaacLabBat`" -p -m pip install gdown"
  }
  # Download into the staged checkpoint root.
  # NOTE: Google Drive frequently rate-limits large public files. Treat this as best-effort.
  cmd /c "`"$isaacLabBat`" -p -m gdown --folder `"$folderUrl`" --output `"$stagedCkptRoot`""
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] gdown folder download failed (exit=$LASTEXITCODE). Will fall back to alternative mirrors where possible." -ForegroundColor Yellow
  }
}

Write-Host ""
Write-Host "Remaining required checkpoint files (manual download may be needed):" -ForegroundColor Yellow
Write-Host "  - dpvo\\dpvo.pth"
Write-Host "  - vitpose\\vitpose-h-multi-coco.pth"
Write-Host "  - hmr2\\epoch=10-step=25000.ckpt"

if ($TryDownloadHeavyCheckpoints) {
  # If Google Drive is rate-limited, try a HuggingFace mirror for the ViTPose checkpoint.
  $vitposeCkpt = Join-Path $stagedCkptRoot "vitpose\\vitpose-h-multi-coco.pth"
  if (-not (Test-Path $vitposeCkpt)) {
    Write-Host ""
    Write-Host "-- Fallback: Download vitpose-h-multi-coco.pth from HuggingFace --" -ForegroundColor Cyan
    try {
      Download-File "https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth?download=true" $vitposeCkpt
    } catch {
      Write-Host "[WARN] HuggingFace download failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
  }
}

Write-Host ""
Write-Host "Once checkpoints are in place, run the real smoke test from the Mac:" -ForegroundColor Green
Write-Host "  ./scripts/smoke_e2e_with_gpu_real.sh"
