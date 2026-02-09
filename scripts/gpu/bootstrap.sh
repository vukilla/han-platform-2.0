#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

say() { echo "== $* =="; }

say "GPU PC Bootstrap"
echo "Repo: $ROOT_DIR"
echo

"$ROOT_DIR/scripts/gpu/preflight.sh" || true
echo

mkdir -p "$ROOT_DIR/external"
mkdir -p "$ROOT_DIR/external/humanoid-projects"

clone_or_update() {
  local url="$1"
  local dest="$2"
  local commit="${3:-}"
  if [[ ! -d "$dest/.git" ]]; then
    say "Cloning $(basename "$dest")"
    git clone "$url" "$dest"
  else
    say "Updating $(basename "$dest")"
    git -C "$dest" fetch --all --tags --prune || true
  fi
  if [[ -n "$commit" ]]; then
    git -C "$dest" checkout "$commit" || true
  fi
}

clone_or_update "https://github.com/zju3dv/GVHMR" "$ROOT_DIR/external/gvhmr" "088caff"
clone_or_update "https://github.com/wyhuai/PhysHOI" "$ROOT_DIR/external/physhoi" "6095c60"
clone_or_update "https://github.com/facebookresearch/sam-3d-objects" "$ROOT_DIR/external/sam3d" "81a8237"

say "Python Environment"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv_gpu}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install -U pip wheel setuptools >/dev/null
"$PY" -m pip install -U gdown >/dev/null
GDOWN="$VENV_DIR/bin/gdown"

say "GVHMR Checkpoints"
GVHMR_ROOT="$ROOT_DIR/external/gvhmr"
mkdir -p "$GVHMR_ROOT/inputs/checkpoints/gvhmr"
mkdir -p "$GVHMR_ROOT/inputs/checkpoints/dpvo"
mkdir -p "$GVHMR_ROOT/inputs/checkpoints/vitpose"
mkdir -p "$GVHMR_ROOT/inputs/checkpoints/hmr2"

GVHMR_MAIN="$GVHMR_ROOT/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
if [[ ! -f "$GVHMR_MAIN" ]]; then
  echo "Downloading gvhmr_siga24_release.ckpt (HuggingFace)..."
  curl -L --retry 3 -o "$GVHMR_MAIN" "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt?download=true"
else
  echo "Exists: $GVHMR_MAIN"
fi

# Best-effort Google Drive folder download for remaining checkpoints (may require login).
GVHMR_GDRIVE_DIR="$ROOT_DIR/external/humanoid-projects/GVHMR_gdrive"
if [[ ! -f "$GVHMR_ROOT/inputs/checkpoints/dpvo/dpvo.pth" ]] || [[ ! -f "$GVHMR_ROOT/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth" ]]; then
  echo
  echo "Attempting to download remaining GVHMR checkpoints from Google Drive (best-effort)..."
  echo "If this fails, open the folder in a browser and download manually:"
  echo "  https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
  "$GDOWN" --folder "https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD" -O "$GVHMR_GDRIVE_DIR" || true

  # Copy known required files into expected locations if found.
  found_dpvo="$(find "$GVHMR_GDRIVE_DIR" -type f -name 'dpvo.pth' 2>/dev/null | head -n 1 || true)"
  found_vitpose="$(find "$GVHMR_GDRIVE_DIR" -type f -name 'vitpose-h-multi-coco.pth' 2>/dev/null | head -n 1 || true)"
  found_hmr2="$(find "$GVHMR_GDRIVE_DIR" -type f -name 'epoch=10-step=25000.ckpt' 2>/dev/null | head -n 1 || true)"

  [[ -n "$found_dpvo" ]] && cp "$found_dpvo" "$GVHMR_ROOT/inputs/checkpoints/dpvo/dpvo.pth" || true
  [[ -n "$found_vitpose" ]] && cp "$found_vitpose" "$GVHMR_ROOT/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth" || true
  [[ -n "$found_hmr2" ]] && cp "$found_hmr2" "$GVHMR_ROOT/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt" || true
fi

say "PhysHOI Checkpoints"
PHYSHOI_ZIP="$ROOT_DIR/external/humanoid-projects/PhysHOI/physhoi_checkpoints.zip"
mkdir -p "$(dirname "$PHYSHOI_ZIP")"
if [[ ! -f "$PHYSHOI_ZIP" ]]; then
  echo "Attempting to download PhysHOI trained_models zip via gdown (best-effort)..."
  echo "If this fails, download manually and place at:"
  echo "  $PHYSHOI_ZIP"
  "$GDOWN" "https://drive.google.com/uc?id=1jPnzd6PVVpiWNA1-MTVuUgIR_GOJMcLu" -O "$PHYSHOI_ZIP" || true
else
  echo "Exists: $PHYSHOI_ZIP"
fi

if [[ -f "$PHYSHOI_ZIP" ]]; then
  PHYSHOI_ROOT="$ROOT_DIR/external/physhoi" "$ROOT_DIR/scripts/install_physhoi_checkpoints.sh" "$PHYSHOI_ZIP" || true
fi

say "Isaac Gym"
echo "Isaac Gym cannot be downloaded automatically (NVIDIA license gate)."
echo "1. Download Isaac Gym Preview 4 from: https://developer.nvidia.com/isaac-gym"
echo "2. Place the archive at: $ROOT_DIR/external/humanoid-projects/isaacgym/isaacgym_preview4.tar.gz (or .zip)"
echo "3. Then run: scripts/gpu/install_isaacgym.sh"
echo

say "Next Commands"
echo "After Isaac Gym is installed, verify PhysHOI inference:"
echo "  scripts/run_physhoi_inference.sh \\"
echo "    external/physhoi/physhoi/data/motions/BallPlay/pass.pt \\"
echo "    external/physhoi/physhoi/data/models/pass/nn/PhysHOI.pth \\"
echo "    16 PhysHOI_BallPlay"
echo
echo "Then run the CPU-side pipeline (GVHMR -> SMPL-X -> PhysHOI motion pack):"
echo "  scripts/run_end_to_end_cpu.sh <video.mp4> <smplx_model_dir> <output_dir>"
