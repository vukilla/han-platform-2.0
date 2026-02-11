#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XGEN_DIR="$ROOT_DIR/services/xgen"
DEFAULT_GVHMR_DIR="$ROOT_DIR/external/gvhmr"
HUMANOID_GVHMR_DIR="$ROOT_DIR/external/humanoid-projects/GVHMR"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <video.mp4> <smplx_model_dir> <output_dir>"
  echo "Example: $0 /path/sample.mp4 /path/smplx/models /tmp/humanoid_network_out"
  exit 1
fi

VIDEO_PATH="$1"
SMPLX_MODEL_DIR="$2"
OUT_DIR="$3"
VIDEO_STEM="$(basename "$VIDEO_PATH")"
VIDEO_STEM="${VIDEO_STEM%.*}"

mkdir -p "$OUT_DIR"

# 1) Run GVHMR demo via the wrapper (requires GVHMR deps + checkpoints)
GVHMR_ROOT="${GVHMR_ROOT:-}"
if [[ -z "$GVHMR_ROOT" ]]; then
  GVHMR_ROOT="$DEFAULT_GVHMR_DIR"
fi

SMPLX_NPZ="$OUT_DIR/${VIDEO_STEM}_gvhmr_smplx.npz"
if [[ -d "$GVHMR_ROOT" ]]; then
  if [[ -f "$GVHMR_ROOT/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" ]] || [[ -f "$HUMANOID_GVHMR_DIR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" ]]; then
    echo "Running GVHMR → SMPL-X (output: $SMPLX_NPZ)"
    VIDEO_PATH="$VIDEO_PATH" OUT_DIR="$OUT_DIR" GVHMR_ROOT="$GVHMR_ROOT" PYTHONPATH="$XGEN_DIR" python - <<'PY'
import os
from pathlib import Path

from xgen.pose.gvhmr_pose import estimate_smplx_from_video

video = Path(os.environ["VIDEO_PATH"])
out_dir = Path(os.environ["OUT_DIR"])
estimate_smplx_from_video(video, out_dir, static_cam=True)
PY
  else
    echo "GVHMR checkpoints not found. Expected either:"
    echo "  - $GVHMR_ROOT/inputs/checkpoints/..."
    echo "  - $HUMANOID_GVHMR_DIR/inputs/checkpoints/..."
    echo "Skipping GVHMR step."
  fi
else
  echo "GVHMR repo not found at $GVHMR_ROOT. Set GVHMR_ROOT or clone GVHMR. Skipping GVHMR step."
fi

# 2) Convert SMPL-X NPZ to PhysHOI motion
PHYS_MOTION="$OUT_DIR/motion_physhoi.pt"
if [[ -f "$SMPLX_NPZ" ]]; then
  echo "Exporting PhysHOI motion: $PHYS_MOTION"
  PYTHONPATH="$XGEN_DIR" python "$XGEN_DIR/scripts/physhoi_export.py" \
    --smplx-npz "$SMPLX_NPZ" \
    --model-dir "$SMPLX_MODEL_DIR" \
    --output "$PHYS_MOTION"
else
  echo "SMPL-X NPZ not found at $SMPLX_NPZ. Provide SMPL-X NPZ and rerun."
fi

# 3) Emit YAML motion pack for PhysHOI
if [[ -f "$PHYS_MOTION" ]]; then
  echo "Generating PhysHOI motion YAML pack"
  PYTHONPATH="$XGEN_DIR" python "$XGEN_DIR/scripts/physhoi_pack.py" \
    --motions "$PHYS_MOTION" \
    --output "$OUT_DIR/motions.yaml" \
    --validate
fi

echo "CPU-side pipeline finished. Next: copy outputs to GPU PC and run PhysHOI."
