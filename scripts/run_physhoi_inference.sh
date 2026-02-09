#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <motion_file> <checkpoint> [num_envs] [task]"
  echo "Example: $0 external/humanoid-projects/PhysHOI/physhoi/data/motions/BallPlay/toss.pt external/humanoid-projects/PhysHOI/physhoi/data/models/toss/nn/PhysHOI.pth 16 PhysHOI_BallPlay"
  exit 1
fi

MOTION_RAW="$1"
CKPT_RAW="$2"
NUM_ENVS="${3:-16}"
TASK="${4:-PhysHOI_BallPlay}"
EXTRA_ARGS=("${@:5}")

PHYSHOI_ROOT="${PHYSHOI_ROOT:-}"
if [[ -z "$PHYSHOI_ROOT" ]]; then
  if [[ -d "$ROOT_DIR/external/humanoid-projects/PhysHOI" ]]; then
    PHYSHOI_ROOT="$ROOT_DIR/external/humanoid-projects/PhysHOI"
  else
    PHYSHOI_ROOT="$ROOT_DIR/external/physhoi"
  fi
fi

MOTION_FILE="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$MOTION_RAW")"
CHECKPOINT="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$CKPT_RAW")"

CFG_ENV="$PHYSHOI_ROOT/physhoi/data/cfg/physhoi.yaml"
CFG_TRAIN="$PHYSHOI_ROOT/physhoi/data/cfg/train/rlg/physhoi.yaml"

python "$PHYSHOI_ROOT/physhoi/run.py" --test --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --motion_file "$MOTION_FILE" \
  --checkpoint "$CHECKPOINT" \
  "${EXTRA_ARGS[@]}"
