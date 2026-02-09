#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ZIP_PATH="${1:-$ROOT_DIR/external/humanoid-projects/PhysHOI/physhoi_checkpoints.zip}"

PHYSHOI_ROOT="${PHYSHOI_ROOT:-}"
if [[ -z "$PHYSHOI_ROOT" ]]; then
  if [[ -d "$ROOT_DIR/external/physhoi" ]]; then
    PHYSHOI_ROOT="$ROOT_DIR/external/physhoi"
  else
    echo "PhysHOI repo not found. Set PHYSHOI_ROOT or clone into external/physhoi." >&2
    exit 2
  fi
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Checkpoint zip not found: $ZIP_PATH" >&2
  exit 2
fi

MODELS_DIR="$PHYSHOI_ROOT/physhoi/data/models"
mkdir -p "$MODELS_DIR"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

unzip -q "$ZIP_PATH" -d "$TMP_DIR"

if [[ ! -d "$TMP_DIR/trained_models" ]]; then
  echo "Unexpected zip layout. Expected trained_models/ at zip root." >&2
  exit 3
fi

installed=0
for task_dir in "$TMP_DIR/trained_models"/*; do
  [[ -d "$task_dir" ]] || continue
  task="$(basename "$task_dir")"
  src="$task_dir/nn/PhysHOI.pth"
  if [[ ! -f "$src" ]]; then
    echo "Skipping $task: missing nn/PhysHOI.pth in zip." >&2
    continue
  fi
  dst="$MODELS_DIR/$task/nn/PhysHOI.pth"
  if [[ -f "$dst" ]]; then
    echo "Exists, skipping: $dst"
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "Installed: $dst"
  installed=$((installed + 1))
done

echo "Done. Installed $installed checkpoint(s) into: $MODELS_DIR"

