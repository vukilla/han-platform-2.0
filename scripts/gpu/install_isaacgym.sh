#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARCHIVE="${ISAACGYM_ARCHIVE:-}"
if [[ -z "$ARCHIVE" ]]; then
  # Common expected locations.
  for p in \\
    "$ROOT_DIR/external/humanoid-projects/isaacgym/isaacgym_preview4.tar.gz" \\
    "$ROOT_DIR/external/humanoid-projects/isaacgym/isaacgym_preview4.tgz" \\
    "$ROOT_DIR/external/humanoid-projects/isaacgym/isaacgym_preview4.zip" \\
    ; do
    if [[ -f "$p" ]]; then
      ARCHIVE="$p"
      break
    fi
  done
fi

if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
  echo "Isaac Gym archive not found."
  echo "Download Isaac Gym Preview 4 from: https://developer.nvidia.com/isaac-gym"
  echo "Then place it at one of:"
  echo "  $ROOT_DIR/external/humanoid-projects/isaacgym/isaacgym_preview4.tar.gz"
  echo "  $ROOT_DIR/external/humanoid-projects/isaacgym/isaacgym_preview4.zip"
  echo "Or set ISAACGYM_ARCHIVE=/path/to/archive"
  exit 2
fi

DEST="$ROOT_DIR/external/isaacgym"
mkdir -p "$DEST"

echo "Extracting Isaac Gym archive:"
echo "  $ARCHIVE"
echo "Into:"
echo "  $DEST"

if [[ "$ARCHIVE" == *.zip ]]; then
  unzip -q "$ARCHIVE" -d "$DEST"
else
  tar -xzf "$ARCHIVE" -C "$DEST"
fi

# Isaac Gym package typically extracts to a subdirectory like IsaacGym_Preview_4_Package/isaacgym
ISAACGYM_DIR="$(find "$DEST" -maxdepth 3 -type d -name isaacgym | head -n 1 || true)"
if [[ -z "$ISAACGYM_DIR" ]]; then
  echo "Could not locate extracted isaacgym/ directory under $DEST"
  exit 3
fi

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv_gpu}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"

echo "Installing Isaac Gym python package into venv: $VENV_DIR"
"$PY" -m pip install -U pip wheel setuptools >/dev/null
if [[ -d "$ISAACGYM_DIR/python" ]]; then
  "$PY" -m pip install -e "$ISAACGYM_DIR/python"
else
  "$PY" -m pip install -e "$ISAACGYM_DIR"
fi

echo
echo "Installed. Quick import test:"
"$PY" - <<'PY'
import isaacgym  # noqa: F401
print("isaacgym import: OK")
PY

