#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUNDLE_DIR="${BUNDLE_DIR:-$ROOT_DIR/tmp/bundles}"
BUNDLE_NAME="${BUNDLE_NAME:-han-platform.bundle}"
BUNDLE_PATH="$BUNDLE_DIR/$BUNDLE_NAME"

mkdir -p "$BUNDLE_DIR"

cd "$ROOT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repo: $ROOT_DIR" >&2
  exit 1
fi

git bundle create "$BUNDLE_PATH" --all

echo "Bundle created:"
echo "  $BUNDLE_PATH"
echo
echo "Clone on another machine:"
echo "  git clone $BUNDLE_NAME han-platform"
