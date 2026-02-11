#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPO_NAME="${1:-han-platform}" # can also be OWNER/REPO
VISIBILITY="${2:-private}"         # private|public|internal
DESCRIPTION="${3:-HumanX Data Factory (XGen + XMimic) platform scaffold}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required but not installed."
  exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Not logged into GitHub via gh."
  echo "Run: gh auth login --hostname github.com --git-protocol https --web"
  echo "If GitHub web/device auth is down, use a PAT:"
  echo "  export GH_TOKEN='<YOUR_PAT>'"
  echo "  echo \"\$GH_TOKEN\" | gh auth login --hostname github.com --with-token"
  exit 3
fi

case "$VISIBILITY" in
  private) VIS_FLAG="--private" ;;
  public) VIS_FLAG="--public" ;;
  internal) VIS_FLAG="--internal" ;;
  *) echo "Unknown visibility: $VISIBILITY (expected private|public|internal)"; exit 4 ;;
esac

# Ensure we have at least one commit.
if [[ ! -d .git ]]; then
  git init
fi
git add .
if ! git diff --cached --quiet; then
  git commit -m "Initial import: han-platform"
fi
git branch -M main

if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote 'origin' already exists. Skipping repo creation."
  echo "Origin: $(git remote get-url origin)"
else
  gh repo create "$REPO_NAME" "$VIS_FLAG" --source=. --remote=origin --push --description "$DESCRIPTION"
  exit 0
fi

# If origin exists, just push.
git push -u origin main
