#!/usr/bin/env bash
set -euo pipefail

echo "== System =="
uname -a || true
echo

echo "== Tools =="
command -v git >/dev/null 2>&1 && echo "git: $(git --version)" || echo "git: MISSING"
command -v curl >/dev/null 2>&1 && echo "curl: ok" || echo "curl: MISSING"
command -v unzip >/dev/null 2>&1 && echo "unzip: ok" || echo "unzip: MISSING"
command -v python3 >/dev/null 2>&1 && echo "python3: $(python3 --version)" || echo "python3: MISSING"
echo

echo "== NVIDIA =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi: MISSING (install NVIDIA driver and CUDA runtime)"
fi
echo

echo "== Notes =="
echo "- If Isaac Gym fails on RTX 5090 (sm_120), prefer Isaac Lab or a newer sim stack."
echo "- Run this script again after driver/tooling changes."

