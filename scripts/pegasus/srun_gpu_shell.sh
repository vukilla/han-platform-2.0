#!/usr/bin/env bash
set -euo pipefail

# Interactive GPU shell on Pegasus (Slurm).
#
# Usage:
#   scripts/pegasus/srun_gpu_shell.sh [PARTITION] [GPUS] [TIME]
#
# Examples:
#   scripts/pegasus/srun_gpu_shell.sh batch 1 02:00:00
#   scripts/pegasus/srun_gpu_shell.sh A100-40GB 1 02:00:00
#   scripts/pegasus/srun_gpu_shell.sh H100 1 01:00:00
#
# Notes:
# - Edit defaults to match your granted partitions/quotas.
# - This is a template; Pegasus partitions differ per user allocation.

PARTITION="${1:-batch}"
GPUS="${2:-1}"
TIME="${3:-02:00:00}"

echo "Requesting interactive GPU shell:"
echo "  partition=$PARTITION gpus=$GPUS time=$TIME"
echo

srun --pty \
  -p "$PARTITION" \
  --gres="gpu:$GPUS" \
  -t "$TIME" \
  --cpus-per-task=8 \
  --mem=32G \
  bash -l
