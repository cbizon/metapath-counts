#!/bin/bash
#SBATCH --job-name=prep_grp_B
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1

set -e

TMP_DIR="$1"

if [ -z "$TMP_DIR" ]; then
    echo "Usage: $0 <tmp_dir>"
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set"
    exit 1
fi

uv run python src/pipeline/workers/prepare_grouping_passB.py \
    --tmp-dir "$TMP_DIR" \
    --shard-index "$SLURM_ARRAY_TASK_ID"
