#!/bin/bash
#SBATCH --job-name=prep_grp_A
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1

set -e

FILES_LIST="$1"
SHARD_COUNT="$2"
OUTPUT_DIR="$3"

if [ -z "$FILES_LIST" ] || [ -z "$SHARD_COUNT" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <files_list> <shard_count> <output_dir>"
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set"
    exit 1
fi

uv run python src/pipeline/workers/prepare_grouping_passA.py \
    --files-list "$FILES_LIST" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --shard-count "$SHARD_COUNT" \
    --output-dir "$OUTPUT_DIR"
