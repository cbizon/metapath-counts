#!/bin/bash
#SBATCH --job-name=prep_grp_reduceA
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1

set -e

TMP_DIR="$1"
PASSB_SHARDS="$2"
RESULTS_DIR="$3"
N_HOPS="$4"

if [ -z "$TMP_DIR" ] || [ -z "$PASSB_SHARDS" ] || [ -z "$RESULTS_DIR" ] || [ -z "$N_HOPS" ]; then
    echo "Usage: $0 <tmp_dir> <passb_shards> <results_dir> <n_hops>"
    exit 1
fi

uv run python src/pipeline/workers/prepare_grouping_reduceA.py \
    --tmp-dir "$TMP_DIR" \
    --passb-shards "$PASSB_SHARDS" \
    --results-dir "$RESULTS_DIR" \
    --n-hops "$N_HOPS"
