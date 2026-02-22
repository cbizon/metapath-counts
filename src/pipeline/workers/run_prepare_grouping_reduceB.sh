#!/bin/bash
#SBATCH --job-name=prep_grp_reduceB
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1

set -e

TMP_DIR="$1"
RESULTS_DIR="$2"
N_HOPS="$3"

if [ -z "$TMP_DIR" ] || [ -z "$RESULTS_DIR" ] || [ -z "$N_HOPS" ]; then
    echo "Usage: $0 <tmp_dir> <results_dir> <n_hops>"
    exit 1
fi

uv run python src/pipeline/workers/prepare_grouping_reduceB.py \
    --tmp-dir "$TMP_DIR" \
    --results-dir "$RESULTS_DIR" \
    --n-hops "$N_HOPS"
