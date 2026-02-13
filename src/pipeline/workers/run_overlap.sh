#!/bin/bash
#SBATCH --partition=lowpri
#SBATCH --time=24:00:00

# Arguments from command line (passed from orchestrate script)
MATRIX1_INDEX=$1
MATRICES_DIR=$2
N_HOPS=$3
SRC_TYPE=$4
PRED=$5
DIRECTION=$6
TGT_TYPE=$7

# Validate arguments
if [ -z "$MATRIX1_INDEX" ]; then
    echo "ERROR: Missing matrix1_index argument"
    echo "Usage: $0 <matrix1_index> <matrices_dir> <n_hops> <src_type> <pred> <direction> <tgt_type>"
    exit 1
fi

if [ -z "$MATRICES_DIR" ]; then
    echo "ERROR: Missing matrices_dir argument"
    echo "Usage: $0 <matrix1_index> <matrices_dir> <n_hops> <src_type> <pred> <direction> <tgt_type>"
    exit 1
fi

if [ -z "$N_HOPS" ]; then
    echo "ERROR: Missing n_hops argument"
    echo "Usage: $0 <matrix1_index> <matrices_dir> <n_hops> <src_type> <pred> <direction> <tgt_type>"
    exit 1
fi

if [ -z "$SRC_TYPE" ] || [ -z "$PRED" ] || [ -z "$DIRECTION" ] || [ -z "$TGT_TYPE" ]; then
    echo "ERROR: Missing matrix1 spec arguments (src_type, pred, direction, tgt_type)"
    echo "Usage: $0 <matrix1_index> <matrices_dir> <n_hops> <src_type> <pred> <direction> <tgt_type>"
    exit 1
fi

if [ ! -d "$MATRICES_DIR" ]; then
    echo "ERROR: Matrices directory not found: $MATRICES_DIR"
    exit 1
fi

# Output file path (n_hop-specific directory)
OUTPUT_FILE="results_${N_HOPS}hop/results_matrix1_$(printf '%03d' $MATRIX1_INDEX).tsv"

echo "Using pre-built matrices from: $MATRICES_DIR"

# Activate environment and run analysis
source .venv/bin/activate

uv run python src/pipeline/workers/run_overlap.py \
  --matrix1-index $MATRIX1_INDEX \
  --matrices-dir $MATRICES_DIR \
  --n-hops $N_HOPS \
  --src-type "$SRC_TYPE" \
  --pred "$PRED" \
  --direction "$DIRECTION" \
  --tgt-type "$TGT_TYPE" \
  --output $OUTPUT_FILE

EXIT_CODE=$?
exit $EXIT_CODE
