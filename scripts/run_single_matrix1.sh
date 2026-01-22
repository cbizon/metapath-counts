#!/bin/bash
#SBATCH --partition=lowpri
#SBATCH --time=24:00:00

# Arguments from command line (passed from orchestrate script via manifest)
MATRIX1_INDEX=$1
NODES_FILE=$2
EDGES_FILE=$3
N_HOPS=${4:-3}  # Default to 3 if not provided

# Validate arguments
if [ -z "$MATRIX1_INDEX" ] || [ -z "$NODES_FILE" ] || [ -z "$EDGES_FILE" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 <matrix1_index> <nodes_file> <edges_file> [n_hops]"
    exit 1
fi

# Output file path (n_hop-specific directory)
OUTPUT_FILE="results_${N_HOPS}hop/results_matrix1_$(printf '%03d' $MATRIX1_INDEX).tsv"

# Build command
CMD="uv run python scripts/analyze_hop_overlap.py \
  --matrix1-index $MATRIX1_INDEX \
  --edges $EDGES_FILE \
  --nodes $NODES_FILE \
  --n-hops $N_HOPS \
  --output $OUTPUT_FILE"

# Activate environment and run analysis
source .venv/bin/activate
$CMD

EXIT_CODE=$?
exit $EXIT_CODE
