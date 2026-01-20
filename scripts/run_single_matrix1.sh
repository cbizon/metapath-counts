#!/bin/bash
#SBATCH --partition=lowpri
#SBATCH --time=24:00:00

# Arguments from command line (passed from orchestrate script via manifest)
MATRIX1_INDEX=$1
NODES_FILE=$2
EDGES_FILE=$3
NO_ZEROING=${4:-""}  # Optional 4th argument

# Validate arguments
if [ -z "$MATRIX1_INDEX" ] || [ -z "$NODES_FILE" ] || [ -z "$EDGES_FILE" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 <matrix1_index> <nodes_file> <edges_file> [no_zeroing]"
    exit 1
fi

# Output file path
OUTPUT_FILE="scripts/metapaths/results/results_matrix1_$(printf '%03d' $MATRIX1_INDEX).tsv"

# Build command with optional --no-zeroing flag
CMD="uv run python scripts/metapaths/analyze_3hop_overlap.py \
  --matrix1-index $MATRIX1_INDEX \
  --edges $EDGES_FILE \
  --nodes $NODES_FILE \
  --output $OUTPUT_FILE"

if [ "$NO_ZEROING" = "no_zeroing" ]; then
    CMD="$CMD --no-zeroing"
fi

# Activate environment and run analysis
source .venv/bin/activate
$CMD

EXIT_CODE=$?
exit $EXIT_CODE
