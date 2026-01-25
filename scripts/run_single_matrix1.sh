#!/bin/bash
#SBATCH --partition=lowpri
#SBATCH --time=24:00:00

# Arguments from command line (passed from orchestrate script via manifest)
MATRIX1_INDEX=$1
NODES_FILE=$2
EDGES_FILE=$3
N_HOPS=${4:-3}  # Default to 3 if not provided
MATRICES_DIR=${5:-""}  # Optional pre-built matrices directory
CONFIG_FILE=${6:-"config/type_expansion.yaml"}  # Optional config file

# Validate arguments
if [ -z "$MATRIX1_INDEX" ]; then
    echo "ERROR: Missing matrix1_index argument"
    echo "Usage: $0 <matrix1_index> <nodes_file> <edges_file> [n_hops] [matrices_dir] [config]"
    exit 1
fi

# Output file path (n_hop-specific directory)
OUTPUT_FILE="results_${N_HOPS}hop/results_matrix1_$(printf '%03d' $MATRIX1_INDEX).tsv"

# Build command (use pre-built matrices if available)
if [ -n "$MATRICES_DIR" ] && [ -d "$MATRICES_DIR" ]; then
    echo "Using pre-built matrices from: $MATRICES_DIR"
    CMD="uv run python scripts/analyze_hop_overlap.py \
      --matrix1-index $MATRIX1_INDEX \
      --matrices-dir $MATRICES_DIR \
      --n-hops $N_HOPS \
      --config $CONFIG_FILE \
      --output $OUTPUT_FILE"
else
    # Fall back to loading from edges
    if [ -z "$NODES_FILE" ] || [ -z "$EDGES_FILE" ]; then
        echo "ERROR: Missing nodes_file or edges_file (required when not using pre-built matrices)"
        exit 1
    fi
    echo "Building matrices from edges: $EDGES_FILE"
    CMD="uv run python scripts/analyze_hop_overlap.py \
      --matrix1-index $MATRIX1_INDEX \
      --edges $EDGES_FILE \
      --nodes $NODES_FILE \
      --n-hops $N_HOPS \
      --config $CONFIG_FILE \
      --output $OUTPUT_FILE"
fi

# Activate environment and run analysis
source .venv/bin/activate
$CMD

EXIT_CODE=$?
exit $EXIT_CODE
