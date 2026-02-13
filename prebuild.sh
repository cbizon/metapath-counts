#!/bin/bash
# Pre-build matrices for faster analysis startup

set -e  # Exit on error

EDGES="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/84e6183aaeef2a8c/edges.jsonl"
NODES="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/84e6183aaeef2a8c/nodes.jsonl"
OUTPUT_DIR="matrices"

echo "Pre-building matrices..."
echo "  Edges: $EDGES"
echo "  Nodes: $NODES"
echo "  Output: $OUTPUT_DIR"
echo ""

uv run python src/pipeline/prebuild_matrices.py \
    --edges "$EDGES" \
    --nodes "$NODES" \
    --output "$OUTPUT_DIR"

echo ""
echo "Done! Pre-built matrices saved to $OUTPUT_DIR"
echo ""
echo "Next: Use --matrices-dir $OUTPUT_DIR in your analysis commands"
