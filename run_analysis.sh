#!/bin/bash
# Full N-hop metapath analysis workflow
# Runs 1-hop, 2-hop, and 3-hop analyses sequentially

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

NODES="/projects/sequence_analysis/vol3/bizon/sub/translator_kg/Jan_20_filtered_nonredundant/nodes.jsonl"
EDGES="/projects/sequence_analysis/vol3/bizon/sub/translator_kg/Jan_20_filtered_nonredundant/edges.jsonl"
MATRICES_DIR="matrices"

# List of N-hop values to analyze
#NHOP_VALUES=(1 2 3)
NHOP_VALUES=(1)

echo "=========================================="
echo "N-HOP METAPATH ANALYSIS PIPELINE"
echo "=========================================="
echo "Edges: $EDGES"
echo "Nodes: $NODES"
echo "Running analyses for: ${NHOP_VALUES[@]} hops"
echo ""

# Pre-build matrices if they don't exist
if [ ! -d "$MATRICES_DIR" ] || [ ! -f "$MATRICES_DIR/manifest.json" ]; then
    echo "=========================================="
    echo "PRE-BUILDING MATRICES"
    echo "=========================================="
    echo "Pre-built matrices not found. Building now..."
    echo "This is a one-time operation."
    echo "Uses single-type assignment (fast, no explosion)."
    echo "Hierarchical aggregation happens during grouping step."
    echo ""
    uv run python scripts/prebuild_matrices.py \
        --edges "$EDGES" \
        --nodes "$NODES" \
        --output "$MATRICES_DIR"
    echo ""
    echo "✓ Matrix pre-building complete!"
    echo ""
else
    echo "Using pre-built matrices from: $MATRICES_DIR"
    echo ""
fi

# Loop over each N-hop value
for N_HOPS in "${NHOP_VALUES[@]}"; do
    echo ""
    echo "######################################"
    echo "# STARTING ${N_HOPS}-HOP ANALYSIS"
    echo "######################################"
    echo ""

    # Step 1: Prepare analysis (create manifest)
    echo "Step 1: Preparing ${N_HOPS}-hop analysis..."
    uv run python scripts/prepare_analysis.py \
        --matrices-dir "$MATRICES_DIR" \
        --n-hops "$N_HOPS"

    # Step 2: Run orchestrator (submit SLURM jobs)
    echo ""
    echo "Step 2: Running ${N_HOPS}-hop orchestrator..."
    echo "This will submit jobs to SLURM and monitor progress."
    echo "You can Ctrl+C and restart later - it will resume from the manifest."
    echo ""
    uv run python scripts/orchestrate_hop_analysis.py \
        --n-hops "$N_HOPS"

    # Step 3: Prepare distributed grouping (create type pair jobs)
    echo ""
    echo "Step 3: Preparing distributed grouping..."
    uv run python scripts/prepare_grouping.py --n-hops "$N_HOPS"

    # Step 4: Run distributed grouping (one SLURM job per type pair)
    echo ""
    echo "Step 4: Running distributed grouping..."
    echo "  - Each type pair (src_type, tgt_type) processed by separate SLURM job"
    echo "  - Each job finds and groups all 1-hop metapaths between those types"
    echo "  - Memory-efficient streaming through relevant result files only"
    echo "  - Automatic hierarchical aggregation"
    echo "  - You can Ctrl+C and restart later - it will resume from the manifest"
    echo ""
    uv run python scripts/orchestrate_grouping.py --n-hops "$N_HOPS"

    echo ""
    echo "✓ ${N_HOPS}-hop analysis complete!"
    echo "  Grouped results: grouped_by_results_${N_HOPS}hop/"
    echo "  Note: No merged file created (grouping is distributed)"
done

echo ""
echo "=========================================="
echo "ALL ANALYSES COMPLETE"
echo "=========================================="
echo ""
echo "Summary of results:"
for N_HOPS in "${NHOP_VALUES[@]}"; do
    echo "  ${N_HOPS}-hop:"
    echo "    - Grouped results: grouped_by_results_${N_HOPS}hop/"
    echo "    - Individual files: results_${N_HOPS}hop/results_matrix1_*.tsv"
done
echo ""
