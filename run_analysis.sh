#!/bin/bash
# Full N-hop metapath analysis workflow
# Runs 1-hop, 2-hop, and 3-hop analyses sequentially

set -e  # Exit on error

EDGES="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/84e6183aaeef2a8c/edges.jsonl"
NODES="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/84e6183aaeef2a8c/nodes.jsonl"
MATRICES_DIR="matrices"

# List of N-hop values to analyze
NHOP_VALUES=(1 2 3)

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
    echo "This is a one-time operation (~9GB compressed)."
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
        --edges "$EDGES" \
        --nodes "$NODES" \
        --n-hops "$N_HOPS" \
        --matrices-dir "$MATRICES_DIR"

    # Step 2: Run orchestrator (submit SLURM jobs)
    echo ""
    echo "Step 2: Running ${N_HOPS}-hop orchestrator..."
    echo "This will submit jobs to SLURM and monitor progress."
    echo "You can Ctrl+C and restart later - it will resume from the manifest."
    echo ""
    uv run python scripts/orchestrate_hop_analysis.py --n-hops "$N_HOPS"

    # Step 3: Merge results
    echo ""
    echo "Step 3: Merging ${N_HOPS}-hop results..."
    uv run python scripts/merge_results.py --n-hops "$N_HOPS"

    # Step 4: Group by 1-hop metapath
    echo ""
    echo "Step 4: Grouping ${N_HOPS}-hop by 1-hop metapath..."
    uv run python scripts/group_by_onehop.py --n-hops "$N_HOPS"

    echo ""
    echo "✓ ${N_HOPS}-hop analysis complete!"
    echo "  Results: results_${N_HOPS}hop/all_${N_HOPS}hop_overlaps.tsv"
    echo "  Grouped: grouped_by_results_${N_HOPS}hop/"
done

echo ""
echo "=========================================="
echo "ALL ANALYSES COMPLETE"
echo "=========================================="
echo ""
echo "Summary of results:"
for N_HOPS in "${NHOP_VALUES[@]}"; do
    echo "  ${N_HOPS}-hop:"
    echo "    - Merged results: results_${N_HOPS}hop/all_${N_HOPS}hop_overlaps.tsv"
    echo "    - Grouped files:  grouped_by_results_${N_HOPS}hop/"
done
echo ""
