#!/bin/bash
# Re-run grouping with filters
# Usage: ./rerun_grouping.sh [n_hops]
# Default: n_hops=2

set -e

N_HOPS="${1:-2}"

echo "Re-running grouping for ${N_HOPS}-hop with filters:"
echo "  --min-count 10"
echo "  --min-precision 0.001"
echo "  --exclude-types Entity,ThingWithTaxon"
echo "  --exclude-predicates related_to_at_instance_level,related_to_at_concept_level"
echo ""

echo "Step 1: Preparing grouping (precomputing counts)..."
echo "  - Creates results_${N_HOPS}hop/aggregated_path_counts.json (1-hop counts)"
echo "  - Creates results_${N_HOPS}hop/aggregated_nhop_counts.json (N-hop counts)"
echo "  - Creates results_${N_HOPS}hop/type_node_counts.json"
echo "  - Creates results_${N_HOPS}hop/grouping_manifest.json"
echo ""
uv run python src/pipeline/prepare_grouping.py --n-hops "$N_HOPS"

echo ""
echo "Step 2: Running distributed grouping..."
uv run python src/pipeline/orchestrate_grouping.py --n-hops "$N_HOPS" \
    --min-count 10 \
    --min-precision 0.001 \
    --partition lowpri

echo ""
echo "Done! Results in: grouped_by_results_${N_HOPS}hop/"
