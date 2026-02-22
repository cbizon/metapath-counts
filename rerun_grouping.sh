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

echo "Step 1: Precomputing aggregated N-hop counts on SLURM..."
precompute_output=$(uv run python src/pipeline/precompute_aggregated_counts_slurm.py --n-hops "$N_HOPS")
echo "$precompute_output"

reduce_b_job_id=$(echo "$precompute_output" | awk '/Reduce B:/ {print $3}')
if [ -n "$reduce_b_job_id" ]; then
    echo "Waiting for Reduce B job ${reduce_b_job_id} to finish..."
    while squeue -j "$reduce_b_job_id" | tail -n +2 | grep -q .; do
        sleep 30
    done
    echo "Reduce B job complete."
else
    echo "WARNING: Could not parse Reduce B job ID. Proceeding without wait."
fi

echo ""
echo "Step 2: Preparing grouping (loading counts)..."
echo "  - Loads results_${N_HOPS}hop/aggregated_nhop_counts.json (N-hop counts)"
echo "  - Creates results_${N_HOPS}hop/type_node_counts.json"
echo "  - Creates results_${N_HOPS}hop/grouping_manifest.json"
echo ""
uv run python src/pipeline/prepare_grouping.py --n-hops "$N_HOPS" --skip-aggregated-precompute

echo ""
echo "Step 3: Running distributed grouping..."
uv run python src/pipeline/orchestrate_grouping.py --n-hops "$N_HOPS" \
    --min-count 10 \
    --min-precision 0.001 \
    --partition lowpri

echo ""
echo "Done! Results in: grouped_by_results_${N_HOPS}hop/"
