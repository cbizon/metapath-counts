#!/bin/bash
# Re-run grouping with filters
# Usage: ./rerun_grouping.sh [n_hops]
# Default: n_hops=2

set -e

N_HOPS="${1:-2}"
RESULTS_DIR="results_${N_HOPS}hop"
TMP_DIR="${RESULTS_DIR}/_tmp_prepare_grouping"
REDUCEA_MEM_GB="${REDUCEA_MEM_GB:-250}"

echo "Re-running grouping for ${N_HOPS}-hop with filters:"
echo "  --min-count 10"
echo "  --min-precision 0.001"
echo "  --exclude-types Entity,ThingWithTaxon,PhysicalEssence,PhysicalEssenceOrOccurrent"
echo "  --exclude-predicates related_to,related_to_at_instance_level,related_to_at_concept_level,associated_with"
echo "  --mem-reducea ${REDUCEA_MEM_GB}G"
echo "  --tmp-dir ${TMP_DIR}"
echo ""

echo "Cleaning old prepare_grouping temp files..."
rm -rf "$TMP_DIR"

echo "Step 1: Precomputing explicit path counts on SLURM..."
precompute_output=$(uv run python src/pipeline/precompute_aggregated_counts_slurm.py \
    --n-hops "$N_HOPS" \
    --tmp-dir "$TMP_DIR" \
    --mem-reducea "$REDUCEA_MEM_GB")
echo "$precompute_output"

reduce_a_job_id=$(echo "$precompute_output" | awk '/Reduce A:/ {print $3}')
if [ -n "$reduce_a_job_id" ]; then
    echo "Waiting for Reduce A job ${reduce_a_job_id} to finish..."
    while squeue -j "$reduce_a_job_id" | tail -n +2 | grep -q .; do
        sleep 30
    done
    reduce_a_state=$(sacct -j "$reduce_a_job_id" --format=State --noheader | awk 'NF {print $1; exit}')
    echo "Reduce A final state: ${reduce_a_state}"
    if [ "$reduce_a_state" != "COMPLETED" ]; then
        echo "ERROR: Reduce A did not complete successfully."
        exit 1
    fi
    echo "Reduce A job complete."
else
    echo "WARNING: Could not parse Reduce A job ID. Proceeding without wait."
fi

echo ""
echo "Step 2: Preparing grouping..."
echo "  - Loads type-pair explicit-count shards from ${TMP_DIR}"
echo "  - Creates ${RESULTS_DIR}/type_node_counts.json"
echo "  - Creates ${RESULTS_DIR}/grouping_manifest.json"
echo ""
uv run python src/pipeline/prepare_grouping.py --n-hops "$N_HOPS" --tmp-dir "$TMP_DIR"

echo ""
echo "Step 3: Running distributed grouping..."
uv run python src/pipeline/orchestrate_grouping.py --n-hops "$N_HOPS" \
    --min-count 10 \
    --min-precision 0.001 \
    --partition lowpri

echo ""
echo "Done! Results in: grouped_by_results_${N_HOPS}hop/"
