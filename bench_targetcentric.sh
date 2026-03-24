#!/bin/bash
# Submit 5 profiling runs for ChemicalEntity/DiseaseOrPhenotypicFeature
# at different precision cutoffs to benchmark the restored target-centric path.
#
# Usage: bash bench_targetcentric.sh

set -e

TYPE1="ChemicalEntity"
TYPE2="DiseaseOrPhenotypicFeature"
N_HOPS=3
FILE_LIST="results_3hop/_tmp_prepare_grouping/files.txt"
EXPLICIT_SHARDS="results_3hop/_tmp_prepare_grouping/typepair_explicit_paths"
TYPE_NODE_COUNTS="results_3hop/type_node_counts.json"
EXCLUDE_TYPES="Entity,ThingWithTaxon"
EXCLUDE_PREDICATES="related_to_at_instance_level,related_to_at_concept_level"
TAG="targetcentric"
LOG_DIR="logs_bench_${TAG}"

mkdir -p "$LOG_DIR"

for PREC in 0.001 0.01 0.1 0.9 0.99; do
    PREC_TAG=$(echo "$PREC" | sed 's/\.//g')
    OUTPUT_DIR="grouped_profiled_sample_3hop_${TAG}${PREC_TAG}"
    PROGRESS_FILE="${LOG_DIR}/progress_${PREC_TAG}.json"
    mkdir -p "$OUTPUT_DIR"

    echo "Submitting: precision=${PREC} -> ${OUTPUT_DIR}"

    sbatch \
        --job-name="bench_${TAG}_${PREC_TAG}" \
        --time=72:00:00 \
        --cpus-per-task=1 \
        --mem=400G \
        --output="${LOG_DIR}/slurm_${PREC_TAG}_%j.out" \
        --error="${LOG_DIR}/slurm_${PREC_TAG}_%j.err" \
        --wrap="uv run python src/pipeline/workers/run_grouping.py \
            --type1 '${TYPE1}' \
            --type2 '${TYPE2}' \
            --file-list '${FILE_LIST}' \
            --output-dir '${OUTPUT_DIR}' \
            --n-hops ${N_HOPS} \
            --explicit-shards-dir '${EXPLICIT_SHARDS}' \
            --type-node-counts '${TYPE_NODE_COUNTS}' \
            --min-count 0 \
            --min-precision ${PREC} \
            --exclude-types '${EXCLUDE_TYPES}' \
            --exclude-predicates '${EXCLUDE_PREDICATES}' \
            --progress-file '${PROGRESS_FILE}'"
done

echo ""
echo "All 5 jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: ${LOG_DIR}/"
