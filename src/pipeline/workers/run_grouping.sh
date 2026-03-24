#!/bin/bash
#SBATCH --job-name=group_typepair
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1

# Group results for all 1-hop metapaths between a type pair
#
# Args:
#   $1: type1 - First type (e.g., "Gene")
#   $2: type2 - Second type (e.g., "Disease")
#   $3: file_list_path - Path to file containing list of result files to scan
#   $4: n_hops - Number of hops (1, 2, or 3)
#   $5: min_count - Minimum N-hop count (default: 0)
#   $6: min_precision - Minimum precision (default: 0)
#   $7: exclude_types - Comma-separated types to exclude (default: Entity,ThingWithTaxon)
#   $8: exclude_predicates - Comma-separated predicates to exclude
#   $9: output_dir - Output directory for grouped results (optional)
#   $10: matrices_dir - Matrices directory for exact pair-set tracking (optional)
#
# This script:
# 1. Receives type pair and file list from orchestrator
# 2. Loads the explicit-count shard for that type pair
# 3. Scans overlap files for relevant rows
# 4. Computes exact target pair sets via GraphBLAS matrix union (Phase A)
# 5. Directly evaluates each (predictor path, target) pair via matrix reconstruction
# 6. Applies filters (min-count, min-precision, excluded types/predicates)
# 7. Writes output with exact matrix-derived counts

set -e

TYPE1="$1"
TYPE2="$2"
FILE_LIST_PATH="$3"
N_HOPS="$4"
MIN_COUNT="${5:-0}"
MIN_PRECISION="${6:-0}"
EXCLUDE_TYPES="${7:-Entity,ThingWithTaxon}"
EXCLUDE_PREDICATES="${8:-related_to,related_to_at_instance_level,related_to_at_concept_level,associated_with}"
OUTPUT_DIR_ARG="${9:-}"
MATRICES_DIR_ARG="${10:-}"

if [ -z "$TYPE1" ] || [ -z "$TYPE2" ] || [ -z "$FILE_LIST_PATH" ] || [ -z "$N_HOPS" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 <type1> <type2> <file_list_path> <n_hops> [min_count] [min_precision] [exclude_types] [exclude_predicates] [output_dir] [matrices_dir]"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR_ARG:-grouped_by_results_${N_HOPS}hop}"
MATRICES_DIR="${MATRICES_DIR_ARG:-matrices}"
RESULTS_DIR="results_${N_HOPS}hop"
TYPE_NODE_COUNTS="${RESULTS_DIR}/type_node_counts.json"
EXPLICIT_SHARDS_DIR="${RESULTS_DIR}/_tmp_prepare_grouping/typepair_explicit_paths"
LOG_DIR="logs_grouping_${N_HOPS}hop"
SAFE_TYPE1=$(echo "$TYPE1" | tr ':/+ ' '____')
SAFE_TYPE2=$(echo "$TYPE2" | tr ':/+ ' '____')
JOB_TAG="${SLURM_JOB_ID:-manual}"
PROGRESS_FILE="${LOG_DIR}/progress_${SAFE_TYPE1}__${SAFE_TYPE2}__${JOB_TAG}.json"

echo "=========================================="
echo "GROUPING TYPE PAIR"
echo "=========================================="
echo "Type pair: ($TYPE1, $TYPE2)"
echo "File list: $FILE_LIST_PATH"
echo "N-hops: $N_HOPS"
echo "Output dir: $OUTPUT_DIR"
echo "Matrices dir: $MATRICES_DIR"
echo "Explicit-count shards: $EXPLICIT_SHARDS_DIR"
echo "Type node counts: $TYPE_NODE_COUNTS"
echo "Progress file: $PROGRESS_FILE"
echo "Min count: $MIN_COUNT"
echo "Min precision: $MIN_PRECISION"
echo "Exclude types: $EXCLUDE_TYPES"
echo "Exclude predicates: $EXCLUDE_PREDICATES"
echo ""

# Check if file list exists
if [ ! -f "$FILE_LIST_PATH" ]; then
    echo "ERROR: File list not found: $FILE_LIST_PATH"
    exit 1
fi

# Check if explicit-count shard dir exists
if [ ! -d "$EXPLICIT_SHARDS_DIR" ]; then
    echo "ERROR: Explicit-count shard directory not found: $EXPLICIT_SHARDS_DIR"
    echo "Run precompute_aggregated_counts_slurm.py first."
    exit 1
fi

# Check if type node counts file exists
if [ ! -f "$TYPE_NODE_COUNTS" ]; then
    echo "ERROR: Type node counts file not found: $TYPE_NODE_COUNTS"
    echo "Run prepare_grouping.py first to generate this file."
    exit 1
fi

# Count files to scan
NUM_FILES=$(wc -l < "$FILE_LIST_PATH")
echo "Files to scan: $NUM_FILES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Verify matrices directory exists (required for exact pair-set evaluation)
if [ ! -d "$MATRICES_DIR" ]; then
    echo "ERROR: Matrices directory not found: $MATRICES_DIR"
    echo "The matrices directory is required for direct matrix evaluation."
    exit 1
fi
echo "Using matrices for direct evaluation: $MATRICES_DIR"

# Run Python worker
uv run python src/pipeline/workers/run_grouping.py \
    --type1 "$TYPE1" \
    --type2 "$TYPE2" \
    --file-list "$FILE_LIST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-hops "$N_HOPS" \
    --explicit-shards-dir "$EXPLICIT_SHARDS_DIR" \
    --type-node-counts "$TYPE_NODE_COUNTS" \
    --min-count "$MIN_COUNT" \
    --min-precision "$MIN_PRECISION" \
    --exclude-types "$EXCLUDE_TYPES" \
    --exclude-predicates "$EXCLUDE_PREDICATES" \
    --progress-file "$PROGRESS_FILE" \
    --matrices-dir "$MATRICES_DIR"

echo ""
echo "✓ Grouping complete"
