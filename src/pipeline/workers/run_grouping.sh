#!/bin/bash
#SBATCH --job-name=group_typepair
#SBATCH --time=2:00:00
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
#
# This script:
# 1. Receives type pair and file list from orchestrator (no manifest access!)
# 2. Calls Python worker to stream through specified files
# 3. Finds all 1-hop metapaths between type1 and type2
# 4. Aggregates and computes metrics for each
# 5. Applies filters (min-count, min-precision, excluded types/predicates)
# 6. Outputs to grouped_by_results_{n_hops}hop/<sanitized_typepair>/

set -e

TYPE1="$1"
TYPE2="$2"
FILE_LIST_PATH="$3"
N_HOPS="$4"
MIN_COUNT="${5:-0}"
MIN_PRECISION="${6:-0}"
EXCLUDE_TYPES="${7:-Entity,ThingWithTaxon}"
EXCLUDE_PREDICATES="${8:-related_to_at_instance_level,related_to_at_concept_level}"

if [ -z "$TYPE1" ] || [ -z "$TYPE2" ] || [ -z "$FILE_LIST_PATH" ] || [ -z "$N_HOPS" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 <type1> <type2> <file_list_path> <n_hops> [min_count] [min_precision] [exclude_types] [exclude_predicates]"
    exit 1
fi

OUTPUT_DIR="grouped_by_results_${N_HOPS}hop"
RESULTS_DIR="results_${N_HOPS}hop"
AGGREGATED_NHOP_COUNTS="${RESULTS_DIR}/aggregated_nhop_counts.json"
TYPE_NODE_COUNTS="${RESULTS_DIR}/type_node_counts.json"

echo "=========================================="
echo "GROUPING TYPE PAIR"
echo "=========================================="
echo "Type pair: ($TYPE1, $TYPE2)"
echo "File list: $FILE_LIST_PATH"
echo "N-hops: $N_HOPS"
echo "Output dir: $OUTPUT_DIR"
echo "Aggregated N-hop counts: $AGGREGATED_NHOP_COUNTS"
echo "Type node counts: $TYPE_NODE_COUNTS"
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

# Check if aggregated N-hop counts file exists
if [ ! -f "$AGGREGATED_NHOP_COUNTS" ]; then
    echo "ERROR: Aggregated N-hop counts file not found: $AGGREGATED_NHOP_COUNTS"
    echo "Run prepare_grouping.py first to generate this file."
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

# Run Python worker
uv run python src/pipeline/workers/run_grouping.py \
    --type1 "$TYPE1" \
    --type2 "$TYPE2" \
    --file-list "$FILE_LIST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-hops "$N_HOPS" \
    --aggregated-nhop-counts "$AGGREGATED_NHOP_COUNTS" \
    --type-node-counts "$TYPE_NODE_COUNTS" \
    --min-count "$MIN_COUNT" \
    --min-precision "$MIN_PRECISION" \
    --exclude-types "$EXCLUDE_TYPES" \
    --exclude-predicates "$EXCLUDE_PREDICATES"

echo ""
echo "✓ Grouping complete"
