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

uv run python scripts/prepare_grouping.py --n-hops "$N_HOPS"
uv run python scripts/orchestrate_grouping.py --n-hops "$N_HOPS" \
    --min-count 10 \
    --min-precision 0.001
