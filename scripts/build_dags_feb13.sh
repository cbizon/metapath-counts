#!/usr/bin/env bash
set -euo pipefail

BASE="../translator_kg/Feb_13_filtered_nonredundant"
OUT1="DAG1_data_filtered"
OUT2="DAG2_data_filtered"
OUT3="DAG3_data_filtered"

# Compression filters (comma-separated, normalized by the Python builders).
# Defaults match the standard grouping workflow excludes.
EXCLUDE_TYPES="${EXCLUDE_TYPES:-Entity,ThingWithTaxon,PhysicalEssence,PhysicalEssenceOrOccurrent,Occurrent}"
EXCLUDE_PREDICATES="${EXCLUDE_PREDICATES:-related_to_at_instance_level,related_to_at_concept_level}"

# Build 1-hop DAG (filtered)
#uv run python src/pipeline/build_onehop_dag.py \
#  --output-dir "$OUT1" \
#  --edges "$BASE/edges.jsonl" \
#  --nodes "$BASE/nodes.jsonl" \
#  ${EXCLUDE_TYPES:+--exclude-types "$EXCLUDE_TYPES"} \
#  ${EXCLUDE_PREDICATES:+--exclude-predicates "$EXCLUDE_PREDICATES"}

# Build 2-hop DAG from 1-hop via per-join SLURM shard jobs (submit + wait + merge).
#./scripts/submit_multihop_shard_jobs.sh \
#  "$OUT1" "$OUT1" "$OUT2" \
#  lowpri 64 24:00:00 \
#  "$EXCLUDE_TYPES" "$EXCLUDE_PREDICATES"
#./scripts/merge_multihop_shard_outputs.sh "$OUT2"

# Build 3-hop DAG from 2-hop + 1-hop via per-join SLURM shard jobs (submit + wait + merge).
./scripts/submit_multihop_shard_jobs.sh \
  "$OUT2" "$OUT1" "$OUT3" \
  lowpri 256 24:00:00 \
  "$EXCLUDE_TYPES" "$EXCLUDE_PREDICATES"
./scripts/merge_multihop_shard_outputs.sh --skip-right-shards "$OUT3"
