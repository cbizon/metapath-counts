#!/usr/bin/env bash
set -euo pipefail

BASE="../translator_kg/Feb_13_filtered_nonredundant"
OUT1="DAG1_data_filtered"
OUT2="DAG2_data_filtered"
OUT3="DAG3_data_filtered"

# Build 1-hop DAG (filtered)
#uv run python src/pipeline/build_onehop_dag.py \
#  --output-dir "$OUT1" \
#  --edges "$BASE/edges.jsonl" \
#  --nodes "$BASE/nodes.jsonl"

# Build 2-hop DAG from 1-hop
uv run python src/pipeline/build_multihop_dag.py \
  --nhop-dir "$OUT1" \
  --onehop-dir "$OUT1" \
  --output-dir "$OUT2" \
  --shard-by-join

# Build 3-hop DAG from 2-hop + 1-hop
uv run python src/pipeline/build_multihop_dag.py \
  --nhop-dir "$OUT2" \
  --onehop-dir "$OUT1" \
  --output-dir "$OUT3" \
  --shard-by-join
