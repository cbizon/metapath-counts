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

# Build 2-hop DAG from 1-hop via per-join SLURM shard jobs (submit + wait + merge).
./scripts/submit_multihop_shard_jobs.sh "$OUT1" "$OUT1" "$OUT2"
./scripts/merge_multihop_shard_outputs.sh "$OUT2"

# Build 3-hop DAG from 2-hop + 1-hop via per-join SLURM shard jobs (submit + wait + merge).
./scripts/submit_multihop_shard_jobs.sh "$OUT2" "$OUT1" "$OUT3" lowpri 128
./scripts/merge_multihop_shard_outputs.sh "$OUT3"
