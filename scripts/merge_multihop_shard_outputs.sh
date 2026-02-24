#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 OUT_DIR" >&2
  exit 1
fi

OUT_DIR="$1"
SHARD_OUT_ROOT="$OUT_DIR/shard_jobs"
SUBMITTED_PATH="$OUT_DIR/shards/submitted_jobs.tsv"
NODES_OUT="$OUT_DIR/nodes.tsv"
EDGES_OUT="$OUT_DIR/edges.tsv"

if [[ ! -d "$SHARD_OUT_ROOT" ]]; then
  echo "Shard output directory not found: $SHARD_OUT_ROOT" >&2
  exit 1
fi

if [[ ! -f "$SUBMITTED_PATH" ]]; then
  echo "Submitted jobs manifest not found: $SUBMITTED_PATH" >&2
  exit 1
fi

missing=0
while IFS=$'\t' read -r job_id join_type shard_path nhop_count; do
  [[ -z "${job_id:-}" ]] && continue
  shard_base="$(basename "$shard_path" .tsv)"
  shard_key="${shard_base#nhop_}"
  shard_dir="$SHARD_OUT_ROOT/$shard_key"
  shard_nodes="$shard_dir/nodes.tsv"
  shard_edges="$shard_dir/edges.tsv"

  if [[ ! -f "$shard_nodes" || ! -f "$shard_edges" ]]; then
    echo "Missing shard output for job $job_id join_type=$join_type: expected $shard_nodes and $shard_edges" >&2
    missing=$((missing + 1))
  fi
done < "$SUBMITTED_PATH"

if [[ "$missing" -gt 0 ]]; then
  echo "Refusing to merge: $missing shard outputs missing" >&2
  exit 1
fi

printf 'metapath\n' > "$NODES_OUT"
printf 'child\tparent\n' > "$EDGES_OUT"

find "$SHARD_OUT_ROOT" -mindepth 2 -maxdepth 2 -name nodes.tsv -print0 \
  | sort -z \
  | while IFS= read -r -d '' f; do
      tail -n +2 "$f" >> "$NODES_OUT"
    done

find "$SHARD_OUT_ROOT" -mindepth 2 -maxdepth 2 -name edges.tsv -print0 \
  | sort -z \
  | while IFS= read -r -d '' f; do
      tail -n +2 "$f" >> "$EDGES_OUT"
    done

echo "Merged shard outputs into $NODES_OUT and $EDGES_OUT" >&2
