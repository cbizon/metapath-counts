#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--skip-right-shards] OUT_DIR" >&2
  exit 1
fi

SKIP_RIGHT_SHARDS=0
OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-right-shards)
      SKIP_RIGHT_SHARDS=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--skip-right-shards] OUT_DIR" >&2
      exit 1
      ;;
    *)
      if [[ -n "$OUT_DIR" ]]; then
        echo "Unexpected extra argument: $1" >&2
        echo "Usage: $0 [--skip-right-shards] OUT_DIR" >&2
        exit 1
      fi
      OUT_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  echo "OUT_DIR is required." >&2
  echo "Usage: $0 [--skip-right-shards] OUT_DIR" >&2
  exit 1
fi
SHARD_OUT_ROOT="$OUT_DIR/shard_jobs"
SUBMITTED_PATH="$OUT_DIR/shards/submitted_jobs.tsv"
EDGES_OUT="$OUT_DIR/edges.bin"
NODE_IDS_OUT="$OUT_DIR/node_ids.tsv.gz"

# Raise soft nofile as early as possible so we know immediately if the shell context can support many shard files.
soft_nofile_before=$(ulimit -Sn)
hard_nofile=$(ulimit -Hn)
if [[ "$soft_nofile_before" -lt "$hard_nofile" ]]; then
  ulimit -n "$hard_nofile" || true
fi
soft_nofile_after=$(ulimit -Sn)

echo "[merge_multihop] start out_dir=$OUT_DIR" >&2
echo "[merge_multihop] nofile soft_before=$soft_nofile_before hard=$hard_nofile soft_after=$soft_nofile_after" >&2

if [[ ! -d "$SHARD_OUT_ROOT" ]]; then
  echo "Shard output directory not found: $SHARD_OUT_ROOT" >&2
  exit 1
fi

if [[ ! -f "$SUBMITTED_PATH" ]]; then
  echo "Submitted jobs manifest not found: $SUBMITTED_PATH" >&2
  exit 1
fi

total_jobs=$(awk 'NR>1 && NF>0 {c++} END{print c+0}' "$SUBMITTED_PATH")
echo "[merge_multihop] validating shard outputs from manifest ($total_jobs jobs)" >&2

missing=0
checked=0
while IFS=$'\t' read -r job_id join_type shard_path nhop_count; do
  [[ -z "${job_id:-}" ]] && continue
  [[ "$job_id" == "job_id" ]] && continue
  checked=$((checked + 1))
  shard_base="$(basename "$shard_path" .tsv)"
  shard_key="${shard_base#nhop_}"
  shard_dir="$SHARD_OUT_ROOT/$shard_key"
  shard_edges_bin="$shard_dir/edges.bin"
  shard_edges_tsv="$shard_dir/edges.tsv"
  shard_node_ids="$shard_dir/node_ids.tsv"
  shard_node_ids_gz="$shard_dir/node_ids.tsv.gz"

  if [[ ( ! -f "$shard_node_ids" && ! -f "$shard_node_ids_gz" ) || ( ! -f "$shard_edges_bin" && ! -f "$shard_edges_tsv" ) ]]; then
    echo "Missing shard output for job $job_id join_type=$join_type: expected one of $shard_node_ids or $shard_node_ids_gz and one of $shard_edges_bin or $shard_edges_tsv" >&2
    missing=$((missing + 1))
  fi
  if (( checked % 25 == 0 )) || (( checked == total_jobs )); then
    echo "[merge_multihop] validated $checked/$total_jobs shards (missing=$missing)" >&2
  fi
done < "$SUBMITTED_PATH"

if [[ "$missing" -gt 0 ]]; then
  echo "Refusing to merge: $missing shard outputs missing" >&2
  exit 1
fi

echo "[merge_multihop] merging shard outputs -> $NODE_IDS_OUT, $EDGES_OUT" >&2
uv run python src/pipeline/merge_multihop_shards.py --out-dir "$OUT_DIR"

SHARDS_RIGHT_LOG="$OUT_DIR/shards_right.materialize.log"
if [[ "$SKIP_RIGHT_SHARDS" -eq 1 ]]; then
  echo "[merge_multihop] skipping right-shard materialization by request" >&2
else
  echo "[merge_multihop] materializing persistent right-sharded artifacts in $OUT_DIR/shards_right (log: $SHARDS_RIGHT_LOG)" >&2
  uv run python src/pipeline/dag_shards.py --dag-dir "$OUT_DIR" --right 2>&1 | tee "$SHARDS_RIGHT_LOG"
fi

echo "[merge_multihop] done merged shard outputs into $NODE_IDS_OUT and $EDGES_OUT" >&2
