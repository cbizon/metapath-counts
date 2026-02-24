#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 NHOP_DIR ONEHOP_DIR OUT_DIR [PARTITION] [MEM_GB] [TIME]" >&2
  exit 1
fi

NHOP_DIR="$1"
ONEHOP_DIR="$2"
OUT_DIR="$3"
PARTITION="${4:-lowpri}"
MEM_GB="${5:-64}"
TIME_LIMIT="${6:-24:00:00}"
CPUS="${CPUS:-1}"
POLL_SECONDS="${POLL_SECONDS:-30}"

INDEX_PATH="$OUT_DIR/shards/index.tsv"
SLURM_LOG_DIR="$OUT_DIR/slurm"
SHARD_OUT_ROOT="$OUT_DIR/shard_jobs"
mkdir -p "$SLURM_LOG_DIR" "$SHARD_OUT_ROOT"
SUBMITTED_PATH="$OUT_DIR/shards/submitted_jobs.tsv"

if [[ ! -f "$INDEX_PATH" ]]; then
  echo "Preparing join-type shards into $OUT_DIR/shards" >&2
  uv run python src/pipeline/build_multihop_dag.py \
    --nhop-dir "$NHOP_DIR" \
    --onehop-dir "$ONEHOP_DIR" \
    --output-dir "$OUT_DIR" \
    --shard-by-join \
    --prepare-shards-only
fi

submitted=0
: > "$SUBMITTED_PATH"
while IFS=$'\t' read -r join_type shard_path nhop_count; do
  if [[ "$join_type" == "join_type" ]]; then
    continue
  fi

  shard_base="$(basename "$shard_path" .tsv)"
  shard_key="${shard_base#nhop_}"
  shard_out="$SHARD_OUT_ROOT/$shard_key"
  mkdir -p "$shard_out"

  job_name="dag_${shard_key}"
  wrap_cmd="cd $(pwd) && uv run python src/pipeline/build_multihop_dag.py --nhop-dir '$NHOP_DIR' --onehop-dir '$ONEHOP_DIR' --output-dir '$shard_out' --join-type '$join_type' --nhop-shard-file '$shard_path' --log-every 0"

  job_id=$(sbatch \
    --parsable \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS" \
    --mem="${MEM_GB}G" \
    --time="$TIME_LIMIT" \
    --job-name="$job_name" \
    --output="$SLURM_LOG_DIR/%x_%j.out" \
    --wrap "$wrap_cmd")

  printf '%s\t%s\t%s\t%s\n' "$job_id" "$join_type" "$shard_path" "$nhop_count"
  printf '%s\t%s\t%s\t%s\n' "$job_id" "$join_type" "$shard_path" "$nhop_count" >> "$SUBMITTED_PATH"
  submitted=$((submitted + 1))
done < "$INDEX_PATH"

echo "Submitted $submitted shard jobs" >&2

if [[ "$submitted" -eq 0 ]]; then
  exit 0
fi

mapfile -t JOB_IDS < <(cut -f1 "$SUBMITTED_PATH")

echo "Waiting for shard jobs to finish (poll every ${POLL_SECONDS}s)..." >&2

while true; do
  running_count=0
  pending_count=0
  for job_id in "${JOB_IDS[@]}"; do
    state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)"
    if [[ -z "$state" ]]; then
      continue
    fi
    if [[ "$state" == "RUNNING" ]]; then
      running_count=$((running_count + 1))
    elif [[ "$state" == "PENDING" || "$state" == "CONFIGURING" || "$state" == "COMPLETING" ]]; then
      pending_count=$((pending_count + 1))
    else
      pending_count=$((pending_count + 1))
    fi
  done

  completed=0
  failed=0
  done_count=0
  if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
    sacct_out="$(sacct -n -P -j "$(IFS=,; echo "${JOB_IDS[*]}")" --format=JobID,State,ExitCode 2>/dev/null || true)"
    while IFS='|' read -r jid state exit_code; do
      [[ -z "${jid:-}" ]] && continue
      [[ "$jid" == *.* ]] && continue
      case "$state" in
        COMPLETED)
          completed=$((completed + 1))
          ;;
        FAILED|OUT_OF_MEMORY|CANCELLED|TIMEOUT|NODE_FAIL)
          failed=$((failed + 1))
          ;;
      esac
      case "$state" in
        COMPLETED|FAILED|OUT_OF_MEMORY|CANCELLED|TIMEOUT|NODE_FAIL)
          done_count=$((done_count + 1))
          ;;
      esac
    done <<< "$sacct_out"
  fi

  total="${#JOB_IDS[@]}"
  echo "Shard jobs: total=$total done=$done_count completed=$completed failed=$failed running=$running_count pending=$pending_count" >&2

  if [[ "$failed" -gt 0 ]]; then
    echo "One or more shard jobs failed. Inspect $SUBMITTED_PATH and $SLURM_LOG_DIR" >&2
    exit 1
  fi

  if [[ "$done_count" -ge "$total" ]]; then
    echo "All shard jobs completed successfully." >&2
    break
  fi

  sleep "$POLL_SECONDS"
done
