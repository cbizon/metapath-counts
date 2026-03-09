#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 NHOP_DIR ONEHOP_DIR OUT_DIR [PARTITION] [MEM_GB] [TIME] [EXCLUDE_TYPES] [EXCLUDE_PREDICATES]" >&2
  exit 1
fi

NHOP_DIR="$1"
ONEHOP_DIR="$2"
OUT_DIR="$3"
PARTITION="${4:-lowpri}"
MEM_GB="${5:-64}"
TIME_LIMIT="${6:-24:00:00}"
EXCLUDE_TYPES="${7:-${EXCLUDE_TYPES:-}}"
EXCLUDE_PREDICATES="${8:-${EXCLUDE_PREDICATES:-}}"
CPUS="${CPUS:-1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_MEM_GB="${MAX_MEM_GB:-1400}"
MIN_MEM_GB="${MIN_MEM_GB:-256}"
MAX_PENDING_JOBS="${MAX_PENDING_JOBS:-1}"

MULTIHOP_EXCLUDE_ARGS=()
if [[ -n "$EXCLUDE_TYPES" ]]; then
  MULTIHOP_EXCLUDE_ARGS+=(--exclude-types "$EXCLUDE_TYPES")
fi
if [[ -n "$EXCLUDE_PREDICATES" ]]; then
  MULTIHOP_EXCLUDE_ARGS+=(--exclude-predicates "$EXCLUDE_PREDICATES")
fi

INDEX_PATH="$OUT_DIR/shards/index.tsv"
NHOP_PERSISTENT_INDEX="$NHOP_DIR/shards_right/index.tsv"
SLURM_LOG_DIR="$OUT_DIR/slurm"
SHARD_OUT_ROOT="$OUT_DIR/shard_jobs"
SHARD_META_DIR="$OUT_DIR/shards"
mkdir -p "$SLURM_LOG_DIR" "$SHARD_OUT_ROOT" "$SHARD_META_DIR"
SUBMITTED_PATH="$OUT_DIR/shards/submitted_jobs.tsv"
ATTEMPTS_PATH="$OUT_DIR/shards/submitted_attempts.tsv"

if [[ -f "$NHOP_PERSISTENT_INDEX" ]]; then
  echo "Reusing persistent NHOP right-shards from $NHOP_PERSISTENT_INDEX" >&2
  INDEX_PATH="$NHOP_PERSISTENT_INDEX"
elif [[ ! -f "$INDEX_PATH" ]]; then
  echo "Preparing join-type shards into $OUT_DIR/shards" >&2
  uv run python src/pipeline/build_multihop_dag.py \
    --nhop-dir "$NHOP_DIR" \
    --onehop-dir "$ONEHOP_DIR" \
    --output-dir "$OUT_DIR" \
    --shard-by-join \
    --prepare-shards-only \
    "${MULTIHOP_EXCLUDE_ARGS[@]}"
fi

is_heavy_join_type() {
  local jt="$1"
  [[ "$jt" == "Entity" || "$jt" == "NamedThing" || "$jt" == "PhysicalEssence" ]]
}

next_memory_tier() {
  local current="$1"
  local tiers=(256 500 1000 1400)
  local t
  for t in "${tiers[@]}"; do
    if (( current < t )); then
      echo "$t"
      return 0
    fi
  done
  return 1
}

effective_partition_for_mem() {
  local base_partition="$1"
  local mem_gb="$2"
  if [[ "$base_partition" == "lowpri" && "$mem_gb" -gt 1000 ]]; then
    echo "largemem"
  else
    echo "$base_partition"
  fi
}

submit_shard_job() {
  local shard_key="$1"
  local join_type="$2"
  local shard_path="$3"
  local nhop_count="$4"
  local mem_gb="$5"
  local attempt="$6"

  local shard_out="$SHARD_OUT_ROOT/$shard_key"
  mkdir -p "$shard_out"

  local job_name="dag_${shard_key}"
  local eff_partition
  eff_partition="$(effective_partition_for_mem "$PARTITION" "$mem_gb")"
  local wrap_cmd="cd $(pwd) && uv run python src/pipeline/build_multihop_dag.py --nhop-dir '$NHOP_DIR' --onehop-dir '$ONEHOP_DIR' --output-dir '$shard_out' --join-type '$join_type' --nhop-shard-file '$shard_path' --log-every 0"
  if [[ -n "$EXCLUDE_TYPES" ]]; then
    wrap_cmd+=" --exclude-types '$EXCLUDE_TYPES'"
  fi
  if [[ -n "$EXCLUDE_PREDICATES" ]]; then
    wrap_cmd+=" --exclude-predicates '$EXCLUDE_PREDICATES'"
  fi

  local job_id
  job_id=$(sbatch \
    --parsable \
    --partition="$eff_partition" \
    --cpus-per-task="$CPUS" \
    --mem="${mem_gb}G" \
    --time="$TIME_LIMIT" \
    --job-name="$job_name" \
    --output="$SLURM_LOG_DIR/%x_%j.out" \
    --wrap "$wrap_cmd")

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job_id" "$join_type" "$shard_path" "$nhop_count" "$mem_gb" "$attempt" "$eff_partition" >> "$ATTEMPTS_PATH"
  echo "$job_id"
}

submitted=0
: > "$SUBMITTED_PATH"
: > "$ATTEMPTS_PATH"

declare -A SHARD_JOIN_TYPE
declare -A SHARD_PATH
declare -A SHARD_NHOP_COUNT
declare -A SHARD_MEM_GB
declare -A SHARD_ATTEMPTS
declare -A SHARD_JOB_ID
declare -A SHARD_DONE
declare -A SHARD_SUCCESS
declare -A SHARD_LAST_STATE
declare -A SHARD_LAST_EXIT
declare -A SHARD_LAST_PARTITION

declare -A JOB_TO_SHARD
declare -a SHARD_KEYS

while IFS=$'\t' read -r join_type shard_path nhop_count _rest; do
  if [[ "$join_type" == "join_type" ]]; then
    continue
  fi

  shard_base="$(basename "$shard_path" .tsv)"
  shard_key="${shard_base#nhop_}"
  shard_mem="$MEM_GB"
  if (( shard_mem < MIN_MEM_GB )); then
    shard_mem="$MIN_MEM_GB"
  fi
  if is_heavy_join_type "$join_type"; then
    shard_mem=$((shard_mem * 2))
  fi
  if (( shard_mem > MAX_MEM_GB )); then
    shard_mem="$MAX_MEM_GB"
  fi

  SHARD_JOIN_TYPE["$shard_key"]="$join_type"
  SHARD_PATH["$shard_key"]="$shard_path"
  SHARD_NHOP_COUNT["$shard_key"]="$nhop_count"
  SHARD_MEM_GB["$shard_key"]="$shard_mem"
  SHARD_ATTEMPTS["$shard_key"]=1
  SHARD_DONE["$shard_key"]=0
  SHARD_SUCCESS["$shard_key"]=0
  SHARD_JOB_ID["$shard_key"]=""
  SHARD_LAST_PARTITION["$shard_key"]="$(effective_partition_for_mem "$PARTITION" "$shard_mem")"
  SHARD_KEYS+=("$shard_key")
done < "$INDEX_PATH"

total="${#SHARD_KEYS[@]}"
echo "Prepared $total shard jobs (max_pending_jobs=$MAX_PENDING_JOBS min_mem_gb=$MIN_MEM_GB)" >&2

if [[ "$total" -eq 0 ]]; then
  exit 0
fi

echo "Waiting for shard jobs to finish (poll every ${POLL_SECONDS}s)..." >&2

final_sacct_out=""
pending_cursor=0

submit_more_jobs() {
  local pending_now="$1"
  local shard_key
  while (( pending_now < MAX_PENDING_JOBS && pending_cursor < total )); do
    shard_key="${SHARD_KEYS[$pending_cursor]}"
    pending_cursor=$((pending_cursor + 1))
    if [[ "${SHARD_DONE[$shard_key]}" -eq 1 ]]; then
      continue
    fi
    if [[ -n "${SHARD_JOB_ID[$shard_key]:-}" ]]; then
      continue
    fi
    job_id="$(submit_shard_job \
      "$shard_key" \
      "${SHARD_JOIN_TYPE[$shard_key]}" \
      "${SHARD_PATH[$shard_key]}" \
      "${SHARD_NHOP_COUNT[$shard_key]}" \
      "${SHARD_MEM_GB[$shard_key]}" \
      "${SHARD_ATTEMPTS[$shard_key]}")"
    SHARD_JOB_ID["$shard_key"]="$job_id"
    JOB_TO_SHARD["$job_id"]="$shard_key"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "${SHARD_JOIN_TYPE[$shard_key]}" "${SHARD_PATH[$shard_key]}" "${SHARD_NHOP_COUNT[$shard_key]}"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "${SHARD_JOIN_TYPE[$shard_key]}" "${SHARD_PATH[$shard_key]}" "${SHARD_NHOP_COUNT[$shard_key]}" >> "$SUBMITTED_PATH"
    submitted=$((submitted + 1))
    pending_now=$((pending_now + 1))
  done
}

submit_more_jobs 0
echo "Submitted $submitted/$total shard jobs (initial)" >&2

while true; do
  running_count=0
  pending_count=0
  active_job_ids=()
  for shard_key in "${SHARD_KEYS[@]}"; do
    if [[ "${SHARD_DONE[$shard_key]}" -eq 1 ]]; then
      continue
    fi
    job_id="${SHARD_JOB_ID[$shard_key]}"
    if [[ -z "${job_id:-}" ]]; then
      pending_count=$((pending_count + 1))
      continue
    fi
    active_job_ids+=("$job_id")
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
  if [[ "${#active_job_ids[@]}" -gt 0 ]]; then
    sacct_out="$(sacct -n -P -j "$(IFS=,; echo "${active_job_ids[*]}")" --format=JobID,State,ExitCode 2>/dev/null || true)"
    final_sacct_out="$sacct_out"
    while IFS='|' read -r jid state exit_code; do
      [[ -z "${jid:-}" ]] && continue
      [[ "$jid" == *.* ]] && continue
      shard_key="${JOB_TO_SHARD[$jid]:-}"
      [[ -z "$shard_key" ]] && continue

      if [[ "${SHARD_DONE[$shard_key]}" -eq 0 ]]; then
        case "$state" in
          COMPLETED)
            SHARD_DONE["$shard_key"]=1
            SHARD_SUCCESS["$shard_key"]=1
            SHARD_LAST_STATE["$shard_key"]="$state"
            SHARD_LAST_EXIT["$shard_key"]="$exit_code"
            ;;
          OUT_OF_MEMORY)
            current_mem="${SHARD_MEM_GB[$shard_key]}"
            if next_mem="$(next_memory_tier "$current_mem")"; then
              if (( next_mem > MAX_MEM_GB )); then
                SHARD_DONE["$shard_key"]=1
                SHARD_SUCCESS["$shard_key"]=0
                SHARD_LAST_STATE["$shard_key"]="OUT_OF_MEMORY"
                SHARD_LAST_EXIT["$shard_key"]="$exit_code"
              else
                attempt=$(( SHARD_ATTEMPTS[$shard_key] + 1 ))
                SHARD_ATTEMPTS["$shard_key"]="$attempt"
                SHARD_MEM_GB["$shard_key"]="$next_mem"
                SHARD_JOB_ID["$shard_key"]=""
                SHARD_LAST_PARTITION["$shard_key"]="$(effective_partition_for_mem "$PARTITION" "$next_mem")"
                pending_count=$((pending_count + 1))
                echo "Queued shard $shard_key for resubmit after OOM: next_mem=${next_mem}G attempt=$attempt partition=${SHARD_LAST_PARTITION[$shard_key]}" >&2
              fi
            else
              SHARD_DONE["$shard_key"]=1
              SHARD_SUCCESS["$shard_key"]=0
              SHARD_LAST_STATE["$shard_key"]="OUT_OF_MEMORY"
              SHARD_LAST_EXIT["$shard_key"]="$exit_code"
            fi
            ;;
          FAILED|CANCELLED|TIMEOUT|NODE_FAIL)
            SHARD_DONE["$shard_key"]=1
            SHARD_SUCCESS["$shard_key"]=0
            SHARD_LAST_STATE["$shard_key"]="$state"
            SHARD_LAST_EXIT["$shard_key"]="$exit_code"
            ;;
        esac
      fi

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

  done_count=0
  completed=0
  failed=0
  for shard_key in "${SHARD_KEYS[@]}"; do
    if [[ "${SHARD_DONE[$shard_key]}" -eq 1 ]]; then
      done_count=$((done_count + 1))
      if [[ "${SHARD_SUCCESS[$shard_key]}" -eq 1 ]]; then
        completed=$((completed + 1))
      else
        failed=$((failed + 1))
      fi
    fi
  done
  queued_count=$((total - done_count - running_count - pending_count))
  if (( queued_count < 0 )); then queued_count=0; fi

  submit_more_jobs "$pending_count"
  if (( submitted > 0 )); then
    :
  fi

  echo "Shard jobs: total=$total submitted=$submitted done=$done_count completed=$completed failed=$failed running=$running_count pending=$pending_count queued=$queued_count" >&2

  if [[ "$done_count" -ge "$total" ]]; then
    if [[ "$failed" -gt 0 ]]; then
      echo "Shard job failures detected:" >&2
      for shard_key in "${!SHARD_DONE[@]}"; do
        if [[ "${SHARD_SUCCESS[$shard_key]}" -eq 1 ]]; then
          continue
        fi
        printf '  job_id=%s join_type=%s state=%s exit_code=%s nhop_count=%s mem=%sG attempts=%s partition=%s shard=%s\n' \
          "${SHARD_JOB_ID[$shard_key]}" \
          "${SHARD_JOIN_TYPE[$shard_key]}" \
          "${SHARD_LAST_STATE[$shard_key]:-UNKNOWN}" \
          "${SHARD_LAST_EXIT[$shard_key]:-UNKNOWN}" \
          "${SHARD_NHOP_COUNT[$shard_key]}" \
          "${SHARD_MEM_GB[$shard_key]}" \
          "${SHARD_ATTEMPTS[$shard_key]}" \
          "${SHARD_LAST_PARTITION[$shard_key]:-UNKNOWN}" \
          "${SHARD_PATH[$shard_key]}" >&2
      done
      echo "Inspect SLURM logs in $SLURM_LOG_DIR" >&2
      exit 1
    fi
    echo "All shard jobs completed successfully." >&2
    break
  fi

  sleep "$POLL_SECONDS"
done
