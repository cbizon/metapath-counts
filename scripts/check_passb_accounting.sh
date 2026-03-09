#!/bin/bash
# Usage: ./scripts/check_passb_accounting.sh <job_id>

set -euo pipefail

JOB_ID="${1:-}"
if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 <job_id>" >&2
  exit 1
fi

command -v scontrol >/dev/null 2>&1 || { echo "ERROR: scontrol not available" >&2; exit 1; }
command -v squeue >/dev/null 2>&1 || { echo "ERROR: squeue not available" >&2; exit 1; }
command -v sacct >/dev/null 2>&1 || { echo "ERROR: sacct not available" >&2; exit 1; }

# Extract ArrayTaskId spec from scontrol output
ARRAY_SPEC=$(scontrol show job "$JOB_ID" \
  | tr ' ' '\n' \
  | grep '^ArrayTaskId=' \
  | head -1 \
  | cut -d= -f2)

if [ -z "$ARRAY_SPEC" ]; then
  echo "ERROR: ArrayTaskId not found for job $JOB_ID" >&2
  exit 1
fi

# Example: 0-511%200 or 103-511%200 (pending range only)
RANGE=${ARRAY_SPEC%%%*}
THROTTLE=${ARRAY_SPEC#*%}
if [ "$RANGE" = "$ARRAY_SPEC" ]; then
  THROTTLE=""
fi

START=${RANGE%-*}
END=${RANGE#*-}

# Assume array starts at 0 if START>0 (scontrol shows pending range only)
if [ "$START" -gt 0 ]; then
  TOTAL=$((END + 1))
else
  TOTAL=$((END - START + 1))
fi

RUNNING=$(squeue -j "$JOB_ID" -h -t RUNNING | wc -l)
PENDING=$(squeue -j "$JOB_ID" -h -t PENDING | wc -l)
ACTIVE=$((RUNNING + PENDING))

# Finished states breakdown from sacct (array tasks only)
FINISHED_BREAKDOWN=$(sacct -j "$JOB_ID" --format=JobID,State --parsable2 \
  | awk -F'|' 'NR>1 {print $1,$2}' \
  | awk -v id="$JOB_ID" '$1 ~ "^"id"_[0-9]+$" && $2 !~ /RUNNING|PENDING/ {print $2}' \
  | sort | uniq -c | sort -nr)

FINISHED=0
if [ -n "$FINISHED_BREAKDOWN" ]; then
  FINISHED=$(echo "$FINISHED_BREAKDOWN" | awk '{sum+=$1} END {print sum+0}')
fi

NOT_STARTED=$((TOTAL - ACTIVE - FINISHED))
if [ "$NOT_STARTED" -lt 0 ]; then
  NOT_STARTED=0
fi

if [ -n "$THROTTLE" ]; then
  echo "Array: $RANGE (total $TOTAL), throttle $THROTTLE"
else
  echo "Array: $RANGE (total $TOTAL)"
fi

echo "RUNNING=$RUNNING PENDING=$PENDING FINISHED=$FINISHED NOT_STARTED=$NOT_STARTED"

if [ -n "$FINISHED_BREAKDOWN" ]; then
  echo "Finished states:"
  echo "$FINISHED_BREAKDOWN"
fi
