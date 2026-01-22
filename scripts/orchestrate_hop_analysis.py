#!/usr/bin/env python3
"""
Orchestrate parallel N-hop metapath analysis across SLURM cluster.

This script:
1. Monitors manifest.json for pending jobs
2. Submits jobs to SLURM with appropriate memory tiers
3. Polls job status and updates manifest
4. Auto-retries OOM failures at higher memory tier (180GB -> 250GB -> 500GB -> 1TB -> 1.5TB)
5. Maintains max 1 pending job to avoid queue hogging

Usage:
    # 3-hop analysis (default)
    uv run python scripts/orchestrate_hop_analysis.py

    # Custom N-hop analysis
    uv run python scripts/orchestrate_hop_analysis.py --n-hops 2
"""

import json
import subprocess
import time
import os
import sys
from datetime import datetime
from collections import defaultdict


WORKER_SCRIPT = "scripts/run_single_matrix1.sh"
POLL_INTERVAL = 30  # seconds


def get_max_concurrent_jobs(n_hops):
    """Get max concurrent jobs based on n_hops.

    Lower hop counts complete faster with pre-built matrices,
    so we can run more in parallel without overwhelming the queue.

    Args:
        n_hops: Number of hops being analyzed

    Returns:
        Max number of concurrent PENDING jobs
    """
    if n_hops == 1:
        return 100
    elif n_hops == 2:
        return 30
    else:  # n_hops >= 3
        return 10


def get_manifest_path(n_hops):
    """Get manifest path for given n_hops."""
    return f"results_{n_hops}hop/manifest.json"


def load_manifest(n_hops):
    """Load manifest from disk."""
    manifest_path = get_manifest_path(n_hops)
    with open(manifest_path, 'r') as f:
        return json.load(f)


def save_manifest(manifest, n_hops):
    """Save manifest to disk."""
    manifest_path = get_manifest_path(n_hops)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def update_job_status(manifest, matrix1_id, n_hops, **kwargs):
    """Update a job's status in the manifest."""
    if matrix1_id not in manifest:
        raise KeyError(f"Job {matrix1_id} not found in manifest")

    for key, value in kwargs.items():
        manifest[matrix1_id][key] = value

    manifest[matrix1_id]["last_update"] = datetime.now().isoformat()
    save_manifest(manifest, n_hops)


def get_jobs_by_status(manifest, status):
    """Get all jobs with given status."""
    return [
        (matrix1_id, data)
        for matrix1_id, data in manifest.items()
        if matrix1_id != "_metadata" and data["status"] == status
    ]


def get_pending_slurm_jobs():
    """Get list of PENDING jobs from SLURM."""
    try:
        # Get username properly
        import getpass
        username = getpass.getuser()

        result = subprocess.run(
            ["squeue", "-u", username, "-t", "PENDING", "-o", "%i", "-h"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    except Exception as e:
        print(f"Warning: Failed to query SLURM pending jobs: {e}")

    return []


def get_running_slurm_jobs():
    """Get dictionary of RUNNING jobs from SLURM: {job_id: job_name}."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-t", "RUNNING", "-o", "%i %j", "-h"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            jobs = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        jobs[parts[0]] = ' '.join(parts[1:])
            return jobs
    except Exception as e:
        print(f"Warning: Failed to query SLURM running jobs: {e}")

    return {}


def get_completed_jobs_since(job_ids, since_time):
    """
    Query SLURM sacct for completed jobs.

    Returns: dict {job_id: (state, exit_code)}
    """
    if not job_ids:
        return {}

    try:
        # Format time for sacct
        start_time = since_time.strftime("%Y-%m-%dT%H:%M:%S")

        result = subprocess.run(
            [
                "sacct",
                "-j", ",".join(job_ids),
                "--starttime", start_time,
                "--format=JobID,State,ExitCode",
                "--noheader",
                "--parsable2"
            ],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return {}

        completed = {}
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue

            parts = line.strip().split('|')
            if len(parts) < 3:
                continue

            job_id, state, exit_code = parts[0], parts[1], parts[2]

            # Skip sub-jobs (e.g., "12345.batch")
            if '.' in job_id:
                continue

            # Parse exit code (format: "0:0" or "137:0")
            exit_code_int = 0
            if ':' in exit_code:
                exit_code_int = int(exit_code.split(':')[0])

            # Only return jobs in terminal states (not RUNNING, PENDING, etc.)
            terminal_states = {"COMPLETED", "FAILED", "OUT_OF_MEMORY", "CANCELLED", "TIMEOUT", "NODE_FAIL"}
            if state in terminal_states:
                completed[job_id] = (state, exit_code_int)

        return completed

    except Exception as e:
        print(f"Warning: Failed to query sacct: {e}")
        return {}


def submit_job(matrix1_index, memory_gb, nodes_file, edges_file, n_hops=3, matrices_dir=None):
    """Submit a single matrix1 job to SLURM with specified memory.

    Args:
        matrix1_index: Index of the starting matrix
        memory_gb: Memory to request in GB
        nodes_file: Path to KGX nodes file
        edges_file: Path to KGX edges file
        n_hops: Number of hops to analyze (default: 3)
        matrices_dir: Optional directory with pre-built matrices (for faster startup)

    Returns: job_id (str) or None on failure
    """
    # Use largemem partition for jobs >1000GB (i.e., 1500GB tier)
    partition = 'largemem' if memory_gb > 1000 else 'lowpri'

    logs_dir = f"logs_{n_hops}hop"

    cmd = [
        'sbatch',
        f'--partition={partition}',
        f'--mem={memory_gb}G',
        f'--job-name={n_hops}hop_m1_{matrix1_index:03d}',
        f'--output={logs_dir}/matrix1_{matrix1_index:03d}_mem{memory_gb}.out',
        f'--error={logs_dir}/matrix1_{matrix1_index:03d}_mem{memory_gb}.err',
        WORKER_SCRIPT,
        str(matrix1_index),
        nodes_file,
        edges_file,
        str(n_hops)
    ]

    # Add matrices_dir if provided
    if matrices_dir:
        cmd.append(matrices_dir)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            print(f"ERROR: sbatch failed for matrix1_{matrix1_index:03d}: {result.stderr}")
            return None

        # Parse job ID from "Submitted batch job 12345678"
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            return job_id
        else:
            print(f"ERROR: Unexpected sbatch output: {output}")
            return None

    except Exception as e:
        print(f"ERROR: Failed to submit job for matrix1_{matrix1_index:03d}: {e}")
        return None


def increment_memory_tier(current_memory):
    """Increment memory tier for retry.

    180 -> 250 -> 500 -> 1000 -> 1400 -> None (give up)

    180GB matches most cluster nodes (~191GB available)
    Higher tiers use fewer but larger nodes
    1400GB uses largemem partition (~1495GB available)
    """
    if current_memory < 250:
        return 250
    elif current_memory < 500:
        return 500
    elif current_memory < 1000:
        return 1000
    elif current_memory < 1400:
        return 1400
    else:
        return None  # Already at max, give up


def print_status_summary(manifest, running_jobs):
    """Print current status summary."""
    status_counts = defaultdict(int)
    for key, data in manifest.items():
        if key != "_metadata":
            status_counts[data["status"]] += 1

    total = len(manifest) - 1  # Exclude _metadata
    pending = status_counts["pending"]
    running = status_counts["running"]
    completed = status_counts["completed"]
    failed = status_counts["failed"]

    pct_complete = (completed / total * 100) if total > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"STATUS: {completed}/{total} completed ({pct_complete:.1f}%) | "
          f"Running: {running} | Pending: {pending} | Failed: {failed}")
    print(f"SLURM queue: {len(running_jobs)} running jobs")
    print(f"{'=' * 80}\n")


def orchestrate(n_hops=3):
    """Main orchestration loop.

    Args:
        n_hops: Number of hops to analyze (default: 3)
    """
    manifest_path = get_manifest_path(n_hops)

    max_concurrent = get_max_concurrent_jobs(n_hops)

    print("=" * 80)
    print(f"STARTING {n_hops}-HOP METAPATH ANALYSIS ORCHESTRATOR")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Worker script: {WORKER_SCRIPT}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Max concurrent jobs: {max_concurrent}")
    print(f"N-hops: {n_hops}")
    print()

    # Load manifest
    manifest = load_manifest(n_hops)

    # Extract input file paths from metadata
    if "_metadata" not in manifest:
        print("ERROR: Manifest missing _metadata with input file paths!")
        print("Please re-run: uv run python scripts/prepare_analysis.py")
        sys.exit(1)

    nodes_file = manifest["_metadata"]["nodes_file"]
    edges_file = manifest["_metadata"]["edges_file"]
    matrices_dir = manifest["_metadata"].get("matrices_dir", None)

    # Get n_hops from manifest if not explicitly provided
    if "_metadata" in manifest and "n_hops" in manifest["_metadata"]:
        manifest_n_hops = manifest["_metadata"]["n_hops"]
        if n_hops != manifest_n_hops:
            print(f"WARNING: CLI n_hops ({n_hops}) differs from manifest ({manifest_n_hops})")
            print(f"Using manifest value: {manifest_n_hops}")
            n_hops = manifest_n_hops

    print(f"\nInput data from manifest:")
    print(f"  Nodes: {nodes_file}")
    print(f"  Edges: {edges_file}")
    print(f"  N-hops: {n_hops}")
    if matrices_dir:
        print(f"  Pre-built matrices: {matrices_dir}")

    total_jobs = len(manifest) - 1  # Subtract 1 for _metadata entry
    print(f"\nLoaded manifest with {total_jobs} jobs")

    # Auto-reset failed jobs to pending at startup (for OOM retry)
    # This allows jobs that failed with GraphBLAS OOM to be retried automatically
    reset_count = 0
    for matrix1_id, data in manifest.items():
        if matrix1_id == "_metadata":
            continue
        if data["status"] == "failed" and data.get("error_type") == "FAILED":
            current_memory = data["memory_tier"]
            next_memory = increment_memory_tier(current_memory)

            if next_memory:
                print(f"Auto-resetting {matrix1_id}: failed at {current_memory}GB -> pending at {next_memory}GB")
                data["status"] = "pending"
                data["memory_tier"] = next_memory
                data["job_id"] = None
                data["error_type"] = None
                reset_count += 1
            else:
                print(f"Skipping {matrix1_id}: already failed at max memory ({current_memory}GB)")

    if reset_count > 0:
        print(f"\nReset {reset_count} failed jobs to pending for OOM retry\n")
        save_manifest(manifest, n_hops)
        manifest = load_manifest(n_hops)  # Reload

    # Track submitted jobs
    submitted_jobs = {}  # job_id -> matrix1_id

    # Load existing running jobs from manifest
    for matrix1_id, data in manifest.items():
        if matrix1_id == "_metadata":
            continue
        if data["status"] == "running" and data["job_id"]:
            submitted_jobs[data["job_id"]] = matrix1_id

    start_time = datetime.now()

    # Main loop
    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} at {datetime.now().strftime('%H:%M:%S')} ---")

        # Reload manifest
        manifest = load_manifest(n_hops)

        # Get SLURM status
        running_jobs = get_running_slurm_jobs()
        pending_jobs = get_pending_slurm_jobs()

        # Check completed jobs
        if submitted_jobs:
            completed_jobs = get_completed_jobs_since(list(submitted_jobs.keys()), start_time)

            for job_id, (state, exit_code) in completed_jobs.items():
                if job_id not in submitted_jobs:
                    continue

                matrix1_id = submitted_jobs[job_id]
                matrix1_index = int(matrix1_id.split('_')[1])

                print(f"Job {job_id} ({matrix1_id}) completed: State={state}, ExitCode={exit_code}")

                if state == "COMPLETED" and exit_code == 0:
                    # Success!
                    update_job_status(manifest, matrix1_id, n_hops, status="completed")
                    print(f"  ✓ {matrix1_id} completed successfully")
                    del submitted_jobs[job_id]

                elif state == "OUT_OF_MEMORY" or exit_code == 137 or state == "FAILED":
                    # OOM failure - retry at higher memory
                    # Treat FAILED as OOM since GraphBLAS internal OOM doesn't trigger SLURM OOM
                    current_memory = manifest[matrix1_id]["memory_tier"]
                    next_memory = increment_memory_tier(current_memory)

                    if next_memory:
                        print(f"  ⚠ {matrix1_id} OOM at {current_memory}GB (state={state}), retrying at {next_memory}GB")
                        update_job_status(
                            manifest, matrix1_id, n_hops,
                            status="pending",
                            memory_tier=next_memory,
                            attempts=manifest[matrix1_id]["attempts"] + 1,
                            job_id=None
                        )
                    else:
                        print(f"  ✗ {matrix1_id} failed OOM at {current_memory}GB (max memory reached)")
                        update_job_status(
                            manifest, matrix1_id, n_hops,
                            status="failed",
                            error_type="OOM_MAX_MEMORY"
                        )

                    del submitted_jobs[job_id]

                else:
                    # Other failure (CANCELLED, TIMEOUT, NODE_FAIL)
                    print(f"  ✗ {matrix1_id} failed: {state}")
                    update_job_status(
                        manifest, matrix1_id, n_hops,
                        status="failed",
                        error_type=state
                    )
                    del submitted_jobs[job_id]

        # Reload manifest after updates
        manifest = load_manifest(n_hops)

        # Submit new jobs if queue space available
        num_pending_in_queue = len(pending_jobs)

        if num_pending_in_queue < max_concurrent:
            pending_jobs_to_submit = get_jobs_by_status(manifest, "pending")

            # Sort by attempts (retry failed jobs first) and memory tier (smaller first)
            pending_jobs_to_submit.sort(key=lambda x: (x[1]["attempts"], -x[1]["memory_tier"]))

            slots_available = max_concurrent - num_pending_in_queue
            to_submit = pending_jobs_to_submit[:min(slots_available * 10, len(pending_jobs_to_submit))]

            for matrix1_id, data in to_submit:
                matrix1_index = int(matrix1_id.split('_')[1])
                memory_tier = data["memory_tier"]

                print(f"Submitting {matrix1_id} with {memory_tier}GB memory (attempt {data['attempts'] + 1})...")
                job_id = submit_job(matrix1_index, memory_tier, nodes_file, edges_file, n_hops, matrices_dir)

                if job_id:
                    print(f"  → Job ID: {job_id}")
                    submitted_jobs[job_id] = matrix1_id
                    update_job_status(
                        manifest, matrix1_id, n_hops,
                        status="running",
                        job_id=job_id,
                        attempts=data["attempts"] + 1
                    )

        # Print status summary
        manifest = load_manifest(n_hops)
        print_status_summary(manifest, running_jobs)

        # Check if all jobs are complete
        pending_count = len(get_jobs_by_status(manifest, "pending"))
        running_count = len(get_jobs_by_status(manifest, "running"))

        if pending_count == 0 and running_count == 0:
            print("\n" + "=" * 80)
            print("ALL JOBS COMPLETE!")
            print("=" * 80)

            completed_count = len(get_jobs_by_status(manifest, "completed"))
            failed_count = len(get_jobs_by_status(manifest, "failed"))

            print(f"\nFinal status:")
            print(f"  Completed: {completed_count}/{total_jobs}")
            print(f"  Failed: {failed_count}/{total_jobs}")

            if failed_count > 0:
                print(f"\nFailed jobs:")
                for matrix1_id, data in get_jobs_by_status(manifest, "failed"):
                    print(f"  - {matrix1_id}: {data.get('error_type', 'UNKNOWN')}")

            print(f"\nNext step:")
            print(f"  Run: uv run python scripts/merge_results.py --n-hops {n_hops}")
            break

        # Sleep before next iteration
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Orchestrate N-hop metapath analysis across SLURM cluster"
    )
    parser.add_argument(
        "--n-hops",
        type=int,
        default=3,
        help="Number of hops to analyze (default: 3)"
    )
    args = parser.parse_args()

    try:
        orchestrate(n_hops=args.n_hops)
    except KeyboardInterrupt:
        print("\n\nOrchestrator interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
