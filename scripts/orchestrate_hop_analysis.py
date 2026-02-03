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

import os
import sys
import time
from datetime import datetime
from collections import defaultdict

# Import path tracking functions and shared utilities
sys.path.insert(0, os.path.dirname(__file__))
from path_tracker import get_path_statistics as get_path_stats_from_tracker
from slurm_utils import (
    load_manifest,
    save_manifest,
    update_job_status,
    get_jobs_by_status,
    get_running_slurm_jobs,
    get_pending_slurm_jobs,
    get_completed_jobs_since,
    increment_memory_tier,
    submit_slurm_job,
    HOP_ANALYSIS_MEMORY_TIERS,
)


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


def submit_job(matrix1_index, memory_gb, matrices_dir, n_hops, src_type, pred, direction, tgt_type):
    """Submit a single matrix1 job to SLURM with specified memory.

    Args:
        matrix1_index: Index of the starting matrix
        memory_gb: Memory to request in GB
        matrices_dir: Directory with pre-built matrices
        n_hops: Number of hops to analyze
        src_type: Source node type for matrix1
        pred: Predicate for matrix1
        direction: Direction for matrix1 (F, R, or A)
        tgt_type: Target node type for matrix1

    Returns: job_id (str) or None on failure
    """
    # Use largemem partition for jobs >1000GB (i.e., 1500GB tier)
    partition = 'largemem' if memory_gb > 1000 else 'lowpri'

    logs_dir = f"logs_{n_hops}hop"
    job_name = f"{n_hops}hop_m1_{matrix1_index:03d}"

    cmd = [
        'sbatch',
        f'--partition={partition}',
        f'--mem={memory_gb}G',
        f'--job-name={job_name}',
        f'--output={logs_dir}/matrix1_{matrix1_index:03d}_mem{memory_gb}.out',
        f'--error={logs_dir}/matrix1_{matrix1_index:03d}_mem{memory_gb}.err',
        WORKER_SCRIPT,
        str(matrix1_index),
        matrices_dir,
        str(n_hops),
        src_type,
        pred,
        direction,
        tgt_type
    ]

    return submit_slurm_job(cmd, job_name=job_name)


def get_path_statistics(matrix1_index, n_hops, current_memory_gb):
    """Get path completion statistics for a Matrix1 job.

    Returns dict with:
        - completed_count: Number of completed paths
        - failed_at_current_tier: Number of paths that failed at current memory tier
        - failed_at_lower_tiers: Number of paths that failed at lower memory tiers
        - total_failed: Total failed paths across all tiers
    """
    results_dir = f"results_{n_hops}hop"

    # Get stats from path_tracker
    stats = get_path_stats_from_tracker(results_dir, matrix1_index)

    completed_count = stats['completed']
    failed_by_tier = stats['failed_by_tier']
    total_failed = stats['total_failed']

    # Count failed paths by tier category
    failed_at_current_tier = failed_by_tier.get(current_memory_gb, 0)
    failed_at_lower_tiers = sum(
        count for tier, count in failed_by_tier.items()
        if tier < current_memory_gb
    )

    return {
        'completed_count': completed_count,
        'failed_at_current_tier': failed_at_current_tier,
        'failed_at_lower_tiers': failed_at_lower_tiers,
        'total_failed': total_failed
    }


def print_status_summary(manifest, running_jobs):
    """Print current status summary."""
    status_counts = defaultdict(int)
    total_paths_completed = 0
    total_paths_failed = 0

    for key, data in manifest.items():
        if key != "_metadata":
            status_counts[data["status"]] += 1
            # Accumulate path-level stats if available
            total_paths_completed += data.get("paths_completed", 0)
            total_paths_failed += data.get("paths_failed", 0)

    total = len(manifest) - 1  # Exclude _metadata
    pending = status_counts["pending"]
    running = status_counts["running"]
    completed = status_counts["completed"]
    failed = status_counts["failed"]

    pct_complete = (completed / total * 100) if total > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"STATUS: {completed}/{total} jobs completed ({pct_complete:.1f}%) | "
          f"Running: {running} | Pending: {pending} | Failed: {failed}")
    print(f"SLURM queue: {len(running_jobs)} running jobs")

    # Show path-level progress if available
    if total_paths_completed > 0 or total_paths_failed > 0:
        total_paths = total_paths_completed + total_paths_failed
        pct_paths = (total_paths_completed / total_paths * 100) if total_paths > 0 else 0
        print(f"PATHS: {total_paths_completed:,} completed, {total_paths_failed:,} failed ({pct_paths:.1f}% success)")

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
    manifest = load_manifest(manifest_path)

    # Extract input file paths from metadata
    if "_metadata" not in manifest:
        print("ERROR: Manifest missing _metadata with input file paths!")
        print("Please re-run: uv run python scripts/prepare_analysis.py")
        sys.exit(1)

    matrices_dir = manifest["_metadata"].get("matrices_dir", None)

    if not matrices_dir:
        print("ERROR: Manifest missing matrices_dir in _metadata!")
        print("Please re-run: uv run python scripts/prepare_analysis.py --matrices-dir <path>")
        sys.exit(1)

    # Get n_hops from manifest if not explicitly provided
    if "_metadata" in manifest and "n_hops" in manifest["_metadata"]:
        manifest_n_hops = manifest["_metadata"]["n_hops"]
        if n_hops != manifest_n_hops:
            print(f"WARNING: CLI n_hops ({n_hops}) differs from manifest ({manifest_n_hops})")
            print(f"Using manifest value: {manifest_n_hops}")
            n_hops = manifest_n_hops

    print(f"\nInput data from manifest:")
    print(f"  Pre-built matrices: {matrices_dir}")
    print(f"  N-hops: {n_hops}")

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
            next_memory = increment_memory_tier(current_memory, HOP_ANALYSIS_MEMORY_TIERS)

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
        save_manifest(manifest, manifest_path)
        manifest = load_manifest(manifest_path)  # Reload

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
        manifest = load_manifest(manifest_path)

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
                current_memory = manifest[matrix1_id]["memory_tier"]

                print(f"Job {job_id} ({matrix1_id}) completed: State={state}, ExitCode={exit_code}")

                # Get path statistics for this job
                path_stats = get_path_statistics(matrix1_index, n_hops, current_memory)
                completed_paths = path_stats['completed_count']
                failed_current = path_stats['failed_at_current_tier']
                failed_lower = path_stats['failed_at_lower_tiers']
                total_failed = path_stats['total_failed']

                print(f"  Path stats: {completed_paths} completed, {failed_current} failed at {current_memory}GB, {failed_lower} failed at lower tiers")

                if state == "COMPLETED" and exit_code == 0:
                    # Job completed successfully - check if all paths are done
                    if total_failed == 0:
                        # All paths completed successfully
                        update_job_status(
                            manifest, matrix1_id, manifest_path,
                            status="completed",
                            paths_completed=completed_paths,
                            paths_failed=0
                        )
                        print(f"  ✓ {matrix1_id} completed successfully ({completed_paths} paths)")
                    else:
                        # Job completed but some paths failed - need retry at higher tier
                        next_memory = increment_memory_tier(current_memory, HOP_ANALYSIS_MEMORY_TIERS)
                        if next_memory:
                            print(f"  ⚠ {matrix1_id} has {total_failed} failed paths, retrying at {next_memory}GB")
                            update_job_status(
                                manifest, matrix1_id, manifest_path,
                                status="pending",
                                memory_tier=next_memory,
                                attempts=manifest[matrix1_id]["attempts"] + 1,
                                job_id=None,
                                paths_completed=completed_paths,
                                paths_failed=total_failed
                            )
                        else:
                            # Already at max tier, mark as completed with partial results
                            print(f"  ✓ {matrix1_id} completed with {total_failed} failed paths (max memory reached)")
                            update_job_status(
                                manifest, matrix1_id, manifest_path,
                                status="completed",
                                paths_completed=completed_paths,
                                paths_failed=total_failed,
                                error_type="PARTIAL_MAX_MEMORY"
                            )

                    del submitted_jobs[job_id]

                elif state == "OUT_OF_MEMORY" or exit_code == 137 or state == "FAILED":
                    # OOM failure - check path stats to decide retry strategy
                    # Treat FAILED as OOM since GraphBLAS internal OOM doesn't trigger SLURM OOM

                    # Decision logic:
                    # 1. If we have failed paths at current tier OR lower tiers -> retry at higher tier
                    # 2. If we have completed some paths -> we made progress, retry at higher tier for failed paths

                    if failed_current > 0 or failed_lower > 0 or completed_paths > 0:
                        # We have failed paths or made some progress
                        next_memory = increment_memory_tier(current_memory, HOP_ANALYSIS_MEMORY_TIERS)

                        if next_memory:
                            print(f"  ⚠ {matrix1_id} OOM at {current_memory}GB (state={state})")
                            print(f"      Progress: {completed_paths} completed, {total_failed} failed")
                            print(f"      Retrying at {next_memory}GB for failed paths")
                            update_job_status(
                                manifest, matrix1_id, manifest_path,
                                status="pending",
                                memory_tier=next_memory,
                                attempts=manifest[matrix1_id]["attempts"] + 1,
                                job_id=None,
                                paths_completed=completed_paths,
                                paths_failed=total_failed
                            )
                        else:
                            print(f"  ✗ {matrix1_id} failed OOM at {current_memory}GB (max memory reached)")
                            print(f"      Final: {completed_paths} completed, {total_failed} failed")
                            update_job_status(
                                manifest, matrix1_id, manifest_path,
                                status="completed",
                                paths_completed=completed_paths,
                                paths_failed=total_failed,
                                error_type="PARTIAL_MAX_MEMORY"
                            )
                    else:
                        # No progress at all - this is a true failure
                        print(f"  ✗ {matrix1_id} failed immediately at {current_memory}GB (no paths computed)")
                        update_job_status(
                            manifest, matrix1_id, manifest_path,
                            status="failed",
                            error_type=f"FAILED_IMMEDIATE_{state}",
                            paths_completed=0,
                            paths_failed=0
                        )

                    del submitted_jobs[job_id]

                else:
                    # Other failure (CANCELLED, TIMEOUT, NODE_FAIL)
                    print(f"  ✗ {matrix1_id} failed: {state}")
                    print(f"      Progress: {completed_paths} completed, {total_failed} failed")
                    update_job_status(
                        manifest, matrix1_id, manifest_path,
                        status="failed",
                        error_type=state,
                        paths_completed=completed_paths,
                        paths_failed=total_failed
                    )
                    del submitted_jobs[job_id]

        # Reload manifest after updates
        manifest = load_manifest(manifest_path)

        # Submit new jobs if queue space available
        num_pending_in_queue = len(pending_jobs)

        if num_pending_in_queue < max_concurrent:
            pending_jobs_to_submit = get_jobs_by_status(manifest, "pending")

            # Sort by attempts (retry failed jobs first) and memory tier (smaller first)
            pending_jobs_to_submit.sort(key=lambda x: (x[1]["attempts"], -x[1]["memory_tier"]))

            slots_available = max_concurrent - num_pending_in_queue
            to_submit = pending_jobs_to_submit[:slots_available]

            for matrix1_id, data in to_submit:
                matrix1_index = int(matrix1_id.split('_')[1])
                memory_tier = data["memory_tier"]

                print(f"Submitting {matrix1_id} with {memory_tier}GB memory (attempt {data['attempts'] + 1})...")
                job_id = submit_job(
                    matrix1_index, memory_tier, matrices_dir, n_hops,
                    data['src_type'], data['pred'], data['direction'], data['tgt_type']
                )

                if job_id:
                    print(f"  → Job ID: {job_id}")
                    submitted_jobs[job_id] = matrix1_id
                    update_job_status(
                        manifest, matrix1_id, manifest_path,
                        status="running",
                        job_id=job_id,
                        attempts=data["attempts"] + 1
                    )

        # Print status summary
        manifest = load_manifest(manifest_path)
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

            # Calculate path-level statistics
            total_paths_completed = 0
            total_paths_failed = 0
            jobs_with_partial_results = 0

            for matrix1_id, data in get_jobs_by_status(manifest, "completed"):
                paths_completed = data.get("paths_completed", 0)
                paths_failed = data.get("paths_failed", 0)
                total_paths_completed += paths_completed
                total_paths_failed += paths_failed

                if paths_failed > 0:
                    jobs_with_partial_results += 1

            print(f"\nFinal status:")
            print(f"  Jobs: {completed_count}/{total_jobs} completed, {failed_count}/{total_jobs} failed")

            if total_paths_completed > 0 or total_paths_failed > 0:
                total_paths = total_paths_completed + total_paths_failed
                pct_success = (total_paths_completed / total_paths * 100) if total_paths > 0 else 0
                print(f"  Paths: {total_paths_completed:,} completed, {total_paths_failed:,} failed ({pct_success:.1f}% success)")

                if jobs_with_partial_results > 0:
                    print(f"  Note: {jobs_with_partial_results} jobs completed with some failed paths (OOM at max tier)")

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
