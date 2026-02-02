#!/usr/bin/env python3
"""
Orchestrate distributed grouping jobs on SLURM.

Submits one job per type pair (src_type, tgt_type), determines relevant files for each,
monitors progress, handles retries with memory tier escalation.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Configuration
WORKER_SCRIPT = "scripts/group_single_onehop.sh"
POLL_INTERVAL = 30  # seconds
MAX_CONCURRENT_JOBS = 100  # Max pending jobs in queue


def load_manifest(n_hops):
    """Load grouping manifest."""
    manifest_path = f"results_{n_hops}hop/grouping_manifest.json"
    with open(manifest_path, 'r') as f:
        return json.load(f)


def save_manifest(manifest, n_hops):
    """Save grouping manifest."""
    manifest_path = f"results_{n_hops}hop/grouping_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_analysis_manifest(n_hops):
    """Load analysis manifest to map matrix indices to edge types."""
    manifest_path = f"results_{n_hops}hop/manifest.json"
    with open(manifest_path, 'r') as f:
        return json.load(f)


def determine_relevant_files(type1, type2, analysis_manifest, n_hops):
    """Determine which result files could contain paths between two types.

    Args:
        type1: First type (e.g. "Gene")
        type2: Second type (e.g. "Disease")
        analysis_manifest: Analysis manifest with matrix metadata
        n_hops: Number of hops

    Returns:
        List of result file paths that could contain paths between these types
    """
    # Find matrix indices where src_type/tgt_type match either type1 or type2
    # This catches paths in either direction
    relevant_indices = []

    for matrix_key, matrix_data in analysis_manifest.items():
        if matrix_key == "_metadata":
            continue

        # Extract matrix index from key like "matrix1_000"
        try:
            matrix_index = int(matrix_key.split('_')[1])
        except:
            continue

        # Get matrix metadata
        src_type = matrix_data.get("src_type")
        tgt_type = matrix_data.get("tgt_type")

        if not all([src_type, tgt_type]):
            continue

        # Check if this matrix could contribute to paths between type1 and type2
        # Match if either src or tgt matches either type1 or type2
        type_match = (
            type1 in src_type or src_type in type1 or
            type2 in src_type or src_type in type2 or
            type1 in tgt_type or tgt_type in type1 or
            type2 in tgt_type or tgt_type in type2
        )

        if type_match:
            relevant_indices.append(matrix_index)

    # Convert indices to file paths
    result_files = [f"results_{n_hops}hop/results_matrix1_{i:03d}.tsv"
                   for i in relevant_indices]

    return result_files


def get_running_slurm_jobs():
    """Get list of currently running job IDs."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    except Exception as e:
        print(f"Warning: Could not query SLURM queue: {e}")
    return []


def get_pending_slurm_jobs():
    """Get list of pending job IDs."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "PENDING", "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    except Exception as e:
        print(f"Warning: Could not query pending jobs: {e}")
    return []


def get_completed_jobs_since(job_ids, start_time):
    """Check which jobs have completed since start_time.

    Returns:
        dict: {job_id: (state, exit_code)}
    """
    if not job_ids:
        return {}

    try:
        # Use sacct to check job status
        result = subprocess.run(
            [
                "sacct",
                "-j", ",".join(job_ids),
                "-S", start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "-o", "JobID,State,ExitCode",
                "-n", "-P"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return {}

        completed = {}
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue

            parts = line.split('|')
            if len(parts) < 3:
                continue

            job_id = parts[0].split('.')[0]  # Remove .batch suffix
            state = parts[1]
            exit_code_str = parts[2]

            # Parse exit code (format: "0:0")
            try:
                exit_code = int(exit_code_str.split(':')[0])
            except:
                exit_code = -1

            # Only include terminal states
            if state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"]:
                completed[job_id] = (state, exit_code)

        return completed

    except Exception as e:
        print(f"Warning: sacct query failed: {e}")
        return {}


def submit_job(type1, type2, file_list, memory_gb, n_hops, job_index):
    """Submit a SLURM job for one type pair.

    Args:
        type1: First type (e.g. "Gene")
        type2: Second type (e.g. "Disease")
        file_list: List of result files to scan
        memory_gb: Memory to allocate in GB
        n_hops: Number of hops
        job_index: Job index for logging

    Returns:
        Job ID string or None if submission failed
    """
    log_dir = f"logs_grouping_{n_hops}hop"
    os.makedirs(log_dir, exist_ok=True)

    # Write file list to a temporary file
    file_list_path = f"{log_dir}/files_typepair_{job_index:04d}.txt"
    with open(file_list_path, 'w') as f:
        f.write('\n'.join(file_list))

    try:
        result = subprocess.run(
            [
                "sbatch",
                f"--mem={memory_gb}G",
                f"--output={log_dir}/typepair_{job_index:04d}_mem{memory_gb}.out",
                f"--error={log_dir}/typepair_{job_index:04d}_mem{memory_gb}.err",
                WORKER_SCRIPT,
                type1,
                type2,
                file_list_path,
                str(n_hops)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"ERROR: sbatch failed for typepair_{job_index:04d}: {result.stderr}")
            return None

        # Extract job ID from "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        return job_id

    except Exception as e:
        print(f"ERROR: Failed to submit typepair_{job_index:04d}: {e}")
        return None


def increment_memory_tier(current_memory):
    """Get next memory tier for retry."""
    if current_memory < 64:
        return 64
    elif current_memory < 128:
        return 128
    elif current_memory < 250:
        return 250
    elif current_memory < 500:
        return 500
    else:
        return None  # Already at max


def update_job_status(manifest, job_key, n_hops, **kwargs):
    """Update job status in manifest."""
    for key, value in kwargs.items():
        manifest[job_key][key] = value
    manifest[job_key]["last_update"] = datetime.now().isoformat()
    save_manifest(manifest, n_hops)


def get_jobs_by_status(manifest, status):
    """Get all jobs with given status."""
    jobs = []
    for key, data in manifest.items():
        if key == "_metadata":
            continue
        if data.get("status") == status:
            jobs.append((key, data))
    return jobs


def print_status_summary(manifest, running_jobs):
    """Print current status summary."""
    pending = len(get_jobs_by_status(manifest, "pending"))
    running = len(get_jobs_by_status(manifest, "running"))
    completed = len(get_jobs_by_status(manifest, "completed"))
    failed = len(get_jobs_by_status(manifest, "failed"))
    total = len(manifest) - 1  # Exclude _metadata

    print(f"\nStatus: {completed}/{total} completed, {running} running, {pending} pending, {failed} failed")
    print(f"SLURM queue: {len(running_jobs)} total jobs")


def orchestrate(n_hops):
    """Main orchestration loop."""
    print("=" * 80)
    print(f"ORCHESTRATING DISTRIBUTED GROUPING FOR {n_hops}-HOP")
    print("=" * 80)
    print(f"Worker script: {WORKER_SCRIPT}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Max concurrent: {MAX_CONCURRENT_JOBS}")
    print()

    # Load manifests
    manifest = load_manifest(n_hops)
    analysis_manifest = load_analysis_manifest(n_hops)

    total_jobs = len(manifest) - 1  # Exclude _metadata
    print(f"Loaded manifest with {total_jobs} grouping jobs")
    print(f"Loaded analysis manifest with {len(analysis_manifest)-1} matrix entries")

    # Pre-compute relevant files for each type pair (do this once upfront)
    print("\nDetermining relevant files for each type pair...")
    typepair_to_files = {}
    for job_key, data in manifest.items():
        if job_key == "_metadata":
            continue
        type1 = data["type1"]
        type2 = data["type2"]
        typepair_key = (type1, type2)
        if typepair_key not in typepair_to_files:
            file_list = determine_relevant_files(type1, type2, analysis_manifest, n_hops)
            typepair_to_files[typepair_key] = file_list

    avg_files = sum(len(files) for files in typepair_to_files.values()) / len(typepair_to_files)
    print(f"Average files per type pair: {avg_files:.1f} (vs {len(analysis_manifest)-1} total)")

    # Track submitted jobs
    submitted_jobs = {}  # job_id -> job_key

    # Load existing running jobs from manifest
    for job_key, data in manifest.items():
        if job_key == "_metadata":
            continue
        if data["status"] == "running" and data["job_id"]:
            submitted_jobs[data["job_id"]] = job_key

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

                job_key = submitted_jobs[job_id]
                current_memory = manifest[job_key]["memory_tier"]

                print(f"Job {job_id} ({job_key}) completed: State={state}, ExitCode={exit_code}")

                if state == "COMPLETED" and exit_code == 0:
                    # Success
                    update_job_status(manifest, job_key, n_hops, status="completed")
                    print(f"  ✓ {job_key} completed successfully")
                    del submitted_jobs[job_id]

                elif state == "OUT_OF_MEMORY" or exit_code == 137:
                    # OOM - retry at higher tier
                    next_memory = increment_memory_tier(current_memory)

                    if next_memory:
                        print(f"  ⚠ {job_key} OOM at {current_memory}GB, retrying at {next_memory}GB")
                        update_job_status(
                            manifest, job_key, n_hops,
                            status="pending",
                            memory_tier=next_memory,
                            attempts=manifest[job_key]["attempts"] + 1,
                            job_id=None
                        )
                    else:
                        print(f"  ✗ {job_key} failed OOM at {current_memory}GB (max memory reached)")
                        update_job_status(
                            manifest, job_key, n_hops,
                            status="failed",
                            error_type="OOM_MAX_MEMORY"
                        )

                    del submitted_jobs[job_id]

                else:
                    # Other failure
                    print(f"  ✗ {job_key} failed: {state}")
                    update_job_status(
                        manifest, job_key, n_hops,
                        status="failed",
                        error_type=state
                    )
                    del submitted_jobs[job_id]

        # Reload manifest after updates
        manifest = load_manifest(n_hops)

        # Submit new jobs if queue space available
        num_pending_in_queue = len(pending_jobs)

        if num_pending_in_queue < MAX_CONCURRENT_JOBS:
            pending_jobs_to_submit = get_jobs_by_status(manifest, "pending")

            # Sort by attempts (retry failed jobs first)
            pending_jobs_to_submit.sort(key=lambda x: x[1]["attempts"], reverse=True)

            slots_available = MAX_CONCURRENT_JOBS - num_pending_in_queue
            to_submit = pending_jobs_to_submit[:slots_available]

            for job_key, data in to_submit:
                type1 = data["type1"]
                type2 = data["type2"]
                job_index = data["index"]
                memory_tier = data["memory_tier"]

                # Get pre-computed file list
                typepair_key = (type1, type2)
                file_list = typepair_to_files.get(typepair_key, [])

                print(f"Submitting {job_key} ({type1}, {type2}) with {memory_tier}GB ({len(file_list)} files, attempt {data['attempts'] + 1})...")
                job_id = submit_job(type1, type2, file_list, memory_tier, n_hops, job_index)

                if job_id:
                    print(f"  → Job ID: {job_id}")
                    submitted_jobs[job_id] = job_key
                    update_job_status(
                        manifest, job_key, n_hops,
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
            print("ALL GROUPING JOBS COMPLETE!")
            print("=" * 80)

            completed_count = len(get_jobs_by_status(manifest, "completed"))
            failed_count = len(get_jobs_by_status(manifest, "failed"))

            print(f"\nFinal status:")
            print(f"  Jobs: {completed_count}/{total_jobs} completed, {failed_count}/{total_jobs} failed")

            if failed_count > 0:
                print(f"\nFailed jobs:")
                for job_key, data in get_jobs_by_status(manifest, "failed"):
                    print(f"  - {job_key}: {data.get('error_type', 'UNKNOWN')}")

            output_dir = f"grouped_by_results_{n_hops}hop"
            print(f"\nGrouped results in: {output_dir}/")
            break

        # Sleep before next iteration
        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(
        description='Orchestrate distributed grouping on SLURM'
    )
    parser.add_argument(
        '--n-hops',
        type=int,
        required=True,
        help='Number of hops (1, 2, or 3)'
    )

    args = parser.parse_args()

    try:
        orchestrate(n_hops=args.n_hops)
    except KeyboardInterrupt:
        print("\n\nOrchestrator interrupted by user. Exiting...")
        print("You can restart later - it will resume from the manifest.")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
