#!/usr/bin/env python3
"""
Shared SLURM orchestration utilities.

Common functions for SLURM job management used by both hop analysis
and grouping orchestrators.
"""

import getpass
import json
import os
import subprocess
from datetime import datetime


# Default memory tiers for different job types
HOP_ANALYSIS_MEMORY_TIERS = [180, 250, 500, 1000, 1400]
GROUPING_MEMORY_TIERS = [64, 128, 250, 500, 1000]

# CPU tiers for thread creation failures
GROUPING_CPU_TIERS = [1, 4, 8, 16]


def load_manifest(manifest_path):
    """Load manifest from disk.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Manifest dict
    """
    with open(manifest_path, 'r') as f:
        return json.load(f)


def save_manifest(manifest, manifest_path):
    """Save manifest to disk.

    Args:
        manifest: Manifest dict
        manifest_path: Path to manifest JSON file
    """
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def update_job_status(manifest, job_key, manifest_path, **kwargs):
    """Update a job's status in the manifest.

    Args:
        manifest: Manifest dict (modified in place)
        job_key: Key identifying the job
        manifest_path: Path to manifest file for saving
        **kwargs: Status fields to update
    """
    if job_key not in manifest:
        raise KeyError(f"Job {job_key} not found in manifest")

    for key, value in kwargs.items():
        manifest[job_key][key] = value

    manifest[job_key]["last_update"] = datetime.now().isoformat()
    save_manifest(manifest, manifest_path)


def get_jobs_by_status(manifest, status):
    """Get all jobs with given status.

    Args:
        manifest: Manifest dict
        status: Status string to filter by

    Returns:
        List of (job_key, job_data) tuples
    """
    return [
        (job_key, data)
        for job_key, data in manifest.items()
        if job_key != "_metadata" and data.get("status") == status
    ]


def get_running_slurm_jobs(include_names=True):
    """Get RUNNING jobs from SLURM.

    Args:
        include_names: If True, return dict {job_id: job_name}.
                      If False, return list of job_ids.

    Returns:
        Dict or list of running jobs
    """
    try:
        username = getpass.getuser()

        if include_names:
            result = subprocess.run(
                ["squeue", "-u", username, "-t", "RUNNING", "-o", "%i %j", "-h"],
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
            return {}
        else:
            result = subprocess.run(
                ["squeue", "-u", username, "-t", "RUNNING", "-o", "%i", "-h"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return []

    except Exception as e:
        print(f"Warning: Failed to query SLURM running jobs: {e}")
        return {} if include_names else []


def get_pending_slurm_jobs():
    """Get list of PENDING jobs from SLURM.

    Returns:
        List of pending job IDs
    """
    try:
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


def get_completed_jobs_since(job_ids, since_time):
    """Query SLURM sacct for completed jobs.

    Args:
        job_ids: List of job IDs to check
        since_time: datetime object for start time filter

    Returns:
        Dict {job_id: (state, exit_code)}
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
        terminal_states = {"COMPLETED", "FAILED", "OUT_OF_MEMORY", "CANCELLED", "TIMEOUT", "NODE_FAIL"}

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

            # Only return jobs in terminal states
            if state in terminal_states:
                completed[job_id] = (state, exit_code_int)

        return completed

    except Exception as e:
        print(f"Warning: Failed to query sacct: {e}")
        return {}


def increment_memory_tier(current_memory, memory_tiers=None):
    """Increment memory tier for retry.

    Args:
        current_memory: Current memory in GB
        memory_tiers: List of memory tiers in ascending order.
                     Defaults to HOP_ANALYSIS_MEMORY_TIERS.

    Returns:
        Next memory tier or None if at max
    """
    if memory_tiers is None:
        memory_tiers = HOP_ANALYSIS_MEMORY_TIERS

    for tier in memory_tiers:
        if current_memory < tier:
            return tier

    return None  # Already at max


def increment_cpu_tier(current_cpus, cpu_tiers=None):
    """Increment CPU tier for retry.

    Args:
        current_cpus: Current number of CPUs
        cpu_tiers: List of CPU tiers in ascending order.
                  Defaults to GROUPING_CPU_TIERS.

    Returns:
        Next CPU tier or None if at max
    """
    if cpu_tiers is None:
        cpu_tiers = GROUPING_CPU_TIERS

    for tier in cpu_tiers:
        if current_cpus < tier:
            return tier

    return None  # Already at max


def check_error_log_for_failure_type(log_file_path):
    """Check error log file to determine failure type.

    Args:
        log_file_path: Path to SLURM .err file

    Returns:
        String indicating error type: 'MEMORY', 'THREAD', 'TIMEOUT', or None
    """
    if not os.path.exists(log_file_path):
        return None

    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

        # Check for various memory-related errors
        if any(indicator in content for indicator in [
            'oom_kill',
            'OOM',
            'ArrayMemoryError',
            'Unable to allocate',
            'MemoryError',
            'out of memory'
        ]):
            return 'MEMORY'

        # Check for thread creation failures
        if 'Thread creation failed' in content or 'Resource temporarily unavailable' in content:
            return 'THREAD'

        # Check for timeout
        if 'TIME LIMIT' in content or 'DUE TO TIME LIMIT' in content:
            return 'TIMEOUT'

        return None

    except Exception as e:
        print(f"Warning: Failed to read log file {log_file_path}: {e}")
        return None


def submit_slurm_job(cmd, job_name=None):
    """Submit a job to SLURM via sbatch.

    Args:
        cmd: List of command arguments for sbatch
        job_name: Optional job name for error messages

    Returns:
        Job ID string or None on failure
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            name = job_name or "job"
            print(f"ERROR: sbatch failed for {name}: {result.stderr}")
            return None

        # Parse job ID from "Submitted batch job 12345678"
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            return output.split()[-1]
        else:
            print(f"ERROR: Unexpected sbatch output: {output}")
            return None

    except Exception as e:
        name = job_name or "job"
        print(f"ERROR: Failed to submit {name}: {e}")
        return None
