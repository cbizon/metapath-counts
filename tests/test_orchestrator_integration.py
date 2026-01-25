"""
Integration test for orchestrator with mock SLURM and path tracking.

This test simulates the full orchestrator workflow:
1. Mock SLURM commands (sbatch, squeue, sacct)
2. Create realistic manifest and path tracking files
3. Simulate job lifecycle (pending -> running -> completed/failed)
4. Verify orchestrator makes correct retry decisions based on path statistics
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from orchestrate_hop_analysis import (
    get_path_statistics,
    increment_memory_tier,
    load_manifest,
    save_manifest,
    update_job_status
)
from path_tracker import (
    record_completed_path,
    record_failed_path,
    get_tracking_dir
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with results directory."""
    workspace = tempfile.mkdtemp(prefix="test_orchestrator_")

    # Create results directory structure
    results_dir = os.path.join(workspace, "results_3hop")
    os.makedirs(results_dir, exist_ok=True)

    # Create logs directory
    logs_dir = os.path.join(workspace, "logs_3hop")
    os.makedirs(logs_dir, exist_ok=True)

    # Change to workspace directory
    original_cwd = os.getcwd()
    os.chdir(workspace)

    yield workspace, results_dir, logs_dir

    # Cleanup
    os.chdir(original_cwd)
    shutil.rmtree(workspace)


def create_test_manifest(results_dir, num_jobs=3):
    """Create a test manifest with a few jobs."""
    manifest_path = os.path.join(results_dir, "manifest.json")

    manifest = {
        "_metadata": {
            "nodes_file": "/fake/nodes.jsonl",
            "edges_file": "/fake/edges.jsonl",
            "n_hops": 3,
            "created_at": datetime.now().isoformat(),
            "total_jobs": num_jobs
        }
    }

    for i in range(num_jobs):
        manifest[f"matrix1_{i:03d}"] = {
            "status": "pending",
            "memory_tier": 180,
            "attempts": 0,
            "job_id": None,
            "last_update": datetime.now().isoformat(),
            "error_type": None,
            "matrix_nvals": 1000 * (i + 1),
            "src_type": f"Type{i}",
            "pred": "predicate",
            "tgt_type": f"Type{i+1}",
            "direction": "F",
            "paths_completed": 0,
            "paths_failed": 0
        }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def simulate_job_completion_all_paths_successful(results_dir, matrix1_index, num_paths=10):
    """Simulate a job that completed all paths successfully."""
    # Record completed paths
    for i in range(num_paths):
        path_id = f"Type{matrix1_index}|pred|F|Type{matrix1_index+1}__Type{matrix1_index+1}|pred|F|Type{matrix1_index+2}__Type{matrix1_index+2}|pred|F|Type{matrix1_index+3}_path{i}"
        record_completed_path(path_id, results_dir, matrix1_index)


def simulate_job_completion_with_failures(results_dir, matrix1_index,
                                          num_completed=7, num_failed=3,
                                          failure_tier=180):
    """Simulate a job that completed some paths and failed others."""
    # Record completed paths
    for i in range(num_completed):
        path_id = f"Type{matrix1_index}|pred|F|Type{matrix1_index+1}__completed_path{i}"
        record_completed_path(path_id, results_dir, matrix1_index)

    # Record failed paths
    for i in range(num_failed):
        path_id = f"Type{matrix1_index}|pred|F|Type{matrix1_index+1}__failed_path{i}"
        record_failed_path(path_id, results_dir, matrix1_index,
                          memory_gb=failure_tier, reason="oom")


def simulate_job_oom_no_progress(results_dir, matrix1_index):
    """Simulate a job that OOMed immediately with no progress."""
    # Don't create any tracking files - simulates immediate failure
    pass


def test_path_statistics_integration(temp_workspace):
    """Test get_path_statistics with realistic tracking files."""
    workspace, results_dir, _ = temp_workspace

    # Simulate job 0: All paths completed
    simulate_job_completion_all_paths_successful(results_dir, 0, num_paths=10)

    stats = get_path_statistics(0, 3, 180)
    assert stats['completed_count'] == 10
    assert stats['failed_at_current_tier'] == 0
    assert stats['total_failed'] == 0

    # Simulate job 1: Some paths completed, some failed at 180GB
    simulate_job_completion_with_failures(results_dir, 1,
                                         num_completed=7, num_failed=3,
                                         failure_tier=180)

    stats = get_path_statistics(1, 3, 180)
    assert stats['completed_count'] == 7
    assert stats['failed_at_current_tier'] == 3
    assert stats['total_failed'] == 3

    # Same job at 250GB tier should show failures as "lower tier"
    stats = get_path_statistics(1, 3, 250)
    assert stats['completed_count'] == 7
    assert stats['failed_at_current_tier'] == 0
    assert stats['failed_at_lower_tiers'] == 3
    assert stats['total_failed'] == 3


def test_manifest_update_workflow(temp_workspace):
    """Test manifest update workflow with path statistics."""
    workspace, results_dir, _ = temp_workspace

    # Create manifest
    create_test_manifest(results_dir, num_jobs=2)

    # Load manifest
    manifest = load_manifest(3)
    assert manifest["matrix1_000"]["status"] == "pending"

    # Simulate job completion
    simulate_job_completion_all_paths_successful(results_dir, 0, num_paths=15)

    # Get path stats
    stats = get_path_statistics(0, 3, 180)

    # Update job status based on stats
    if stats['total_failed'] == 0:
        update_job_status(
            manifest, "matrix1_000", 3,
            status="completed",
            paths_completed=stats['completed_count'],
            paths_failed=0
        )

    # Reload and verify
    manifest = load_manifest(3)
    assert manifest["matrix1_000"]["status"] == "completed"
    assert manifest["matrix1_000"]["paths_completed"] == 15
    assert manifest["matrix1_000"]["paths_failed"] == 0


def test_retry_decision_all_paths_successful(temp_workspace):
    """Test retry decision when all paths complete successfully."""
    workspace, results_dir, _ = temp_workspace

    create_test_manifest(results_dir, num_jobs=1)
    simulate_job_completion_all_paths_successful(results_dir, 0, num_paths=20)

    manifest = load_manifest(3)
    stats = get_path_statistics(0, 3, 180)

    # Decision logic: All paths completed, no failures -> mark as completed
    assert stats['total_failed'] == 0

    # This should be marked as completed
    update_job_status(
        manifest, "matrix1_000", 3,
        status="completed",
        paths_completed=stats['completed_count'],
        paths_failed=0
    )

    manifest = load_manifest(3)
    assert manifest["matrix1_000"]["status"] == "completed"
    assert manifest["matrix1_000"]["paths_completed"] == 20
    assert manifest["matrix1_000"]["paths_failed"] == 0


def test_retry_decision_some_failures_at_current_tier(temp_workspace):
    """Test retry decision when job has failures at current tier."""
    workspace, results_dir, _ = temp_workspace

    create_test_manifest(results_dir, num_jobs=1)

    # Simulate: 15 completed, 5 failed at 180GB
    simulate_job_completion_with_failures(
        results_dir, 0,
        num_completed=15, num_failed=5,
        failure_tier=180
    )

    manifest = load_manifest(3)
    stats = get_path_statistics(0, 3, 180)

    # Decision logic: Has failures -> retry at next tier
    assert stats['total_failed'] == 5
    assert stats['failed_at_current_tier'] == 5

    current_memory = 180
    next_memory = increment_memory_tier(current_memory)

    assert next_memory == 250

    # Should retry at 250GB
    update_job_status(
        manifest, "matrix1_000", 3,
        status="pending",
        memory_tier=next_memory,
        attempts=1,
        job_id=None,
        paths_completed=stats['completed_count'],
        paths_failed=stats['total_failed']
    )

    manifest = load_manifest(3)
    assert manifest["matrix1_000"]["status"] == "pending"
    assert manifest["matrix1_000"]["memory_tier"] == 250
    assert manifest["matrix1_000"]["attempts"] == 1
    assert manifest["matrix1_000"]["paths_completed"] == 15
    assert manifest["matrix1_000"]["paths_failed"] == 5


def test_retry_decision_failures_at_lower_tier(temp_workspace):
    """Test retry decision when job has failures from lower tier."""
    workspace, results_dir, _ = temp_workspace

    create_test_manifest(results_dir, num_jobs=1)

    # First run at 180GB: 10 completed, 5 failed
    simulate_job_completion_with_failures(
        results_dir, 0,
        num_completed=10, num_failed=5,
        failure_tier=180
    )

    manifest = load_manifest(3)

    # Simulate retry at 250GB
    manifest["matrix1_000"]["memory_tier"] = 250
    manifest["matrix1_000"]["status"] = "running"
    save_manifest(manifest, 3)

    # At 250GB: complete 3 more paths (from the 5 that failed at 180GB)
    # 2 still fail at 250GB
    for i in range(3):
        path_id = f"Type0|pred|F|Type1__retry_success_path{i}"
        record_completed_path(path_id, results_dir, 0)

    for i in range(2):
        path_id = f"Type0|pred|F|Type1__retry_failed_path{i}"
        record_failed_path(path_id, results_dir, 0, memory_gb=250, reason="oom")

    # Get stats at 250GB tier
    stats = get_path_statistics(0, 3, 250)

    assert stats['completed_count'] == 13  # 10 from 180GB + 3 from 250GB
    assert stats['failed_at_current_tier'] == 2  # Failed at 250GB
    assert stats['failed_at_lower_tiers'] == 5  # Original failures at 180GB
    assert stats['total_failed'] == 7  # 5 from 180GB + 2 from 250GB

    # Should retry again at 500GB
    next_memory = increment_memory_tier(250)
    assert next_memory == 500


def test_retry_decision_at_max_memory(temp_workspace):
    """Test retry decision when already at max memory tier."""
    workspace, results_dir, _ = temp_workspace

    create_test_manifest(results_dir, num_jobs=1)

    # Simulate job at 1400GB (max tier) with some failures
    manifest = load_manifest(3)
    manifest["matrix1_000"]["memory_tier"] = 1400
    save_manifest(manifest, 3)

    simulate_job_completion_with_failures(
        results_dir, 0,
        num_completed=18, num_failed=2,
        failure_tier=1400
    )

    stats = get_path_statistics(0, 3, 1400)

    assert stats['total_failed'] == 2

    # At max tier, can't retry
    next_memory = increment_memory_tier(1400)
    assert next_memory is None

    # Should mark as completed with partial results
    update_job_status(
        manifest, "matrix1_000", 3,
        status="completed",
        paths_completed=stats['completed_count'],
        paths_failed=stats['total_failed'],
        error_type="PARTIAL_MAX_MEMORY"
    )

    manifest = load_manifest(3)
    assert manifest["matrix1_000"]["status"] == "completed"
    assert manifest["matrix1_000"]["paths_completed"] == 18
    assert manifest["matrix1_000"]["paths_failed"] == 2
    assert manifest["matrix1_000"]["error_type"] == "PARTIAL_MAX_MEMORY"


def test_retry_decision_no_progress(temp_workspace):
    """Test retry decision when job fails with no progress."""
    workspace, results_dir, _ = temp_workspace

    create_test_manifest(results_dir, num_jobs=1)
    simulate_job_oom_no_progress(results_dir, 0)

    stats = get_path_statistics(0, 3, 180)

    # No paths completed or failed - true failure
    assert stats['completed_count'] == 0
    assert stats['total_failed'] == 0

    # This indicates immediate failure - should NOT retry
    manifest = load_manifest(3)
    update_job_status(
        manifest, "matrix1_000", 3,
        status="failed",
        error_type="FAILED_IMMEDIATE_OUT_OF_MEMORY",
        paths_completed=0,
        paths_failed=0
    )

    manifest = load_manifest(3)
    assert manifest["matrix1_000"]["status"] == "failed"
    assert manifest["matrix1_000"]["error_type"] == "FAILED_IMMEDIATE_OUT_OF_MEMORY"


def test_multi_job_workflow(temp_workspace):
    """Test orchestrator logic with multiple jobs at different stages."""
    workspace, results_dir, _ = temp_workspace

    # Create manifest with 3 jobs
    create_test_manifest(results_dir, num_jobs=3)

    # Job 0: All paths successful
    simulate_job_completion_all_paths_successful(results_dir, 0, num_paths=10)

    # Job 1: Some failures at 180GB
    simulate_job_completion_with_failures(
        results_dir, 1,
        num_completed=8, num_failed=2,
        failure_tier=180
    )

    # Job 2: No progress (immediate failure)
    simulate_job_oom_no_progress(results_dir, 2)

    # Process each job
    manifest = load_manifest(3)

    # Job 0: Should complete successfully
    stats_0 = get_path_statistics(0, 3, 180)
    assert stats_0['total_failed'] == 0
    update_job_status(
        manifest, "matrix1_000", 3,
        status="completed",
        paths_completed=stats_0['completed_count'],
        paths_failed=0
    )

    # Job 1: Should retry at 250GB
    stats_1 = get_path_statistics(1, 3, 180)
    assert stats_1['total_failed'] == 2
    update_job_status(
        manifest, "matrix1_001", 3,
        status="pending",
        memory_tier=250,
        attempts=1,
        paths_completed=stats_1['completed_count'],
        paths_failed=stats_1['total_failed']
    )

    # Job 2: Should fail
    stats_2 = get_path_statistics(2, 3, 180)
    assert stats_2['completed_count'] == 0 and stats_2['total_failed'] == 0
    update_job_status(
        manifest, "matrix1_002", 3,
        status="failed",
        error_type="FAILED_IMMEDIATE_OUT_OF_MEMORY"
    )

    # Verify final state
    manifest = load_manifest(3)

    assert manifest["matrix1_000"]["status"] == "completed"
    assert manifest["matrix1_000"]["paths_completed"] == 10
    assert manifest["matrix1_000"]["paths_failed"] == 0

    assert manifest["matrix1_001"]["status"] == "pending"
    assert manifest["matrix1_001"]["memory_tier"] == 250
    assert manifest["matrix1_001"]["paths_completed"] == 8
    assert manifest["matrix1_001"]["paths_failed"] == 2

    assert manifest["matrix1_002"]["status"] == "failed"
    assert manifest["matrix1_002"]["error_type"] == "FAILED_IMMEDIATE_OUT_OF_MEMORY"


def test_memory_tier_progression():
    """Test memory tier increment logic."""
    assert increment_memory_tier(180) == 250
    assert increment_memory_tier(250) == 500
    assert increment_memory_tier(500) == 1000
    assert increment_memory_tier(1000) == 1400
    assert increment_memory_tier(1400) is None  # Max tier


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
