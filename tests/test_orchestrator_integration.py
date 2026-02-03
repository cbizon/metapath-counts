"""
Integration test for orchestrator with path tracking.

This test verifies the orchestrator workflow:
1. Create realistic manifest and path tracking files
2. Simulate job lifecycle (pending -> running -> completed/failed)
3. Verify orchestrator makes correct retry decisions based on path statistics
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pytest

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from orchestrate_hop_analysis import get_path_statistics
from slurm_utils import (
    load_manifest,
    save_manifest,
    update_job_status,
    increment_memory_tier,
    HOP_ANALYSIS_MEMORY_TIERS,
)
from path_tracker import (
    record_completed_path,
    record_failed_path,
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


def get_manifest_path(results_dir):
    """Get manifest path from results directory."""
    return os.path.join(results_dir, "manifest.json")


def create_test_manifest(results_dir, num_jobs=3):
    """Create a test manifest with a few jobs."""
    manifest_path = get_manifest_path(results_dir)

    manifest = {
        "_metadata": {
            "matrices_dir": "/fake/matrices",
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

    save_manifest(manifest, manifest_path)
    return manifest_path


def simulate_job_completion_all_paths_successful(results_dir, matrix1_index, num_paths=10):
    """Simulate a job that completed all paths successfully."""
    for i in range(num_paths):
        path_id = f"Type{matrix1_index}|pred|F|Type{matrix1_index+1}__path{i}"
        record_completed_path(path_id, results_dir, matrix1_index)


def simulate_job_completion_with_failures(results_dir, matrix1_index,
                                          num_completed=7, num_failed=3,
                                          failure_tier=180):
    """Simulate a job that completed some paths and failed others."""
    for i in range(num_completed):
        path_id = f"Type{matrix1_index}|pred|F|Type{matrix1_index+1}__completed_path{i}"
        record_completed_path(path_id, results_dir, matrix1_index)

    for i in range(num_failed):
        path_id = f"Type{matrix1_index}|pred|F|Type{matrix1_index+1}__failed_path{i}"
        record_failed_path(path_id, results_dir, matrix1_index,
                          memory_gb=failure_tier, reason="oom")


class TestPathStatistics:
    """Test path statistics collection."""

    def test_all_paths_successful(self, temp_workspace):
        """Test get_path_statistics when all paths complete."""
        workspace, results_dir, _ = temp_workspace

        simulate_job_completion_all_paths_successful(results_dir, 0, num_paths=10)

        stats = get_path_statistics(0, 3, 180)
        assert stats['completed_count'] == 10
        assert stats['failed_at_current_tier'] == 0
        assert stats['total_failed'] == 0

    def test_some_paths_failed(self, temp_workspace):
        """Test get_path_statistics with failures at current tier."""
        workspace, results_dir, _ = temp_workspace

        simulate_job_completion_with_failures(
            results_dir, 0,
            num_completed=7, num_failed=3,
            failure_tier=180
        )

        stats = get_path_statistics(0, 3, 180)
        assert stats['completed_count'] == 7
        assert stats['failed_at_current_tier'] == 3
        assert stats['total_failed'] == 3

    def test_failures_at_lower_tier(self, temp_workspace):
        """Test that failures at lower tier are categorized correctly."""
        workspace, results_dir, _ = temp_workspace

        simulate_job_completion_with_failures(
            results_dir, 0,
            num_completed=7, num_failed=3,
            failure_tier=180
        )

        # At 250GB tier, the 180GB failures should be "lower tier"
        stats = get_path_statistics(0, 3, 250)
        assert stats['completed_count'] == 7
        assert stats['failed_at_current_tier'] == 0
        assert stats['failed_at_lower_tiers'] == 3
        assert stats['total_failed'] == 3


class TestManifestWorkflow:
    """Test manifest update workflow."""

    def test_load_and_update_manifest(self, temp_workspace):
        """Test loading and updating manifest."""
        workspace, results_dir, _ = temp_workspace

        manifest_path = create_test_manifest(results_dir, num_jobs=2)

        manifest = load_manifest(manifest_path)
        assert manifest["matrix1_000"]["status"] == "pending"

        simulate_job_completion_all_paths_successful(results_dir, 0, num_paths=15)

        stats = get_path_statistics(0, 3, 180)

        update_job_status(
            manifest, "matrix1_000", manifest_path,
            status="completed",
            paths_completed=stats['completed_count'],
            paths_failed=0
        )

        manifest = load_manifest(manifest_path)
        assert manifest["matrix1_000"]["status"] == "completed"
        assert manifest["matrix1_000"]["paths_completed"] == 15


class TestRetryDecisions:
    """Test retry decision logic."""

    def test_no_retry_when_all_successful(self, temp_workspace):
        """Test that no retry is needed when all paths complete."""
        workspace, results_dir, _ = temp_workspace

        manifest_path = create_test_manifest(results_dir, num_jobs=1)
        simulate_job_completion_all_paths_successful(results_dir, 0, num_paths=20)

        stats = get_path_statistics(0, 3, 180)
        assert stats['total_failed'] == 0

        # Should mark as completed
        manifest = load_manifest(manifest_path)
        update_job_status(
            manifest, "matrix1_000", manifest_path,
            status="completed",
            paths_completed=stats['completed_count'],
            paths_failed=0
        )

        manifest = load_manifest(manifest_path)
        assert manifest["matrix1_000"]["status"] == "completed"

    def test_retry_at_next_tier_on_failure(self, temp_workspace):
        """Test that job retries at next tier when paths fail."""
        workspace, results_dir, _ = temp_workspace

        manifest_path = create_test_manifest(results_dir, num_jobs=1)
        simulate_job_completion_with_failures(
            results_dir, 0,
            num_completed=15, num_failed=5,
            failure_tier=180
        )

        stats = get_path_statistics(0, 3, 180)
        assert stats['total_failed'] == 5

        current_memory = 180
        next_memory = increment_memory_tier(current_memory, HOP_ANALYSIS_MEMORY_TIERS)
        assert next_memory == 250

        manifest = load_manifest(manifest_path)
        update_job_status(
            manifest, "matrix1_000", manifest_path,
            status="pending",
            memory_tier=next_memory,
            attempts=1,
            job_id=None,
            paths_completed=stats['completed_count'],
            paths_failed=stats['total_failed']
        )

        manifest = load_manifest(manifest_path)
        assert manifest["matrix1_000"]["status"] == "pending"
        assert manifest["matrix1_000"]["memory_tier"] == 250

    def test_no_retry_at_max_tier(self, temp_workspace):
        """Test that job doesn't retry when at max memory tier."""
        workspace, results_dir, _ = temp_workspace

        manifest_path = create_test_manifest(results_dir, num_jobs=1)

        manifest = load_manifest(manifest_path)
        manifest["matrix1_000"]["memory_tier"] = 1400
        save_manifest(manifest, manifest_path)

        simulate_job_completion_with_failures(
            results_dir, 0,
            num_completed=18, num_failed=2,
            failure_tier=1400
        )

        next_memory = increment_memory_tier(1400, HOP_ANALYSIS_MEMORY_TIERS)
        assert next_memory is None

        # Should mark as completed with partial results
        manifest = load_manifest(manifest_path)
        stats = get_path_statistics(0, 3, 1400)

        update_job_status(
            manifest, "matrix1_000", manifest_path,
            status="completed",
            paths_completed=stats['completed_count'],
            paths_failed=stats['total_failed'],
            error_type="PARTIAL_MAX_MEMORY"
        )

        manifest = load_manifest(manifest_path)
        assert manifest["matrix1_000"]["status"] == "completed"
        assert manifest["matrix1_000"]["error_type"] == "PARTIAL_MAX_MEMORY"


class TestMemoryTierProgression:
    """Test memory tier increment logic."""

    def test_tier_progression(self):
        """Test memory tier increments correctly."""
        tiers = HOP_ANALYSIS_MEMORY_TIERS

        assert increment_memory_tier(100, tiers) == 180
        assert increment_memory_tier(180, tiers) == 250
        assert increment_memory_tier(250, tiers) == 500
        assert increment_memory_tier(500, tiers) == 1000
        assert increment_memory_tier(1000, tiers) == 1400
        assert increment_memory_tier(1400, tiers) is None


class TestMultiJobWorkflow:
    """Test orchestrator with multiple jobs."""

    def test_multi_job_different_outcomes(self, temp_workspace):
        """Test handling multiple jobs with different outcomes."""
        workspace, results_dir, _ = temp_workspace

        manifest_path = create_test_manifest(results_dir, num_jobs=3)

        # Job 0: All paths successful
        simulate_job_completion_all_paths_successful(results_dir, 0, num_paths=10)

        # Job 1: Some failures
        simulate_job_completion_with_failures(
            results_dir, 1,
            num_completed=8, num_failed=2,
            failure_tier=180
        )

        # Job 2: No progress (immediate failure) - don't create any tracking files

        manifest = load_manifest(manifest_path)

        # Process job 0
        stats_0 = get_path_statistics(0, 3, 180)
        assert stats_0['total_failed'] == 0
        update_job_status(
            manifest, "matrix1_000", manifest_path,
            status="completed",
            paths_completed=stats_0['completed_count'],
            paths_failed=0
        )

        # Process job 1
        manifest = load_manifest(manifest_path)
        stats_1 = get_path_statistics(1, 3, 180)
        assert stats_1['total_failed'] == 2
        update_job_status(
            manifest, "matrix1_001", manifest_path,
            status="pending",
            memory_tier=250,
            attempts=1,
            paths_completed=stats_1['completed_count'],
            paths_failed=stats_1['total_failed']
        )

        # Process job 2
        manifest = load_manifest(manifest_path)
        stats_2 = get_path_statistics(2, 3, 180)
        assert stats_2['completed_count'] == 0 and stats_2['total_failed'] == 0
        update_job_status(
            manifest, "matrix1_002", manifest_path,
            status="failed",
            error_type="FAILED_IMMEDIATE_OUT_OF_MEMORY"
        )

        # Verify final state
        manifest = load_manifest(manifest_path)

        assert manifest["matrix1_000"]["status"] == "completed"
        assert manifest["matrix1_000"]["paths_completed"] == 10

        assert manifest["matrix1_001"]["status"] == "pending"
        assert manifest["matrix1_001"]["memory_tier"] == 250

        assert manifest["matrix1_002"]["status"] == "failed"
        assert manifest["matrix1_002"]["error_type"] == "FAILED_IMMEDIATE_OUT_OF_MEMORY"
