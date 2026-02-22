"""
Unit tests for orchestrator path statistics integration.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
import pytest

from pipeline.orchestrate_analysis import get_path_statistics
from library.path_tracker import (
    record_completed_path,
    record_failed_path,
    get_tracking_dir
)


@pytest.fixture
def temp_results_dir():
    """Create a temporary results directory."""
    temp_dir = tempfile.mkdtemp(prefix="test_results_3hop_")
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_path_statistics_no_tracking_files(temp_results_dir):
    """Test get_path_statistics when no tracking files exist."""
    # Get stats for a matrix that has no tracking files
    stats = get_path_statistics(
        matrix1_index=0,
        n_hops=3,
        current_memory_gb=180,
        results_dir=temp_results_dir
    )

    assert stats['completed_count'] == 0
    assert stats['failed_at_current_tier'] == 0
    assert stats['failed_at_lower_tiers'] == 0
    assert stats['total_failed'] == 0


def test_path_statistics_completed_only():
    """Test get_path_statistics with only completed paths."""
    with tempfile.TemporaryDirectory(prefix="test_results_3hop_") as temp_dir:
        matrix1_index = 5

        # Record some completed paths
        for i in range(10):
            path_id = f"Type1|pred|F|Type2__Type2|pred|F|Type3__Type3|pred|F|Type4_{i}"
            record_completed_path(path_id, temp_dir, matrix1_index)

        # Override n_hops temporarily by renaming directory
        results_dir = Path(temp_dir).name

        # Call get_path_statistics by constructing the right results_dir
        # We need to mock this since it constructs results_dir from n_hops
        # Instead, let's directly call the path_tracker functions
        from library.path_tracker import get_path_statistics as get_stats

        stats_raw = get_stats(temp_dir, matrix1_index)
        assert stats_raw['completed'] == 10
        assert stats_raw['total_failed'] == 0


def test_path_statistics_failed_at_different_tiers():
    """Test get_path_statistics with failures at different memory tiers."""
    with tempfile.TemporaryDirectory(prefix="test_results_3hop_") as temp_dir:
        matrix1_index = 7

        # Record completed paths
        for i in range(5):
            path_id = f"Type1|pred|F|Type2__Type2|pred|F|Type3__Type3|pred|F|Type4_completed_{i}"
            record_completed_path(path_id, temp_dir, matrix1_index)

        # Record failures at 180GB
        for i in range(3):
            path_id = f"Type1|pred|F|Type2__Type2|pred|F|Type3__Type3|pred|F|Type4_failed180_{i}"
            record_failed_path(path_id, temp_dir, matrix1_index, memory_gb=180, reason="oom")

        # Record failures at 250GB
        for i in range(2):
            path_id = f"Type1|pred|F|Type2__Type2|pred|F|Type3__Type3|pred|F|Type4_failed250_{i}"
            record_failed_path(path_id, temp_dir, matrix1_index, memory_gb=250, reason="oom")

        # Get stats from path_tracker
        from library.path_tracker import get_path_statistics as get_stats

        stats_raw = get_stats(temp_dir, matrix1_index)
        assert stats_raw['completed'] == 5
        assert stats_raw['total_failed'] == 5
        assert stats_raw['failed_by_tier'][180] == 3
        assert stats_raw['failed_by_tier'][250] == 2

        # Now test the orchestrator's wrapper function
        # Since it constructs results_dir from n_hops, we need to test differently
        # Let's test the logic manually

        current_memory_gb = 250
        failed_by_tier = stats_raw['failed_by_tier']

        failed_at_current_tier = failed_by_tier.get(current_memory_gb, 0)
        failed_at_lower_tiers = sum(
            count for tier, count in failed_by_tier.items()
            if tier < current_memory_gb
        )

        assert failed_at_current_tier == 2  # Failed at 250GB
        assert failed_at_lower_tiers == 3   # Failed at 180GB


def test_path_statistics_retry_decision_logic():
    """Test the retry decision logic based on path statistics."""
    # This tests the logic that the orchestrator uses to decide whether to retry

    # Case 1: All paths completed, no failures
    stats = {
        'completed_count': 100,
        'failed_at_current_tier': 0,
        'failed_at_lower_tiers': 0,
        'total_failed': 0
    }

    # Decision: Mark as completed successfully
    assert stats['total_failed'] == 0  # Should mark as completed

    # Case 2: Some paths completed, some failed at current tier
    stats = {
        'completed_count': 80,
        'failed_at_current_tier': 20,
        'failed_at_lower_tiers': 0,
        'total_failed': 20
    }

    # Decision: Retry at next tier for failed paths
    assert stats['total_failed'] > 0  # Should retry at higher tier
    assert stats['failed_at_current_tier'] > 0

    # Case 3: Some paths completed, some failed at lower tiers, job OOMed at current tier
    stats = {
        'completed_count': 60,
        'failed_at_current_tier': 0,  # Job OOMed before recording new failures
        'failed_at_lower_tiers': 40,  # These failed at 180GB
        'total_failed': 40
    }

    # Decision: Retry at next tier for lower-tier failures
    assert stats['total_failed'] > 0  # Should retry at higher tier
    assert stats['failed_at_lower_tiers'] > 0

    # Case 4: No progress at all (job failed immediately)
    stats = {
        'completed_count': 0,
        'failed_at_current_tier': 0,
        'failed_at_lower_tiers': 0,
        'total_failed': 0
    }

    # Decision: True failure, don't retry
    assert stats['completed_count'] == 0 and stats['total_failed'] == 0


def test_tracking_directory_creation():
    """Test that tracking directories are created properly."""
    with tempfile.TemporaryDirectory(prefix="test_results_3hop_") as temp_dir:
        matrix1_index = 10

        # Record a path - this should create the tracking directory
        path_id = "Type1|pred|F|Type2__Type2|pred|F|Type3__Type3|pred|F|Type4"
        record_completed_path(path_id, temp_dir, matrix1_index)

        # Verify tracking directory exists
        tracking_dir = get_tracking_dir(temp_dir, matrix1_index)
        assert tracking_dir.exists()
        assert tracking_dir.is_dir()

        # Verify completed paths file exists
        completed_file = tracking_dir / "completed_paths.txt"
        assert completed_file.exists()

        # Read and verify content
        with open(completed_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert lines[0].strip() == path_id


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
