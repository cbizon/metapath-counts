"""
Unit tests for path_tracker module.

Tests path ID generation, tracking files, and downstream path enumeration.
"""

import pytest
import tempfile
import shutil
from library.path_tracker import (
    generate_path_id,
    parse_path_id,
    load_completed_paths,
    load_failed_paths,
    load_failed_paths_by_tier,
    record_completed_path,
    record_failed_path,
    record_path_in_progress,
    clear_path_in_progress,
    read_path_in_progress,
    count_completed_paths,
    count_failed_paths,
    get_path_statistics
)


class TestPathIDGeneration:
    """Test path ID generation and parsing."""

    def test_2hop_path(self):
        """Test 2-hop path ID generation."""
        node_types = ["SmallMolecule", "Disease"]
        predicates = ["treats"]
        directions = ["F"]

        path_id = generate_path_id(node_types, predicates, directions)
        assert path_id == "SmallMolecule|treats|F|Disease"

        parsed = parse_path_id(path_id)
        assert parsed == (node_types, predicates, directions)

    def test_3hop_path(self):
        """Test 3-hop path ID generation."""
        node_types = ["SmallMolecule", "Disease", "Gene"]
        predicates = ["treats", "affects"]
        directions = ["F", "F"]

        path_id = generate_path_id(node_types, predicates, directions)
        assert path_id == "SmallMolecule|treats|F|Disease__Disease|affects|F|Gene"

        parsed = parse_path_id(path_id)
        assert parsed == (node_types, predicates, directions)

    def test_4hop_path(self):
        """Test 4-hop path ID with mixed directions."""
        node_types = ["SmallMolecule", "Disease", "Gene", "Protein"]
        predicates = ["treats", "affects", "regulates"]
        directions = ["F", "R", "A"]

        path_id = generate_path_id(node_types, predicates, directions)
        expected = "SmallMolecule|treats|F|Disease__Disease|affects|R|Gene__Gene|regulates|A|Protein"
        assert path_id == expected

        parsed = parse_path_id(path_id)
        assert parsed == (node_types, predicates, directions)

    def test_invalid_lengths(self):
        """Test that mismatched lengths raise errors."""
        with pytest.raises(ValueError):
            # Too few node types
            generate_path_id(["A"], ["pred"], ["F"])

        with pytest.raises(ValueError):
            # Mismatched predicates/directions
            generate_path_id(["A", "B"], ["pred"], ["F", "R"])

    def test_parse_invalid_format(self):
        """Test parsing invalid path IDs."""
        with pytest.raises(ValueError):
            parse_path_id("invalid_format")

        with pytest.raises(ValueError):
            parse_path_id("A|B|C")  # Missing direction/target


class TestCompletedPathsTracking:
    """Test completed paths tracking."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_record_and_load_completed(self):
        """Test recording and loading completed paths."""
        path_id = "SmallMolecule|treats|F|Disease"

        # Record completion
        record_completed_path(path_id, self.temp_dir, matrix1_index=0)

        # Load and verify
        completed = load_completed_paths(self.temp_dir, matrix1_index=0)
        assert path_id in completed
        assert len(completed) == 1

    def test_multiple_completions(self):
        """Test recording multiple completed paths."""
        paths = [
            "SmallMolecule|treats|F|Disease",
            "Gene|regulates|F|Protein",
            "Disease|associated_with|A|Gene"
        ]

        for path_id in paths:
            record_completed_path(path_id, self.temp_dir, matrix1_index=0)

        completed = load_completed_paths(self.temp_dir, matrix1_index=0)
        assert len(completed) == 3
        for path_id in paths:
            assert path_id in completed

    def test_empty_completed(self):
        """Test loading when no paths completed."""
        completed = load_completed_paths(self.temp_dir, matrix1_index=0)
        assert len(completed) == 0

    def test_count_completed(self):
        """Test counting completed paths."""
        paths = ["Path1", "Path2", "Path3"]
        for path_id in paths:
            record_completed_path(path_id, self.temp_dir, matrix1_index=0)

        count = count_completed_paths(self.temp_dir, matrix1_index=0)
        assert count == 3


class TestFailedPathsTracking:
    """Test failed paths tracking."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_record_and_load_failed(self):
        """Test recording and loading failed paths."""
        path_id = "SmallMolecule|treats|F|Disease"

        # Record failure
        record_failed_path(path_id, self.temp_dir, matrix1_index=0,
                          memory_gb=180, depth=2, reason="oom")

        # Load and verify
        failed = load_failed_paths(self.temp_dir, matrix1_index=0, memory_gb=180)
        assert path_id in failed
        assert len(failed) == 1

    def test_multiple_tiers(self):
        """Test failures at different memory tiers."""
        path_id = "SmallMolecule|treats|F|Disease"

        # Record failures at different tiers
        record_failed_path(path_id, self.temp_dir, matrix1_index=0,
                          memory_gb=180, depth=2, reason="oom")
        record_failed_path(path_id, self.temp_dir, matrix1_index=0,
                          memory_gb=250, depth=2, reason="oom")

        # Load by tier
        failed_180 = load_failed_paths(self.temp_dir, matrix1_index=0, memory_gb=180)
        failed_250 = load_failed_paths(self.temp_dir, matrix1_index=0, memory_gb=250)

        assert path_id in failed_180
        assert path_id in failed_250

        # Load all tiers
        failed_all = load_failed_paths(self.temp_dir, matrix1_index=0, memory_gb=None)
        assert path_id in failed_all

    def test_load_by_tier_dict(self):
        """Test loading failed paths grouped by tier."""
        paths_180 = ["Path1", "Path2"]
        paths_250 = ["Path3"]

        for path_id in paths_180:
            record_failed_path(path_id, self.temp_dir, matrix1_index=0,
                              memory_gb=180, depth=2, reason="oom")

        for path_id in paths_250:
            record_failed_path(path_id, self.temp_dir, matrix1_index=0,
                              memory_gb=250, depth=2, reason="oom")

        by_tier = load_failed_paths_by_tier(self.temp_dir, matrix1_index=0)

        assert len(by_tier[180]) == 2
        assert len(by_tier[250]) == 1
        assert "Path1" in by_tier[180]
        assert "Path3" in by_tier[250]

    def test_count_failed(self):
        """Test counting failed paths."""
        paths = ["Path1", "Path2", "Path3"]
        for path_id in paths:
            record_failed_path(path_id, self.temp_dir, matrix1_index=0,
                              memory_gb=180, depth=2, reason="oom")

        count = count_failed_paths(self.temp_dir, matrix1_index=0, memory_gb=180)
        assert count == 3


class TestPathInProgress:
    """Test path in progress tracking."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_record_and_read_progress(self):
        """Test recording and reading path in progress."""
        path_id = "SmallMolecule|treats|F|Disease"

        # Record in progress
        record_path_in_progress(path_id, self.temp_dir, matrix1_index=0,
                               depth=2, memory_gb=180)

        # Read back
        progress = read_path_in_progress(self.temp_dir, matrix1_index=0)

        assert progress is not None
        assert progress['path_id'] == path_id
        assert progress['depth'] == 2
        assert progress['memory_gb'] == 180

    def test_overwrite_progress(self):
        """Test that progress file is overwritten."""
        # Record first path
        record_path_in_progress("Path1", self.temp_dir, matrix1_index=0,
                               depth=1, memory_gb=180)

        # Record second path (should overwrite)
        record_path_in_progress("Path2", self.temp_dir, matrix1_index=0,
                               depth=2, memory_gb=180)

        # Should only see second path
        progress = read_path_in_progress(self.temp_dir, matrix1_index=0)
        assert progress['path_id'] == "Path2"
        assert progress['depth'] == 2

    def test_clear_progress(self):
        """Test clearing path in progress."""
        record_path_in_progress("Path1", self.temp_dir, matrix1_index=0,
                               depth=1, memory_gb=180)

        clear_path_in_progress(self.temp_dir, matrix1_index=0)

        progress = read_path_in_progress(self.temp_dir, matrix1_index=0)
        assert progress is None

    def test_read_nonexistent(self):
        """Test reading when no path in progress."""
        progress = read_path_in_progress(self.temp_dir, matrix1_index=0)
        assert progress is None


class TestPathStatistics:
    """Test path statistics aggregation."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_statistics(self):
        """Test getting comprehensive statistics."""
        # Record some completions
        for i in range(100):
            record_completed_path(f"Completed_{i}", self.temp_dir, matrix1_index=0)

        # Record some failures at different tiers
        for i in range(10):
            record_failed_path(f"Failed_180_{i}", self.temp_dir, matrix1_index=0,
                              memory_gb=180, depth=2, reason="oom")

        for i in range(5):
            record_failed_path(f"Failed_250_{i}", self.temp_dir, matrix1_index=0,
                              memory_gb=250, depth=2, reason="oom")

        # Get statistics
        stats = get_path_statistics(self.temp_dir, matrix1_index=0)

        assert stats['completed'] == 100
        assert stats['total_failed'] == 15
        assert stats['failed_by_tier'][180] == 10
        assert stats['failed_by_tier'][250] == 5


class TestSeparateMatrix1Jobs:
    """Test that different Matrix1 jobs have separate tracking."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_separate_completed(self):
        """Test that completed paths are tracked separately per Matrix1."""
        record_completed_path("Path1", self.temp_dir, matrix1_index=0)
        record_completed_path("Path2", self.temp_dir, matrix1_index=1)

        completed_0 = load_completed_paths(self.temp_dir, matrix1_index=0)
        completed_1 = load_completed_paths(self.temp_dir, matrix1_index=1)

        assert "Path1" in completed_0
        assert "Path1" not in completed_1
        assert "Path2" not in completed_0
        assert "Path2" in completed_1

    def test_separate_failed(self):
        """Test that failed paths are tracked separately per Matrix1."""
        record_failed_path("Path1", self.temp_dir, matrix1_index=0,
                          memory_gb=180, depth=2, reason="oom")
        record_failed_path("Path2", self.temp_dir, matrix1_index=1,
                          memory_gb=180, depth=2, reason="oom")

        failed_0 = load_failed_paths(self.temp_dir, matrix1_index=0, memory_gb=180)
        failed_1 = load_failed_paths(self.temp_dir, matrix1_index=1, memory_gb=180)

        assert "Path1" in failed_0
        assert "Path1" not in failed_1
        assert "Path2" not in failed_0
        assert "Path2" in failed_1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
