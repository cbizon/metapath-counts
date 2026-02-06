"""End-to-end tests for the 2-hop analysis pipeline.

These tests verify the 2-hop pipeline produces correct predictor counts
and properly aggregates across all paths.
"""

import pytest
from pathlib import Path

from conftest import parse_raw_results
from golden_graph import GRAPH_STATS


class Test2HopAnalysisRuns:
    """Basic tests that 2-hop analysis completes."""

    def test_result_file_created(self, pipeline_2hop):
        """Verify result file was created."""
        result_file = pipeline_2hop["result_file"]
        assert result_file.exists()

    def test_grouped_files_created(self, pipeline_2hop):
        """Verify grouped output files were created."""
        grouped_results = pipeline_2hop["grouped_results"]
        assert len(grouped_results) > 0, "No grouped output files created"


class Test2HopPrecomputedCounts:
    """Test that 2-hop predictor counts are precomputed correctly."""

    def test_explicit_2hop_path_exists(self, pipeline_2hop):
        """Verify explicit 2-hop paths are in the precomputed counts."""
        nhop_counts = pipeline_2hop["aggregated_nhop_counts"]

        # Gene_A --regulates--> Gene_B --affects--> Disease_P
        # This is one explicit 2-hop path
        path = "Gene|regulates|F|Gene|affects|F|Disease"
        assert path in nhop_counts, f"Expected path not found: {path}"
        assert nhop_counts[path] >= 1

    def test_aggregated_2hop_includes_contributions(self, pipeline_2hop):
        """Aggregated 2-hop counts should include all contributing explicit paths."""
        nhop_counts = pipeline_2hop["aggregated_nhop_counts"]

        # BiologicalEntity|related_to|F|BiologicalEntity|related_to|F|Disease
        # should include contributions from all 2-hop paths ending in Disease
        path = "BiologicalEntity|related_to|F|BiologicalEntity|related_to|F|Disease"

        if path in nhop_counts:
            # Should be > 0 since we have 2-hop paths to Disease
            assert nhop_counts[path] > 0


class Test2HopNoPseudoTypesInOutput:
    """Verify pseudo-types are filtered from 2-hop grouped output."""

    def test_no_pseudo_type_in_predictor_paths(self, pipeline_2hop):
        """Predictor (2-hop) paths in output should not contain pseudo-types."""
        grouped_results = pipeline_2hop["grouped_results"]

        for filename, rows in grouped_results.items():
            for row in rows:
                nhop_path = row.get("2hop_metapath", "")
                assert '+' not in nhop_path, (
                    f"Pseudo-type in 2-hop predictor path: {nhop_path} (file: {filename})"
                )

    def test_no_pseudo_type_in_target_filenames(self, pipeline_2hop):
        """Output filenames should not contain pseudo-types."""
        grouped_results = pipeline_2hop["grouped_results"]

        for filename in grouped_results.keys():
            assert '+' not in filename, f"Pseudo-type in filename: {filename}"


class Test2HopPredictorCountsAreGlobal:
    """Test that predictor counts come from precomputed global counts.

    This is THE key test for the bug we fixed: predictor counts should be
    the global count for that path variant, not the count from just the
    rows in the current type-pair partition.
    """

    def test_same_predictor_has_consistent_count(self, pipeline_2hop):
        """Same predictor path should have same count across different targets."""
        grouped_results = pipeline_2hop["grouped_results"]

        # Collect all (predictor, count) pairs
        predictor_counts = {}

        for filename, rows in grouped_results.items():
            for row in rows:
                predictor = row.get("2hop_metapath", "")
                count = row.get("2hop_count", 0)

                if predictor in predictor_counts:
                    # Should be same count!
                    assert predictor_counts[predictor] == count, (
                        f"Predictor {predictor} has inconsistent counts: "
                        f"{predictor_counts[predictor]} vs {count}"
                    )
                else:
                    predictor_counts[predictor] = count

    def test_predictor_count_matches_precomputed(self, pipeline_2hop):
        """Predictor counts in output should match precomputed counts."""
        grouped_results = pipeline_2hop["grouped_results"]
        nhop_counts = pipeline_2hop["aggregated_nhop_counts"]

        for filename, rows in grouped_results.items():
            for row in rows:
                predictor = row.get("2hop_metapath", "")
                output_count = row.get("2hop_count", 0)

                if predictor in nhop_counts:
                    precomputed_count = nhop_counts[predictor]
                    assert output_count == precomputed_count, (
                        f"Predictor {predictor}: output count {output_count} "
                        f"!= precomputed {precomputed_count}"
                    )


class Test2HopMetrics:
    """Test that 2-hop metrics are reasonable."""

    def test_precision_with_overlap(self, pipeline_2hop):
        """When overlap > 0, precision should be > 0."""
        grouped_results = pipeline_2hop["grouped_results"]

        for filename, rows in grouped_results.items():
            for row in rows:
                overlap = row.get("overlap", 0)
                precision = row.get("precision", 0)

                if overlap > 0:
                    assert precision > 0, (
                        f"overlap={overlap} but precision={precision}"
                    )

    def test_recall_with_overlap(self, pipeline_2hop):
        """When overlap > 0, recall should be > 0."""
        grouped_results = pipeline_2hop["grouped_results"]

        for filename, rows in grouped_results.items():
            for row in rows:
                overlap = row.get("overlap", 0)
                recall = row.get("recall", 0)

                if overlap > 0:
                    assert recall > 0, (
                        f"overlap={overlap} but recall={recall}"
                    )


class Test2HopTypePairs:
    """Test type pair handling for 2-hop."""

    def test_type_pairs_exclude_pseudo_types(self, pipeline_2hop):
        """Type pairs should not include pseudo-types."""
        type_pairs = pipeline_2hop["type_pairs"]

        for t1, t2 in type_pairs:
            assert '+' not in t1, f"Pseudo-type in type pair: {t1}"
            assert '+' not in t2, f"Pseudo-type in type pair: {t2}"
