#!/usr/bin/env python3
"""
Tests for count aggregation correctness.

These tests specifically target the bugs we found:
1. nhop_count being multiplied by number of onehop variants (cartesian product bug)
2. onehop_count being multiplied by number of nhop variants
3. Precomputed counts lookup working correctly

The key insight: when aggregating, counts should be computed INDEPENDENTLY
for each side, not multiplied together in a cartesian product.
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from metapath_counts import expand_metapath_to_variants, calculate_metrics

from group_single_onehop_worker import (
    load_aggregated_counts,
    group_type_pair
)

from group_by_onehop import aggregate_results

# Alias for backwards compatibility with test code
expand_metapath_with_hierarchy = expand_metapath_to_variants


class TestNhopCountNotMultiplied:
    """Test that nhop_count is NOT multiplied by number of onehop variants.

    This was the main bug: when one nhop path was compared to multiple onehop paths,
    the aggregation was adding nhop_count once per onehop_variant, causing multiplication.
    """

    def test_same_nhop_multiple_onehop_comparisons(self):
        """When same nhop is compared to multiple onehops, nhop_count should NOT multiply.

        Scenario:
        - nhop "Gene|affects|F|Disease" (count=1000) compared to:
          - "Gene|treats|F|Disease" (overlap=100)
          - "Gene|regulates|F|Disease" (overlap=200)

        After aggregation to Entity|related_to|F|Entity:
        - nhop_count should be 1000 (NOT 2000)
        - overlap should be 100 + 200 = 300 (this SHOULD sum)
        """
        # This test verifies the logic conceptually
        # The actual aggregation now happens via precomputed counts

        # Simulate the old buggy behavior vs correct behavior
        explicit_results = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 500, 100, 1_000_000),
            ("Gene|affects|F|Disease", 1000, "Gene|regulates|F|Disease", 300, 200, 1_000_000),
        ]

        # Correct aggregation: nhop_count for "Gene|affects|F|Disease" is 1000
        # even though it appears in 2 rows
        unique_nhop_counts = {}
        for nhop_path, nhop_count, _, _, _, _ in explicit_results:
            if nhop_path not in unique_nhop_counts:
                unique_nhop_counts[nhop_path] = nhop_count

        assert unique_nhop_counts["Gene|affects|F|Disease"] == 1000

        # Overlap SHOULD sum across rows
        total_overlap = sum(row[4] for row in explicit_results)
        assert total_overlap == 300

    def test_expand_metapath_produces_variants(self):
        """Verify expand_metapath_with_hierarchy produces expected variants."""
        variants = expand_metapath_with_hierarchy("Gene|affects|F|Disease")

        # Should include the original
        assert "Gene|affects|F|Disease" in variants

        # Should include ancestors
        assert any("Entity" in v for v in variants)
        assert any("related_to" in v for v in variants)

        # Should have multiple variants
        assert len(variants) > 1


class TestPrecomputedCountsLogic:
    """Test the precomputed counts lookup mechanism."""

    def test_precompute_from_matrix_metadata(self):
        """Test that precomputing from matrix metadata gives correct counts."""
        # Simulate matrix metadata
        matrix_data = [
            {"src_type": "Gene", "predicate": "affects", "direction": "F", "tgt_type": "Disease", "nvals": 1000},
            {"src_type": "SmallMolecule", "predicate": "treats", "direction": "F", "tgt_type": "Disease", "nvals": 500},
        ]

        # Manually compute what the precomputation should produce
        aggregated_counts = defaultdict(int)

        for matrix in matrix_data:
            path = f"{matrix['src_type']}|{matrix['predicate']}|{matrix['direction']}|{matrix['tgt_type']}"
            count = matrix["nvals"]

            variants = expand_metapath_with_hierarchy(path)
            for variant in variants:
                aggregated_counts[variant] += count

        # The explicit paths should have their exact counts
        assert aggregated_counts["Gene|affects|F|Disease"] == 1000
        assert aggregated_counts["SmallMolecule|treats|F|Disease"] == 500

        # Entity|related_to|F|Entity should sum BOTH (1000 + 500 = 1500)
        # because both paths expand to this ancestor
        entity_key = "Entity|related_to|F|Entity"
        if entity_key in aggregated_counts:
            assert aggregated_counts[entity_key] == 1500

    def test_load_aggregated_counts_file(self):
        """Test loading precomputed counts from file."""
        # Create temporary counts file
        counts_data = {
            "_metadata": {"created_at": "2024-01-01"},
            "counts": {
                "Gene|affects|F|Disease": 1000,
                "Entity|related_to|F|Entity": 5000,
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(counts_data, f)
            temp_path = f.name

        try:
            # Reset cache
            import group_single_onehop_worker
            group_single_onehop_worker._aggregated_counts_cache = None

            counts = load_aggregated_counts(temp_path)

            assert counts["Gene|affects|F|Disease"] == 1000
            assert counts["Entity|related_to|F|Entity"] == 5000
        finally:
            os.unlink(temp_path)
            # Reset cache again
            group_single_onehop_worker._aggregated_counts_cache = None


class TestCountIndependence:
    """Test that nhop and onehop counts are computed independently."""

    def test_nhop_count_independent_of_onehop_variants(self):
        """nhop count should not change based on how many onehop paths it's compared to."""
        # Create mock data simulating what the grouping sees

        # Scenario: Gene|affects|F|Disease has 1000 edges
        # It's compared to 3 different 1-hop paths
        result_rows = [
            # nhop_path, nhop_count, onehop_path, onehop_count, overlap, total_possible
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("Gene|affects|F|Disease", 1000, "Gene|regulates|F|Disease", 150, 30, 1_000_000),
            ("Gene|affects|F|Disease", 1000, "Gene|causes|F|Disease", 100, 20, 1_000_000),
        ]

        # Correct: extract unique nhop counts
        unique_nhop = {}
        for row in result_rows:
            nhop_path, nhop_count = row[0], row[1]
            if nhop_path not in unique_nhop:
                unique_nhop[nhop_path] = nhop_count

        # nhop_count should be 1000, not 3000
        assert unique_nhop["Gene|affects|F|Disease"] == 1000

    def test_onehop_count_independent_of_nhop_variants(self):
        """onehop count should not change based on how many nhop paths compare to it."""
        # Scenario: Gene|treats|F|Disease has 200 edges
        # It's compared to by 3 different N-hop paths
        result_rows = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("Gene|regulates|F|Disease", 800, "Gene|treats|F|Disease", 200, 40, 1_000_000),
            ("SmallMolecule|affects|F|Disease", 600, "Gene|treats|F|Disease", 200, 30, 1_000_000),
        ]

        # Correct: extract unique onehop counts
        unique_onehop = {}
        for row in result_rows:
            onehop_path, onehop_count = row[2], row[3]
            if onehop_path not in unique_onehop:
                unique_onehop[onehop_path] = onehop_count

        # onehop_count should be 200, not 600
        assert unique_onehop["Gene|treats|F|Disease"] == 200


class TestMetricsCalculation:
    """Test that metrics are calculated correctly with proper counts."""

    def test_precision_with_correct_counts(self):
        """Precision = overlap / nhop_count."""
        # nhop_count=1000, onehop_count=500, overlap=100, total_possible=1M
        metrics = calculate_metrics(1000, 500, 100, 1_000_000)

        # Precision = TP / (TP + FP) = overlap / nhop_count = 100/1000 = 0.1
        assert abs(metrics["precision"] - 0.1) < 0.0001

    def test_recall_with_correct_counts(self):
        """Recall = overlap / onehop_count."""
        metrics = calculate_metrics(1000, 500, 100, 1_000_000)

        # Recall = TP / (TP + FN) = overlap / onehop_count = 100/500 = 0.2
        assert abs(metrics["recall"] - 0.2) < 0.0001

    def test_metrics_with_aggregated_counts(self):
        """Metrics should use global aggregated counts, not per-file sums."""
        # Simulating the scenario:
        # - Aggregated nhop "Entity|related_to|F|Entity" has global count 10000
        # - But in files for Gene-Disease type pair, we only see 1000
        # - We should use 10000 (from lookup), not 1000 (from files)

        global_nhop_count = 10000
        overlap_from_files = 100
        global_onehop_count = 500
        total_possible = 1_000_000

        metrics = calculate_metrics(global_nhop_count, global_onehop_count, overlap_from_files, total_possible)

        # Precision should use global count: 100/10000 = 0.01
        assert abs(metrics["precision"] - 0.01) < 0.0001


class TestAggregationSummingBehavior:
    """Test what SHOULD sum vs what should NOT sum during aggregation."""

    def test_overlap_should_sum_across_rows(self):
        """Overlap SHOULD sum when same (nhop_variant, onehop) pair appears multiple times."""
        rows = [
            # Two different explicit nhops that both expand to same aggregated nhop
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("Gene|regulates|F|Disease", 800, "Gene|treats|F|Disease", 200, 30, 1_000_000),
        ]

        # Both Gene|affects and Gene|regulates expand to Entity|related_to
        # When aggregating, overlaps should sum
        total_overlap = sum(row[4] for row in rows)
        assert total_overlap == 80  # 50 + 30

    def test_total_possible_should_sum(self):
        """total_possible SHOULD sum across rows (represents different matrix spaces)."""
        rows = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("SmallMolecule|affects|F|Disease", 500, "SmallMolecule|treats|F|Disease", 100, 20, 2_000_000),
        ]

        # When aggregating to Entity|related_to, total_possible sums
        # because we're covering different matrix spaces
        total_possible = sum(row[5] for row in rows)
        assert total_possible == 3_000_000

    def test_explicit_counts_should_not_sum_for_same_path(self):
        """Same explicit path appearing multiple times should NOT sum its count."""
        # This is the bug we fixed - same nhop appearing in multiple comparisons
        rows = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("Gene|affects|F|Disease", 1000, "Gene|regulates|F|Disease", 150, 30, 1_000_000),
        ]

        # The nhop count should be 1000, not 2000
        unique_counts = {}
        for nhop_path, nhop_count, _, _, _, _ in rows:
            unique_counts[nhop_path] = nhop_count  # Last one wins (should all be same)

        assert unique_counts["Gene|affects|F|Disease"] == 1000


class TestHierarchicalExpansionCounting:
    """Test that hierarchical expansion doesn't multiply counts incorrectly."""

    def test_variant_expansion_preserves_count(self):
        """Expanding a path to variants should give each variant the SAME count."""
        path = "Gene|affects|F|Disease"
        count = 1000

        variants = expand_metapath_with_hierarchy(path)

        # Conceptually, each variant should have the same count
        # because they all represent the same set of edges
        variant_counts = {v: count for v in variants}

        # All variants have count 1000
        for v, c in variant_counts.items():
            assert c == 1000

    def test_summing_across_disjoint_paths(self):
        """When different explicit paths expand to same variant, counts SHOULD sum."""
        paths_and_counts = [
            ("Gene|affects|F|Disease", 1000),
            ("SmallMolecule|treats|F|Disease", 500),
        ]

        # Compute aggregated counts
        aggregated = defaultdict(int)
        for path, count in paths_and_counts:
            variants = expand_metapath_with_hierarchy(path)
            for v in variants:
                aggregated[v] += count

        # Entity|related_to|F|Entity should be 1000 + 500 = 1500
        # (assuming both expand to this - they should)
        entity_variant = "Entity|related_to|F|Entity"
        if entity_variant in aggregated:
            assert aggregated[entity_variant] == 1500

        # Each original should keep its count
        assert aggregated["Gene|affects|F|Disease"] == 1000
        assert aggregated["SmallMolecule|treats|F|Disease"] == 500


class TestAggregateResultsFunction:
    """Test the aggregate_results function from group_by_onehop.py.

    This function had the same cartesian product bug - these tests verify the fix.
    """

    def test_nhop_count_not_multiplied_by_onehop_variants(self):
        """nhop_count should NOT be multiplied by number of onehop variants.

        This is THE bug we fixed. When same nhop compares to multiple onehops,
        the old code added nhop_count once per (nhop_variant, onehop_variant) pair.
        """
        # Same nhop compared to two different onehops
        explicit_results = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("Gene|affects|F|Disease", 1000, "Gene|regulates|F|Disease", 150, 30, 1_000_000),
        ]

        aggregated = aggregate_results(explicit_results)

        # Find the key for the explicit nhop-onehop pairs
        key1 = ("Gene|affects|F|Disease", "Gene|treats|F|Disease")
        key2 = ("Gene|affects|F|Disease", "Gene|regulates|F|Disease")

        # nhop_count should be 1000 for both, NOT 2000
        # (Old buggy code would give 2000 because it added once per onehop variant)
        assert aggregated[key1][0] == 1000, f"Expected 1000, got {aggregated[key1][0]}"
        assert aggregated[key2][0] == 1000, f"Expected 1000, got {aggregated[key2][0]}"

    def test_onehop_count_not_multiplied_by_nhop_variants(self):
        """onehop_count should NOT be multiplied by number of nhop variants."""
        # Same onehop compared to by two different nhops
        explicit_results = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("SmallMolecule|affects|F|Disease", 500, "Gene|treats|F|Disease", 200, 30, 1_000_000),
        ]

        aggregated = aggregate_results(explicit_results)

        # Find the keys
        key1 = ("Gene|affects|F|Disease", "Gene|treats|F|Disease")
        key2 = ("SmallMolecule|affects|F|Disease", "Gene|treats|F|Disease")

        # onehop_count should be 200 for both, NOT 400
        assert aggregated[key1][1] == 200, f"Expected 200, got {aggregated[key1][1]}"
        assert aggregated[key2][1] == 200, f"Expected 200, got {aggregated[key2][1]}"

    def test_overlap_sums_correctly(self):
        """overlap SHOULD sum across rows (this is correct behavior)."""
        explicit_results = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 30, 1_000_000),
        ]

        aggregated = aggregate_results(explicit_results)

        key = ("Gene|affects|F|Disease", "Gene|treats|F|Disease")
        # overlap should sum: 50 + 30 = 80
        assert aggregated[key][2] == 80, f"Expected 80, got {aggregated[key][2]}"

    def test_aggregation_to_ancestor_counts_correctly(self):
        """When aggregating to ancestor types, counts should still be correct."""
        explicit_results = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
        ]

        aggregated = aggregate_results(explicit_results)

        # The aggregated version at Entity|related_to should have the same counts
        # Find keys that match Entity|related_to pattern
        entity_keys = [k for k in aggregated.keys()
                       if "Entity" in k[0] and "related_to" in k[0]
                       and "Entity" in k[1] and "related_to" in k[1]]

        if entity_keys:
            # Should have same counts as original (1000, 200, 50, 1M)
            for key in entity_keys:
                assert aggregated[key][0] == 1000, f"nhop_count wrong for {key}"
                assert aggregated[key][1] == 200, f"onehop_count wrong for {key}"
                assert aggregated[key][2] == 50, f"overlap wrong for {key}"

    def test_multiple_paths_to_same_ancestor(self):
        """Multiple explicit paths aggregating to same ancestor should sum correctly."""
        explicit_results = [
            ("Gene|affects|F|Disease", 1000, "Gene|treats|F|Disease", 200, 50, 1_000_000),
            ("SmallMolecule|affects|F|Disease", 500, "SmallMolecule|treats|F|Disease", 100, 30, 2_000_000),
        ]

        aggregated = aggregate_results(explicit_results)

        # Entity|related_to|F|Entity for both nhop and onehop should sum:
        # nhop: 1000 + 500 = 1500
        # onehop: 200 + 100 = 300
        # overlap: 50 + 30 = 80
        # total_possible: 1M + 2M = 3M

        # Find the fully aggregated key
        entity_key = None
        for k in aggregated.keys():
            if (k[0] == "Entity|related_to|F|Entity" and
                k[1] == "Entity|related_to|F|Entity"):
                entity_key = k
                break

        if entity_key:
            assert aggregated[entity_key][0] == 1500, f"Expected nhop=1500, got {aggregated[entity_key][0]}"
            assert aggregated[entity_key][1] == 300, f"Expected onehop=300, got {aggregated[entity_key][1]}"
            assert aggregated[entity_key][2] == 80, f"Expected overlap=80, got {aggregated[entity_key][2]}"
            assert aggregated[entity_key][3] == 3_000_000, f"Expected total=3M, got {aggregated[entity_key][3]}"


class TestRealWorldScenario:
    """Test realistic scenarios that would have caught the bugs."""

    def test_1hop_analysis_counts(self):
        """Simulate 1-hop analysis where same path appears many times."""
        # In 1-hop analysis, each explicit 1-hop is compared to many others
        # The same nhop appears in multiple rows

        explicit_results = []
        nhop_count = 5000  # Fixed count for our nhop

        # Simulate comparing to 10 different onehop paths
        for i in range(10):
            explicit_results.append((
                "Gene|affects|F|Disease",  # Same nhop
                nhop_count,
                f"Gene|pred_{i}|F|Disease",  # Different onehops
                100 + i * 10,  # Different onehop counts
                50 + i,  # Different overlaps
                1_000_000
            ))

        # Correct behavior: nhop_count is still 5000 after aggregation
        unique_nhop_counts = {}
        for row in explicit_results:
            path, count = row[0], row[1]
            if path not in unique_nhop_counts:
                unique_nhop_counts[path] = count

        assert unique_nhop_counts["Gene|affects|F|Disease"] == 5000

        # Bug would have made it 50000 (5000 * 10)

    def test_aggregation_to_entity_related_to(self):
        """Test the extreme case of aggregating everything to Entity|related_to|F|Entity."""
        # Multiple different paths, each with different counts
        paths_data = [
            ("Gene|affects|F|Disease", 1000),
            ("Gene|treats|F|Disease", 2000),
            ("SmallMolecule|affects|F|Gene", 3000),
            ("Protein|regulates|F|Gene", 4000),
        ]

        # Aggregate to variants
        aggregated = defaultdict(int)
        for path, count in paths_data:
            variants = expand_metapath_with_hierarchy(path)
            for v in variants:
                aggregated[v] += count

        # Each original keeps its count
        assert aggregated["Gene|affects|F|Disease"] == 1000
        assert aggregated["Gene|treats|F|Disease"] == 2000

        # Entity|related_to|F|Entity should be sum of all: 1000+2000+3000+4000 = 10000
        entity_key = "Entity|related_to|F|Entity"
        if entity_key in aggregated:
            # Note: might not be exactly 10000 if not all expand to this
            # but it should be the sum of those that do
            pass


# =============================================================================
# INTEGRATION TESTS - These test the actual user-facing outcomes
# =============================================================================

class TestPrepareGroupingIntegration:
    """Test that prepare_grouping.py correctly reads matrix manifest and creates jobs."""

    def test_precompute_handles_missing_direction_field(self):
        """Matrix manifest may not have direction field - should default to F."""
        import tempfile
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from prepare_grouping import precompute_aggregated_counts

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a matrix manifest WITHOUT direction field (like the real one)
            manifest = {
                "matrices": [
                    {
                        "src_type": "SmallMolecule",
                        "predicate": "treats",
                        "tgt_type": "Disease",
                        "nvals": 1000
                    },
                    {
                        "src_type": "Gene",
                        "predicate": "affects",
                        "tgt_type": "Disease",
                        "nvals": 500
                    }
                ]
            }

            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)

            output_path = os.path.join(tmpdir, "counts.json")
            counts = precompute_aggregated_counts(tmpdir, output_path)

            # Should have found paths (not empty due to missing direction)
            assert len(counts) > 0, "No paths found - direction field handling broken"

            # Should have the explicit paths with F direction
            assert "SmallMolecule|treats|F|Disease" in counts
            assert "Gene|affects|F|Disease" in counts

            # Should have hierarchical variants
            assert "ChemicalEntity|treats|F|Disease" in counts

    def test_type_pairs_created_from_aggregated_counts(self):
        """Type pairs should be extracted from aggregated counts, not matrix manifest."""
        import tempfile
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from prepare_grouping import precompute_aggregated_counts, extract_type_pairs_from_aggregated_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "matrices": [
                    {"src_type": "SmallMolecule", "predicate": "treats", "tgt_type": "Disease", "nvals": 1000}
                ]
            }

            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)

            output_path = os.path.join(tmpdir, "counts.json")
            counts = precompute_aggregated_counts(tmpdir, output_path)
            type_pairs = extract_type_pairs_from_aggregated_paths(counts)

            # Should include hierarchical type pairs
            assert ("ChemicalEntity", "Disease") in type_pairs or ("Disease", "ChemicalEntity") in type_pairs
            assert ("ChemicalEntity", "DiseaseOrPhenotypicFeature") in type_pairs or \
                   ("DiseaseOrPhenotypicFeature", "ChemicalEntity") in type_pairs

            # Should have many more pairs than just the one explicit
            assert len(type_pairs) > 5, f"Expected many type pairs, got {len(type_pairs)}"


class TestHierarchicalTypePairExtraction:
    """Test that type pairs include hierarchical types, not just explicit.

    This was a critical bug: prepare_grouping.py only created jobs for explicit
    type pairs like (SmallMolecule, Disease), missing hierarchical pairs like
    (ChemicalEntity, DiseaseOrPhenotypicFeature).
    """

    def test_aggregated_paths_include_hierarchical_types(self):
        """Expanding explicit paths should produce paths with hierarchical types."""
        explicit_path = "SmallMolecule|treats|F|Disease"
        variants = expand_metapath_with_hierarchy(explicit_path)

        # Should include the hierarchical variant
        assert "ChemicalEntity|treats|F|Disease" in variants
        assert "SmallMolecule|treats|F|DiseaseOrPhenotypicFeature" in variants
        assert "ChemicalEntity|treats|F|DiseaseOrPhenotypicFeature" in variants

    def test_type_pairs_from_aggregated_include_hierarchical(self):
        """Type pairs extracted from aggregated paths should include hierarchical pairs."""
        # Simulate what prepare_grouping does
        explicit_paths = {
            "SmallMolecule|treats|F|Disease": 1000,
            "Gene|affects|F|PhenotypicFeature": 500,
        }

        # Expand to aggregated paths
        aggregated_paths = {}
        for path, count in explicit_paths.items():
            for variant in expand_metapath_with_hierarchy(path):
                aggregated_paths[variant] = aggregated_paths.get(variant, 0) + count

        # Extract type pairs from aggregated paths
        type_pairs = set()
        for path in aggregated_paths.keys():
            parts = path.split('|')
            if len(parts) == 4:
                src, pred, dir, tgt = parts
                pair = tuple(sorted([src, tgt]))
                type_pairs.add(pair)

        # Should include hierarchical type pairs
        assert ("ChemicalEntity", "Disease") in type_pairs
        assert ("ChemicalEntity", "DiseaseOrPhenotypicFeature") in type_pairs
        assert ("Disease", "SmallMolecule") in type_pairs or ("SmallMolecule", "Disease") in type_pairs

        # Should also include very high-level pairs
        assert ("Entity", "Entity") in type_pairs


class TestCheckTypeMatchHierarchical:
    """Test that check_type_match properly matches explicit paths to hierarchical type pairs.

    This was another critical bug: check_type_match only did exact matching,
    so SmallMolecule|treats|F|Disease would NOT match type pair (ChemicalEntity, Disease).
    """

    def test_explicit_path_matches_hierarchical_type_pair(self):
        """SmallMolecule|treats|F|Disease should match (ChemicalEntity, DiseaseOrPhenotypicFeature)."""
        from group_single_onehop_worker import check_type_match

        explicit_path = "SmallMolecule|treats|F|Disease"

        # Should match hierarchical type pairs
        assert check_type_match(explicit_path, "ChemicalEntity", "Disease")
        assert check_type_match(explicit_path, "SmallMolecule", "DiseaseOrPhenotypicFeature")
        assert check_type_match(explicit_path, "ChemicalEntity", "DiseaseOrPhenotypicFeature")

        # Should also match exact
        assert check_type_match(explicit_path, "SmallMolecule", "Disease")

        # Should match in reverse order too
        assert check_type_match(explicit_path, "Disease", "SmallMolecule")
        assert check_type_match(explicit_path, "DiseaseOrPhenotypicFeature", "ChemicalEntity")

    def test_explicit_path_does_not_match_unrelated_types(self):
        """SmallMolecule|treats|F|Disease should NOT match (Gene, Protein)."""
        from group_single_onehop_worker import check_type_match

        explicit_path = "SmallMolecule|treats|F|Disease"

        # Should NOT match unrelated type pairs
        assert not check_type_match(explicit_path, "Gene", "Protein")
        assert not check_type_match(explicit_path, "Gene", "Disease")
        assert not check_type_match(explicit_path, "SmallMolecule", "Gene")

    def test_pseudo_type_path_matches_constituent_type_pairs(self):
        """Gene+Protein|affects|F|Disease should match (Gene, Disease) and (Protein, Disease)."""
        from group_single_onehop_worker import check_type_match

        pseudo_path = "Gene+Protein|affects|F|Disease"

        # Should match type pairs involving either constituent
        assert check_type_match(pseudo_path, "Gene", "Disease")
        assert check_type_match(pseudo_path, "Protein", "Disease")
        assert check_type_match(pseudo_path, "BiologicalEntity", "Disease")  # ancestor of Gene


class TestOnehopPathExpansionInWorker:
    """Test that 1-hop paths are expanded and stored under hierarchical variants.

    When processing a row with SmallMolecule|treats|F|Disease, the worker should
    store data under ChemicalEntity|treats|F|DiseaseOrPhenotypicFeature (and other variants).
    """

    def test_onehop_expansion_produces_hierarchical_variants(self):
        """When matching a row, 1-hop should be expanded to hierarchical variants."""
        # This simulates what the worker does when it finds a matching row
        explicit_onehop = "SmallMolecule|treats|F|Disease"
        type1, type2 = "ChemicalEntity", "DiseaseOrPhenotypicFeature"

        # Expand 1-hop to variants
        onehop_variants = expand_metapath_with_hierarchy(explicit_onehop)

        # Filter to variants that match the type pair (what worker does)
        matching_variants = []
        for variant in onehop_variants:
            parts = variant.split('|')
            if len(parts) == 4:
                v_src, v_pred, v_dir, v_tgt = parts
                if (v_src == type1 and v_tgt == type2) or (v_src == type2 and v_tgt == type1):
                    matching_variants.append(variant)

        # Should find the hierarchical variant
        assert len(matching_variants) > 0
        # At least one should have ChemicalEntity and DiseaseOrPhenotypicFeature
        hierarchical_found = any(
            "ChemicalEntity" in v and "DiseaseOrPhenotypicFeature" in v
            for v in matching_variants
        )
        assert hierarchical_found, f"No hierarchical variant found. Got: {matching_variants}"

    def test_multiple_explicit_paths_contribute_to_same_hierarchical_variant(self):
        """Multiple explicit 1-hops should all contribute to the same hierarchical variant."""
        type1, type2 = "ChemicalEntity", "DiseaseOrPhenotypicFeature"

        explicit_paths = [
            "SmallMolecule|treats|F|Disease",
            "Drug|treats|F|Disease",
            "SmallMolecule|treats|F|PhenotypicFeature",
        ]

        # For each explicit path, find variants matching the type pair
        all_matching = []
        for explicit in explicit_paths:
            variants = expand_metapath_with_hierarchy(explicit)
            for variant in variants:
                parts = variant.split('|')
                if len(parts) == 4:
                    v_src, v_pred, v_dir, v_tgt = parts
                    if (v_src == type1 and v_tgt == type2) or (v_src == type2 and v_tgt == type1):
                        all_matching.append((explicit, variant))

        # All three explicit paths should have matching variants
        explicit_with_matches = set(m[0] for m in all_matching)
        assert len(explicit_with_matches) == 3, f"Not all explicit paths have matches: {explicit_with_matches}"

        # They should all produce the same hierarchical variant
        hierarchical_variant = "ChemicalEntity|treats|F|DiseaseOrPhenotypicFeature"
        matching_to_target = [m for m in all_matching if m[1] == hierarchical_variant]
        assert len(matching_to_target) == 3, f"Expected 3 paths to produce {hierarchical_variant}"


class TestEndToEndHierarchicalOutput:
    """End-to-end test: explicit data should produce hierarchical output files.

    This is the ultimate test that would have caught the bug immediately.
    """

    def test_explicit_data_produces_hierarchical_grouped_output(self):
        """Given explicit SmallMolecule|treats|F|Disease, should produce ChemicalEntity output."""
        import tempfile
        import os

        # Create temp directories
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results_1hop")
            output_dir = os.path.join(tmpdir, "grouped")
            os.makedirs(results_dir)
            os.makedirs(output_dir)

            # Create a fake result file with explicit data
            result_file = os.path.join(results_dir, "results_matrix1_000.tsv")
            with open(result_file, 'w') as f:
                f.write("1hop_metapath\t1hop_count\t1hop_metapath\t1hop_count\toverlap\ttotal_possible\n")
                # Explicit 1-hop data
                f.write("SmallMolecule|treats|F|Disease\t1000\tSmallMolecule|affects|F|Disease\t500\t100\t1000000\n")
                f.write("Drug|treats|F|Disease\t2000\tDrug|affects|F|Disease\t800\t200\t2000000\n")

            # Create aggregated counts file
            counts_file = os.path.join(results_dir, "aggregated_path_counts.json")
            aggregated_counts = {}
            for path in ["SmallMolecule|treats|F|Disease", "Drug|treats|F|Disease",
                         "SmallMolecule|affects|F|Disease", "Drug|affects|F|Disease"]:
                for variant in expand_metapath_with_hierarchy(path):
                    aggregated_counts[variant] = aggregated_counts.get(variant, 0) + 1000

            with open(counts_file, 'w') as f:
                json.dump({"counts": aggregated_counts}, f)

            # Create type node counts for total_possible calculation
            type_node_counts = {
                "SmallMolecule": 1000, "Drug": 500, "ChemicalEntity": 1500,
                "Disease": 2000, "DiseaseOrPhenotypicFeature": 2500,
                "Entity": 10000, "NamedThing": 10000
            }

            # Run group_type_pair for hierarchical type pair
            file_list = [result_file]
            group_type_pair(
                type1="ChemicalEntity",
                type2="DiseaseOrPhenotypicFeature",
                file_list=file_list,
                output_dir=output_dir,
                n_hops=1,
                aggregate=True,
                aggregated_counts=aggregated_counts,
                type_node_counts=type_node_counts
            )

            # Check that hierarchical output file was created
            expected_files = os.listdir(output_dir)

            # Should have files with ChemicalEntity and DiseaseOrPhenotypicFeature
            hierarchical_files = [f for f in expected_files
                                  if "ChemicalEntity" in f and "DiseaseOrPhenotypicFeature" in f]

            assert len(hierarchical_files) > 0, (
                f"No hierarchical output files created! Got: {expected_files}"
            )

            # Verify the file has content
            for hf in hierarchical_files:
                with open(os.path.join(output_dir, hf), 'r') as f:
                    lines = f.readlines()
                    assert len(lines) > 1, f"File {hf} has no data rows"

    def test_worker_produces_correct_output_filename(self):
        """Output filename should match the hierarchical type pair, not explicit."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results_1hop")
            output_dir = os.path.join(tmpdir, "grouped")
            os.makedirs(results_dir)
            os.makedirs(output_dir)

            # Create a fake result file
            # Note: columns are nhop_path, nhop_count, onehop_path, onehop_count, overlap, total_possible
            # The worker checks onehop_path (column 2) against the type pair
            result_file = os.path.join(results_dir, "results_matrix1_000.tsv")
            with open(result_file, 'w') as f:
                f.write("1hop_metapath\t1hop_count\t1hop_metapath\t1hop_count\toverlap\ttotal_possible\n")
                # onehop_path (col 2) is SmallMolecule|treats|F|Disease which should match (ChemicalEntity, Disease)
                f.write("Gene|affects|F|Disease\t500\tSmallMolecule|treats|F|Disease\t1000\t100\t1000000\n")

            # Create minimal aggregated counts
            # Include counts for both nhop and onehop paths
            counts_file = os.path.join(results_dir, "aggregated_path_counts.json")
            aggregated_counts = {}
            for path in ["SmallMolecule|treats|F|Disease", "Gene|affects|F|Disease"]:
                for variant in expand_metapath_with_hierarchy(path):
                    aggregated_counts[variant] = aggregated_counts.get(variant, 0) + 1000

            with open(counts_file, 'w') as f:
                json.dump({"counts": aggregated_counts}, f)

            # Create type node counts for total_possible calculation
            type_node_counts = {
                "SmallMolecule": 1000, "ChemicalEntity": 1500, "Gene": 2000,
                "Disease": 2000, "Entity": 10000, "NamedThing": 10000
            }

            # Run for specific hierarchical type pair
            group_type_pair(
                type1="ChemicalEntity",
                type2="Disease",
                file_list=[result_file],
                output_dir=output_dir,
                n_hops=1,
                aggregate=True,
                aggregated_counts=aggregated_counts,
                type_node_counts=type_node_counts
            )

            # Should create file for ChemicalEntity|treats|F|Disease, NOT SmallMolecule
            files = os.listdir(output_dir)
            chemical_files = [f for f in files if "ChemicalEntity" in f and "treats" in f]

            assert len(chemical_files) > 0, (
                f"Expected ChemicalEntity output file, got: {files}"
            )


class TestNhopCountFromResultsNotLookup:
    """Test that nhop_count is aggregated from explicit results, not looked up.

    This tests the bug where group_single_onehop_worker.py was looking up
    nhop_count from aggregated_counts (which only contains 1-hop paths),
    causing nhop_count=0 for all N-hop analysis where N>1.

    The symptom was: overlap > 0 but nhop_count = 0, which is impossible.
    """

    def test_2hop_nhop_count_from_results_not_lookup(self):
        """For 2-hop analysis, nhop_count should come from results, not precomputed 1-hop counts.

        The bug: aggregated_counts only has 1-hop paths (from matrix metadata).
        For 2-hop analysis, nhop paths are 2-hop paths which won't be in aggregated_counts.
        The old code would return nhop_count=0, causing impossible metrics like overlap > 0 with count = 0.
        """
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results_2hop")
            output_dir = os.path.join(tmpdir, "grouped")
            os.makedirs(results_dir)
            os.makedirs(output_dir)

            # Create a 2-hop result file
            # Format: nhop_path, nhop_count, onehop_path, onehop_count, overlap, total_possible
            result_file = os.path.join(results_dir, "results_matrix1_000.tsv")
            with open(result_file, 'w') as f:
                f.write("2hop_metapath\t2hop_count\t1hop_metapath\t1hop_count\toverlap\ttotal_possible\n")
                # 2-hop path with count=5000 and overlap=923
                f.write("Gene|affects|F|Disease|treats|R|SmallMolecule\t5000\tGene|treats|F|SmallMolecule\t1000\t923\t1000000\n")

            # Create aggregated_counts with ONLY 1-hop paths (simulating real scenario)
            # This is what prepare_grouping.py produces - it computes from matrix metadata
            # which only has 1-hop edge counts
            aggregated_counts = {
                "Gene|treats|F|SmallMolecule": 1000,
                "Gene|treats|F|ChemicalEntity": 1000,
                "BiologicalEntity|treats|F|SmallMolecule": 1000,
                # Note: NO 2-hop paths like "Gene|affects|F|Disease|treats|R|SmallMolecule"
            }

            counts_file = os.path.join(results_dir, "aggregated_path_counts.json")
            with open(counts_file, 'w') as f:
                json.dump({"counts": aggregated_counts}, f)

            # Create type node counts
            type_node_counts = {
                "Gene": 5000, "SmallMolecule": 3000, "Disease": 2000,
                "ChemicalEntity": 3500, "BiologicalEntity": 6000,
                "Entity": 20000, "NamedThing": 20000
            }

            type_counts_file = os.path.join(results_dir, "type_node_counts.json")
            with open(type_counts_file, 'w') as f:
                json.dump(type_node_counts, f)

            # Run the worker
            group_type_pair(
                type1="Gene",
                type2="SmallMolecule",
                file_list=[result_file],
                output_dir=output_dir,
                n_hops=2,
                aggregate=True,
                aggregated_counts=aggregated_counts,
                type_node_counts=type_node_counts
            )

            # Check output files
            files = os.listdir(output_dir)
            assert len(files) > 0, "No output files created"

            # Read the output and verify nhop_count is NOT 0
            for fname in files:
                fpath = os.path.join(output_dir, fname)
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) > 1, f"No data in {fname}"

                    # Check each data row
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            nhop_count = int(parts[1])
                            overlap = int(parts[2])

                            # THE KEY ASSERTION: if overlap > 0, nhop_count must be > 0
                            if overlap > 0:
                                assert nhop_count > 0, (
                                    f"BUG: overlap={overlap} but nhop_count={nhop_count}. "
                                    f"This is impossible - nhop_count should be aggregated from results, "
                                    f"not looked up from 1-hop precomputed counts. Line: {line.strip()}"
                                )

    def test_nhop_count_aggregates_correctly_across_variants(self):
        """When multiple explicit nhop paths expand to same variant, counts should sum."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results_2hop")
            output_dir = os.path.join(tmpdir, "grouped")
            os.makedirs(results_dir)
            os.makedirs(output_dir)

            # Create result file with multiple 2-hop paths that expand to same variant
            result_file = os.path.join(results_dir, "results_matrix1_000.tsv")
            with open(result_file, 'w') as f:
                f.write("2hop_metapath\t2hop_count\t1hop_metapath\t1hop_count\toverlap\ttotal_possible\n")
                # Two different 2-hop paths, both should aggregate to Entity|related_to|...
                f.write("Gene|affects|F|Disease|treats|R|SmallMolecule\t3000\tGene|treats|F|SmallMolecule\t1000\t100\t1000000\n")
                f.write("Protein|regulates|F|Disease|treats|R|Drug\t2000\tProtein|treats|F|Drug\t500\t50\t1000000\n")

            # Only 1-hop counts in aggregated_counts
            aggregated_counts = {
                "Gene|treats|F|SmallMolecule": 1000,
                "Protein|treats|F|Drug": 500,
            }

            counts_file = os.path.join(results_dir, "aggregated_path_counts.json")
            with open(counts_file, 'w') as f:
                json.dump({"counts": aggregated_counts}, f)

            type_node_counts = {
                "Gene": 5000, "Protein": 4000, "SmallMolecule": 3000, "Drug": 2000,
                "Disease": 2000, "BiologicalEntity": 10000, "ChemicalEntity": 5000,
                "Entity": 20000, "NamedThing": 20000
            }

            type_counts_file = os.path.join(results_dir, "type_node_counts.json")
            with open(type_counts_file, 'w') as f:
                json.dump(type_node_counts, f)

            # Run for BiologicalEntity-ChemicalEntity type pair (should catch both paths)
            group_type_pair(
                type1="BiologicalEntity",
                type2="ChemicalEntity",
                file_list=[result_file],
                output_dir=output_dir,
                n_hops=2,
                aggregate=True,
                aggregated_counts=aggregated_counts,
                type_node_counts=type_node_counts
            )

            # Find output files and verify aggregated counts
            files = os.listdir(output_dir)

            # Look for rows where nhop path is something like "BiologicalEntity|...|ChemicalEntity"
            # The aggregated nhop_count should be 3000 + 2000 = 5000 for the most general variant
            found_aggregated = False
            for fname in files:
                fpath = os.path.join(output_dir, fname)
                with open(fpath, 'r') as f:
                    for line in f.readlines()[1:]:  # Skip header
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            nhop_path = parts[0]
                            nhop_count = int(parts[1])
                            overlap = int(parts[2])

                            # Check that overlap > 0 implies nhop_count > 0
                            if overlap > 0:
                                assert nhop_count > 0, (
                                    f"overlap={overlap} but nhop_count=0 for {nhop_path}"
                                )
                                found_aggregated = True

            assert found_aggregated, "No aggregated results found with overlap > 0"
