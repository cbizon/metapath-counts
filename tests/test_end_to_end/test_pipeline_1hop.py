"""End-to-end tests for the 1-hop analysis pipeline.

These tests verify the full pipeline produces correct results for the golden graph.
"""

import pytest
from pathlib import Path

from conftest import parse_raw_results
from golden_graph import GRAPH_STATS


class TestGoldenGraphSetup:
    """Verify the golden graph is set up correctly."""

    def test_matrices_created(self, golden_workspace):
        """Verify matrices were created."""
        matrices = golden_workspace["matrices"]
        assert len(matrices) > 0

    def test_expected_node_count(self, golden_workspace):
        """Verify correct number of nodes."""
        node_types = golden_workspace["node_types"]
        assert len(node_types) == GRAPH_STATS["num_nodes"]

    def test_pseudo_type_node_exists(self, golden_workspace):
        """Verify the pseudo-type node (Gene+Protein) was created."""
        node_types = golden_workspace["node_types"]

        # Find nodes with pseudo-types (contain '+')
        pseudo_type_nodes = [
            node_id for node_id, type_name in node_types.items()
            if '+' in type_name
        ]

        assert len(pseudo_type_nodes) == GRAPH_STATS["num_pseudo_type_nodes"]

        # The pseudo-type should be Gene+Protein (alphabetically sorted)
        pseudo_type = node_types[pseudo_type_nodes[0]]
        assert pseudo_type == "Gene+Protein"


class TestExplicit1HopCounts:
    """Test that explicit 1-hop path counts match expected values.

    Note: Due to duplicate elimination, 1-hop paths may be stored in reverse
    direction. The counts should still be correct.
    """

    def test_affects_disease_count(self, pipeline_1hop):
        """Gene affects Disease aggregated count should be 4 (3 explicit + 1 pseudo-type)."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "Disease" < "Gene", so canonical is reverse direction
        canonical = "Disease|affects|R|Gene"
        wrong_direction = "Gene|affects|F|Disease"

        count = nhop_counts.get(canonical, 0)
        assert count == 4, (
            f"Expected count=4 (3 explicit + 1 pseudo-type) for {canonical}, got {count}"
        )
        # Wrong direction should not exist
        assert nhop_counts.get(wrong_direction, 0) == 0, (
            f"Non-canonical direction {wrong_direction} should not exist"
        )

    def test_pseudo_type_affects_count(self, pipeline_1hop):
        """Gene+Protein|affects|F|Disease should have count=1."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "Disease" < "Gene+Protein", so canonical is reverse direction
        canonical = "Disease|affects|R|Gene+Protein"
        wrong_direction = "Gene+Protein|affects|F|Disease"

        count = nhop_counts.get(canonical, 0)
        assert count == 1, f"Expected count=1 for {canonical}, got {count}"
        # Wrong direction should not exist
        assert nhop_counts.get(wrong_direction, 0) == 0, (
            f"Non-canonical direction {wrong_direction} should not exist"
        )

    def test_protein_affects_count(self, pipeline_1hop):
        """Protein affects Disease aggregated count should be 2 (1 explicit + 1 pseudo-type)."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "Disease" < "Protein", so canonical is reverse direction
        canonical = "Disease|affects|R|Protein"
        wrong_direction = "Protein|affects|F|Disease"

        count = nhop_counts.get(canonical, 0)
        assert count == 2, (
            f"Expected count=2 (1 explicit + 1 pseudo-type) for {canonical}, got {count}"
        )
        # Wrong direction should not exist
        assert nhop_counts.get(wrong_direction, 0) == 0, (
            f"Non-canonical direction {wrong_direction} should not exist"
        )

    def test_treats_count(self, pipeline_1hop):
        """SmallMolecule|treats|F|Disease should have count=2."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "Disease" < "SmallMolecule", so canonical is reverse direction
        canonical = "Disease|treats|R|SmallMolecule"
        wrong_direction = "SmallMolecule|treats|F|Disease"

        count = nhop_counts.get(canonical, 0)
        assert count == 2, f"Expected count=2 for {canonical}, got {count}"
        # Wrong direction should not exist
        assert nhop_counts.get(wrong_direction, 0) == 0, (
            f"Non-canonical direction {wrong_direction} should not exist"
        )

    def test_gene_regulates_gene_count(self, pipeline_1hop):
        """Gene|regulates|Gene should have count=1 in both directions."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # For same-type edges: "Gene" == "Gene", both directions are computed
        # Forward: Gene_A -> Gene_B
        forward = "Gene|regulates|F|Gene"
        # Reverse: Gene_B <- Gene_A (equivalently, looking for Gene_X -> Gene_Y in reverse)
        reverse = "Gene|regulates|R|Gene"

        forward_count = nhop_counts.get(forward, 0)
        reverse_count = nhop_counts.get(reverse, 0)
        assert forward_count == 1, f"Expected count=1 for {forward}, got {forward_count}"
        assert reverse_count == 1, f"Expected count=1 for {reverse}, got {reverse_count}"


class TestAggregated1HopCounts:
    """Test that hierarchical aggregation produces correct counts.

    Note: Paths may be in forward or reverse direction due to duplicate elimination.
    Aggregated counts roll up to ancestor types.
    """

    def test_pseudo_type_contributes_to_gene(self, pipeline_1hop):
        """Gene aggregated affects count should include pseudo-type contribution."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "Disease" < "Gene", so canonical is reverse direction
        canonical = "Disease|affects|R|Gene"

        count = nhop_counts.get(canonical, 0)
        assert count == 4, f"Expected 4 (3 explicit + 1 pseudo-type) for {canonical}, got {count}"

    def test_pseudo_type_contributes_to_protein(self, pipeline_1hop):
        """Protein aggregated affects count should include pseudo-type contribution."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "Disease" < "Protein", so canonical is reverse direction
        canonical = "Disease|affects|R|Protein"

        count = nhop_counts.get(canonical, 0)
        assert count == 2, f"Expected 2 (1 explicit + 1 pseudo-type) for {canonical}, got {count}"

    def test_biological_entity_aggregation(self, pipeline_1hop):
        """BiologicalEntity aggregated should sum all biological entity affects."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "BiologicalEntity" < "Disease" (B < D), so canonical is forward direction
        canonical = "BiologicalEntity|affects|F|Disease"

        count = nhop_counts.get(canonical, 0)
        # Gene(3) + Protein(1) + Gene+Protein(1) = 5
        assert count == 5, f"Expected 5 for {canonical}, got {count}"

    def test_chemical_entity_aggregation(self, pipeline_1hop):
        """ChemicalEntity aggregated treats count should equal SmallMolecule count."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "ChemicalEntity" < "Disease" (C < D), so canonical is forward direction
        # (Re-canonicalized from the explicit Disease|treats|R|SmallMolecule)
        canonical = "ChemicalEntity|treats|F|Disease"

        count = nhop_counts.get(canonical, 0)
        # SmallMolecule is only child of ChemicalEntity in our graph
        assert count == 2, f"Expected 2 for {canonical}, got {count}"


class TestNoPseudoTypesInOutput:
    """Verify pseudo-types are filtered from grouped output."""

    def test_no_pseudo_type_in_target_filenames(self, pipeline_1hop):
        """Output filenames should not contain pseudo-types."""
        grouped_results = pipeline_1hop["grouped_results"]

        for filename in grouped_results.keys():
            assert '+' not in filename, f"Pseudo-type in filename: {filename}"

    def test_no_pseudo_type_in_predictor_paths(self, pipeline_1hop):
        """Predictor (N-hop) paths in output should not contain pseudo-types."""
        grouped_results = pipeline_1hop["grouped_results"]

        for filename, rows in grouped_results.items():
            for row in rows:
                nhop_path = row.get("1hop_metapath", "")
                assert '+' not in nhop_path, (
                    f"Pseudo-type in predictor path: {nhop_path} (file: {filename})"
                )


class TestTypeNodeCounts:
    """Test that type node counts are computed and aggregated.

    Note: Type node counts come from matrix dimensions (nrows/ncols), not actual
    unique node counts. When a node appears as both src and tgt in different
    matrices, it may be counted multiple times. The hierarchical aggregation
    sums child types to parent types.
    """

    def test_explicit_types_have_counts(self, pipeline_1hop):
        """Verify explicit types have positive counts."""
        type_counts = pipeline_1hop["type_node_counts"]

        # All explicit types should have counts > 0
        assert type_counts.get("Gene", 0) > 0
        assert type_counts.get("Protein", 0) > 0
        assert type_counts.get("Disease", 0) > 0
        assert type_counts.get("SmallMolecule", 0) > 0

    def test_pseudo_type_has_count(self, pipeline_1hop):
        """Verify pseudo-type Gene+Protein has a count."""
        type_counts = pipeline_1hop["type_node_counts"]
        assert type_counts.get("Gene+Protein", 0) > 0

    def test_hierarchical_aggregation(self, pipeline_1hop):
        """Verify hierarchical types have >= sum of child types."""
        type_counts = pipeline_1hop["type_node_counts"]

        # BiologicalEntity should include Gene, Protein, Gene+Protein
        bio_count = type_counts.get("BiologicalEntity", 0)
        gene_count = type_counts.get("Gene", 0)
        protein_count = type_counts.get("Protein", 0)

        assert bio_count >= gene_count, "BiologicalEntity should >= Gene count"
        assert bio_count >= protein_count, "BiologicalEntity should >= Protein count"

        # NamedThing should be largest
        named_thing = type_counts.get("NamedThing", 0)
        assert named_thing >= bio_count, "NamedThing should >= BiologicalEntity"


class TestGroupedOutputCreated:
    """Test that grouped output files are created."""

    def test_grouped_files_created(self, pipeline_1hop):
        """Verify grouped output files were created."""
        grouped_results = pipeline_1hop["grouped_results"]
        assert len(grouped_results) > 0, "No grouped output files created"

    def test_files_are_zstd_compressed(self, pipeline_1hop):
        """Verify output files have .tsv.zst extension."""
        grouped_results = pipeline_1hop["grouped_results"]

        for filename in grouped_results.keys():
            assert filename.endswith(".tsv.zst"), f"File not zstd compressed: {filename}"


class TestTypePairsExcludePseudoTypes:
    """Test that type pairs don't include pseudo-types."""

    def test_type_pairs_exclude_pseudo_types(self, pipeline_1hop):
        """Type pairs should not include pseudo-types like Gene+Protein."""
        type_pairs = pipeline_1hop["type_pairs"]

        type_set = set()
        for t1, t2 in type_pairs:
            type_set.add(t1)
            type_set.add(t2)

        # Should have regular types
        assert "Gene" in type_set
        assert "Protein" in type_set
        assert "Disease" in type_set

        # Should NOT have pseudo-types
        assert "Gene+Protein" not in type_set, "Pseudo-type found in type pairs"


class TestSymmetricPredicateDirections:
    """Test that symmetric predicates get direction 'A' in aggregated paths."""

    def test_related_to_has_direction_A_in_aggregated_counts(self, pipeline_1hop):
        """related_to is symmetric - all aggregated paths should use direction 'A'.

        Note: We check for exact predicate match (|related_to|) to avoid matching
        predicates that contain 'related_to' as a substring (e.g., related_to_at_instance_level).
        """
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]

        for path in nhop_counts.keys():
            # Check for exact 'related_to' predicate (not substrings like related_to_at_instance_level)
            if "|related_to|" in path:
                # Should have 'A' direction
                assert "|related_to|A|" in path, (
                    f"Symmetric predicate 'related_to' should have A direction in: {path}"
                )

    def test_no_symmetric_predicates_with_F_or_R(self, pipeline_1hop):
        """Symmetric predicates should never have F or R direction."""
        from metapath_counts import get_symmetric_predicates

        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        symmetric_preds = get_symmetric_predicates()

        for path in nhop_counts.keys():
            parts = path.split('|')
            # Extract predicates (positions 1, 4, 7, ...)
            for i in range(1, len(parts), 3):
                pred = parts[i]
                if pred in symmetric_preds and i + 1 < len(parts):
                    direction = parts[i + 1]
                    assert direction == 'A', (
                        f"Symmetric predicate '{pred}' has direction '{direction}' in: {path}"
                    )

    def test_interacts_with_has_direction_A(self, pipeline_1hop):
        """interacts_with is symmetric - should use direction 'A'."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]

        for path in nhop_counts.keys():
            if "|interacts_with|" in path:
                # Should have 'A' direction
                assert "|interacts_with|A|" in path, (
                    f"Symmetric predicate 'interacts_with' should have A direction: {path}"
                )


class TestMetricsCalculation:
    """Test that metrics are calculated correctly."""

    def test_precision_not_exceeds_one_for_explicit(self, pipeline_1hop):
        """For explicit paths, precision should not exceed 1.0."""
        grouped_results = pipeline_1hop["grouped_results"]

        for filename, rows in grouped_results.items():
            for row in rows:
                nhop_path = row.get("1hop_metapath", "")
                precision = row.get("precision", 0)

                # Only check paths without "related_to" (explicit predicates)
                if "related_to" not in nhop_path:
                    assert precision <= 1.001, (
                        f"Precision > 1.0 for explicit path: {nhop_path} has {precision}"
                    )
