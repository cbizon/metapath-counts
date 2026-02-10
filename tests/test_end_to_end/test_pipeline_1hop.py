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

    def test_interacts_with_symmetric_count(self, pipeline_1hop):
        """Gene|interacts_with|A|Protein should have count=3 (2 explicit + 1 from pseudo-type).

        The key insight: For symmetric predicates, pseudo-type expansion can flip direction!
        - Gene_A ↔ Protein_M (explicit Gene ↔ Protein)
        - Gene_B ↔ Protein_N (explicit Gene ↔ Protein)
        - GeneProtein_Z ↔ Gene_B: When Gene+Protein expands to Protein:
          Protein ↔ Gene = Gene ↔ Protein (symmetric!)
        """
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # Alphabetically: "Gene" < "Protein", so canonical is forward direction
        # For symmetric predicates, direction should be 'A', not 'F' or 'R'
        canonical = "Gene|interacts_with|A|Protein"
        wrong_direction_f = "Gene|interacts_with|F|Protein"
        wrong_direction_r = "Gene|interacts_with|R|Protein"
        reverse_canonical = "Protein|interacts_with|A|Gene"

        count = nhop_counts.get(canonical, 0)
        assert count == 3, f"Expected count=3 (2 explicit + 1 pseudo-type) for {canonical}, got {count}"

        # Wrong directions (F or R) should not exist for symmetric predicates
        assert nhop_counts.get(wrong_direction_f, 0) == 0, (
            f"Symmetric predicate should not use F direction: {wrong_direction_f}"
        )
        assert nhop_counts.get(wrong_direction_r, 0) == 0, (
            f"Symmetric predicate should not use R direction: {wrong_direction_r}"
        )

        # Reverse direction should not exist due to duplicate elimination
        assert nhop_counts.get(reverse_canonical, 0) == 0, (
            f"Non-canonical direction {reverse_canonical} should not exist"
        )


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

    def test_biological_entity_affects_biological_entity(self, pipeline_1hop):
        """BiologicalEntity|affects|F|BiologicalEntity aggregation test.

        When different-type paths (e.g. Disease|affects|R|Gene) expand to same-type
        variants (BiologicalEntity|affects|*|BiologicalEntity), both F and R directions
        must be generated. The 5 affects edges (Gene→Disease x3, Protein→Disease x1,
        Gene+Protein→Disease x1) plus pseudo-type expansion to Protein (+1) give 6 total,
        contributed to both F and R directions.
        """
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        forward = "BiologicalEntity|affects|F|BiologicalEntity"
        reverse = "BiologicalEntity|affects|R|BiologicalEntity"

        forward_count = nhop_counts.get(forward, 0)
        reverse_count = nhop_counts.get(reverse, 0)

        assert forward_count == 6, f"Expected 6 for {forward}, got {forward_count}"
        assert reverse_count == 6, f"Expected 6 for {reverse}, got {reverse_count}"

    def test_named_thing_related_to_aggregation(self, pipeline_1hop):
        """NamedThing|related_to|A|NamedThing should aggregate all edges (related_to is symmetric root predicate).

        All edges roll up to NamedThing (top-level type) and related_to (top-level symmetric predicate):
        - affects: 5 edges
        - treats: 2 edges
        - interacts_with: 3 edges
        - regulates: 1 edge
        - associated_with: 1 edge
        Total: 12 edges
        """
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]
        # NamedThing == NamedThing (same type), symmetric predicate uses 'A'
        canonical = "NamedThing|related_to|A|NamedThing"

        count = nhop_counts.get(canonical, 0)
        # All 12 edges in the graph roll up to this
        assert count == 12, f"Expected 12 (all edges) for {canonical}, got {count}"

    def test_biological_entity_related_to_chemical_entity(self, pipeline_1hop):
        """BiologicalEntity|related_to|A|ChemicalEntity should aggregate treats edges."""
        nhop_counts = pipeline_1hop["aggregated_nhop_counts"]

        # related_to is symmetric, so direction is 'A'
        # BiologicalEntity < ChemicalEntity alphabetically, but symmetric uses canonical form
        path = "BiologicalEntity|related_to|A|ChemicalEntity"

        count = nhop_counts.get(path, 0)

        # Expected: 2 treats edges (SmallMolecule→Disease) aggregate to this path
        # Disease|treats|R|SmallMolecule expands to:
        # - Disease → BiologicalEntity (type hierarchy)
        # - SmallMolecule → ChemicalEntity (type hierarchy)
        # - treats → related_to (predicate hierarchy, becomes 'A' for symmetric)
        expected_count = 2

        assert count == expected_count, (
            f"Expected count={expected_count} for {path}, got {count}. "
            f"Should aggregate treats edges between ChemicalEntity and BiologicalEntity types."
        )


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
                nhop_path = row.get("predictor_metapath", "")
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
                nhop_path = row.get("predictor_metapath", "")
                precision = row.get("precision", 0)

                # Only check paths without "related_to" (explicit predicates)
                if "related_to" not in nhop_path:
                    assert precision <= 1.001, (
                        f"Precision > 1.0 for explicit path: {nhop_path} has {precision}"
                    )


class TestRawResultOverlaps:
    """Test overlap values in the raw analysis output.

    For 1-hop analysis, each predictor matrix is compared with all 1-hop
    target matrices of the same (src, tgt) type pair, plus an aggregated
    ANY target. Since the golden graph has at most one predicate per
    type pair direction, all non-ANY overlaps are self-comparisons.
    """

    def test_all_overlaps_positive(self, pipeline_1hop):
        """Every row in raw results has overlap > 0 (zero-overlap rows are skipped)."""
        raw_results = pipeline_1hop["raw_results"]

        for row in raw_results:
            assert row['overlap'] > 0, (
                f"Zero overlap in raw results: {row['predictor_path']} vs {row['predicted_path']}"
            )

    def test_self_comparison_overlap_equals_count(self, pipeline_1hop):
        """When predictor_path == predicted_path, overlap == predictor_count."""
        raw_results = pipeline_1hop["raw_results"]

        self_comparisons = [
            r for r in raw_results
            if r['predictor_path'] == r['predicted_path']
        ]

        assert len(self_comparisons) > 0, "No self-comparisons found"

        for row in self_comparisons:
            assert row['overlap'] == row['predictor_count'], (
                f"Self-comparison overlap mismatch: {row['predictor_path']} "
                f"has overlap={row['overlap']} but count={row['predictor_count']}"
            )

    def test_overlap_leq_min_counts(self, pipeline_1hop):
        """Overlap <= min(predictor_count, predicted_count) for all rows."""
        raw_results = pipeline_1hop["raw_results"]

        for row in raw_results:
            max_possible = min(row['predictor_count'], row['predicted_count'])
            assert row['overlap'] <= max_possible, (
                f"Overlap {row['overlap']} exceeds min(predictor={row['predictor_count']}, "
                f"predicted={row['predicted_count']}) for {row['predictor_path']} vs {row['predicted_path']}"
            )

    def test_regulates_forward_vs_any_overlap(self, pipeline_1hop):
        """Gene|regulates|F|Gene vs Gene|ANY|A|Gene: overlap=1 < predicted_count=2.

        The ANY matrix has both (A,B) and (B,A) from forward and reverse regulates.
        The forward predictor only has (A,B), so overlap is 1 out of 2.
        """
        raw_results = pipeline_1hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'] == 'Gene|regulates|F|Gene'
            and r['predicted_path'] == 'Gene|ANY|A|Gene'
        ]

        assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"
        row = matches[0]
        assert row['overlap'] == 1, f"Expected overlap=1, got {row['overlap']}"
        assert row['predicted_count'] == 2, f"Expected predicted_count=2, got {row['predicted_count']}"

    def test_regulates_reverse_vs_any_overlap(self, pipeline_1hop):
        """Gene|regulates|R|Gene vs Gene|ANY|A|Gene: overlap=1 < predicted_count=2.

        Same as forward case - reverse has (B,A) which is 1 of 2 pairs in ANY.
        """
        raw_results = pipeline_1hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'] == 'Gene|regulates|R|Gene'
            and r['predicted_path'] == 'Gene|ANY|A|Gene'
        ]

        assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"
        row = matches[0]
        assert row['overlap'] == 1, f"Expected overlap=1, got {row['overlap']}"
        assert row['predicted_count'] == 2, f"Expected predicted_count=2, got {row['predicted_count']}"

    def test_no_cross_predicate_overlap_rows(self, pipeline_1hop):
        """No raw result rows where predictor and predicted have different predicates.

        In the golden graph, different predicates between the same type pair have
        different node pairs, so cross-predicate overlap would be 0 (and excluded).
        The only "cross-predicate" rows are the ANY aggregated targets.
        """
        raw_results = pipeline_1hop["raw_results"]

        for row in raw_results:
            pred_parts = row['predictor_path'].split('|')
            tgt_parts = row['predicted_path'].split('|')

            if 'ANY' in tgt_parts:
                continue  # ANY targets are expected

            # For non-ANY targets, predictor and predicted should have the same predicate
            pred_predicate = pred_parts[1] if len(pred_parts) >= 4 else None
            tgt_predicate = tgt_parts[1] if len(tgt_parts) >= 4 else None

            assert pred_predicate == tgt_predicate, (
                f"Cross-predicate row found: {row['predictor_path']} vs {row['predicted_path']}"
            )

    def test_disease_affects_gene_self_overlap(self, pipeline_1hop):
        """Disease|affects|R|Gene self-comparison: overlap=3 (all 3 affects edges)."""
        raw_results = pipeline_1hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'] == 'Disease|affects|R|Gene'
            and r['predicted_path'] == 'Disease|affects|R|Gene'
        ]

        assert len(matches) == 1
        assert matches[0]['overlap'] == 3
        assert matches[0]['predictor_count'] == 3

    def test_interacts_with_self_overlap(self, pipeline_1hop):
        """Gene|interacts_with|A|Protein self-comparison: overlap=2."""
        raw_results = pipeline_1hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'] == 'Gene|interacts_with|A|Protein'
            and r['predicted_path'] == 'Gene|interacts_with|A|Protein'
        ]

        assert len(matches) == 1
        assert matches[0]['overlap'] == 2
        assert matches[0]['predictor_count'] == 2


class TestGroupedOverlapPrecision:
    """Test that grouped output overlap and metrics reflect hierarchical aggregation.

    The interesting cases are when broad aggregated predictors (like
    NamedThing|related_to|A|NamedThing) are evaluated against narrow targets.
    The precision should be < 1.0 because the predictor covers many more pairs
    than the specific target.
    """

    def test_named_thing_related_to_predicting_affects(self, pipeline_1hop):
        """NamedThing|related_to|A|NamedThing predicting Disease|affects|R|Gene.

        The broad predictor covers all edges in the graph. 4 of them are
        Gene-type→Disease affects edges: 3 from Gene + 1 from Gene+Protein
        (which rolls up to Gene during aggregation). So overlap=4 and
        precision = 4/nhop_count < 1.0.
        """
        grouped_results = pipeline_1hop["grouped_results"]

        # Use exact filename - loose matching picks wrong files
        target_filename = "Disease_affects_R_Gene.tsv.zst"

        assert target_filename in grouped_results, (
            f"No grouped file found for Disease|affects|R|Gene. "
            f"Available: {sorted(grouped_results.keys())}"
        )

        rows = grouped_results[target_filename]
        nt_rows = [
            r for r in rows
            if r.get('predictor_metapath') == 'NamedThing|related_to|A|NamedThing'
        ]

        assert len(nt_rows) == 1, (
            f"Expected 1 row for NamedThing|related_to|A|NamedThing predictor, "
            f"got {len(nt_rows)}"
        )

        row = nt_rows[0]
        # 4 edges: Gene_A→Disease_P, Gene_A→Disease_Q, Gene_B→Disease_P (Gene|affects),
        # plus GeneProtein_Z→Disease_Q (Gene+Protein|affects, rolls up to Gene)
        assert row['overlap'] == 4, f"Expected overlap=4, got {row['overlap']}"
        assert row['predictor_count'] > 4, (
            f"NamedThing|related_to|A|NamedThing should have count > 4, got {row['predictor_count']}"
        )
        assert row['precision'] < 1.0, (
            f"Precision should be < 1.0 for broad predictor, got {row['precision']}"
        )

    def test_self_comparison_precision_is_one(self, pipeline_1hop):
        """When predictor variant matches the target, precision should be 1.0.

        For example, Disease|affects|R|Gene predicting Disease|affects|R|Gene.
        """
        grouped_results = pipeline_1hop["grouped_results"]

        # Find a file with self-comparison row
        found_self = False
        for filename, rows in grouped_results.items():
            # Derive the target path from the filename
            # Filename format: Type1_pred_dir_Type2.tsv.zst
            for row in rows:
                predictor = row.get('predictor_metapath', '')
                # Check if predictor matches what the filename encodes
                # Self-comparison = predictor that matches the target pattern
                safe_pred = predictor.replace('|', '_').replace(':', '_').replace(' ', '_')
                expected_filename = f"{safe_pred}.tsv.zst"

                if expected_filename == filename:
                    found_self = True
                    assert abs(row['precision'] - 1.0) < 0.001, (
                        f"Self-comparison precision should be 1.0, got {row['precision']} "
                        f"for {predictor} in {filename}"
                    )

        assert found_self, "No self-comparison rows found in grouped output"

    def test_narrow_predictor_low_recall_for_broad_target(self, pipeline_1hop):
        """A narrow predictor predicting a broad target should have recall < 1.0.

        Gene|regulates|F|Gene predicting BiologicalEntity|affects|F|BiologicalEntity.
        The regulates edge is just 1 pair, but the aggregated target has many more.
        """
        grouped_results = pipeline_1hop["grouped_results"]

        # Find BiologicalEntity|affects|F|BiologicalEntity target file
        target_filename = None
        for filename in grouped_results:
            if ('BiologicalEntity_affects_F_BiologicalEntity' in filename
                    or 'BiologicalEntity_affects_R_BiologicalEntity' in filename):
                target_filename = filename
                break

        if target_filename is None:
            pytest.skip("BiologicalEntity|affects|*|BiologicalEntity target not found")

        rows = grouped_results[target_filename]
        reg_rows = [
            r for r in rows
            if r.get('predictor_metapath') == 'Gene|regulates|F|Gene'
        ]

        if len(reg_rows) == 0:
            pytest.skip("Gene|regulates|F|Gene not found as predictor")

        row = reg_rows[0]
        assert row['overlap'] >= 1, f"Expected overlap >= 1, got {row['overlap']}"
        assert row['recall'] < 1.0, (
            f"Narrow predictor should have recall < 1.0 for broad target, got {row['recall']}"
        )
