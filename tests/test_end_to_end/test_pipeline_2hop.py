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
        # This 2-hop path gets canonicalized: "Disease" < "Gene" alphabetically
        # Original: Gene|regulates|F|Gene|affects|F|Disease
        # Canonical (reversed): Disease|affects|R|Gene|regulates|R|Gene
        path = "Disease|affects|R|Gene|regulates|R|Gene"
        assert path in nhop_counts, f"Expected canonical path not found: {path}"
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
                nhop_path = row.get("predictor_metapath", "")
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
                predictor = row.get("predictor_metapath", "")
                count = row.get("predictor_count", 0)

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
                predictor = row.get("predictor_metapath", "")
                output_count = row.get("predictor_count", 0)

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
        """When overlap > 0, recall should be > 0.

        Note: This test may fail if prepare_grouping.py didn't compute counts for all
        target 1-hop paths. The root cause is when onehop_count=0 but overlap>0,
        which indicates the target path is missing from precomputed counts.
        """
        grouped_results = pipeline_2hop["grouped_results"]

        for filename, rows in grouped_results.items():
            for row in rows:
                overlap = row.get("overlap", 0)
                recall = row.get("recall", 0)

                if overlap > 0:
                    assert recall > 0, (
                        f"overlap={overlap} but recall={recall} in file {filename}. "
                        f"This usually means the target 1-hop path has onehop_count=0. "
                        f"Run prepare_grouping.py to fix precomputed counts."
                    )


class Test2HopTypePairs:
    """Test type pair handling for 2-hop."""

    def test_type_pairs_exclude_pseudo_types(self, pipeline_2hop):
        """Type pairs should not include pseudo-types."""
        type_pairs = pipeline_2hop["type_pairs"]

        for t1, t2 in type_pairs:
            assert '+' not in t1, f"Pseudo-type in type pair: {t1}"
            assert '+' not in t2, f"Pseudo-type in type pair: {t2}"


class TestRawResult2HopOverlaps:
    """Test overlap values in the raw 2-hop analysis output.

    For 2-hop, the predictor is a 2-hop path and the target is a 1-hop path.
    The overlap measures how many (src, tgt) pairs reachable in 2 hops also
    have the target 1-hop edge.

    Key 2-hop paths with known overlaps in the golden graph:

    Gene|regulates|F|Gene|affects|F|Disease:
      Gene_A→Gene_B (regulates), Gene_B→Disease_P (affects)
      Result: {(Gene_A, Disease_P)} — 1 pair
      Target Gene|affects|F|Disease: {(Gene_A, Disease_P), (Gene_A, Disease_Q), (Gene_B, Disease_P)}
      Overlap = 1

    Gene|affects|F|Disease|affects|R|Gene (Gene→Disease→Gene):
      Gene_A→{Disease_P, Disease_Q}, Gene_B→{Disease_P}
      Then reverse: Disease_P→{Gene_A, Gene_B}, Disease_Q→{Gene_A}
      Result: {(Gene_A,Gene_A), (Gene_A,Gene_B), (Gene_B,Gene_A), (Gene_B,Gene_B)} — 4 pairs
      Target Gene|regulates|F|Gene: {(Gene_A, Gene_B)}
      Overlap = 1

    Gene|affects|F|Disease|affects|R|Protein (Gene→Disease→Protein):
      Gene_A→{Disease_P, Disease_Q}, Gene_B→{Disease_P}
      Then Disease_P→Protein_M (reverse of Protein_M→Disease_P)
      Result: {(Gene_A, Protein_M), (Gene_B, Protein_M)} — 2 pairs
      Target Gene|interacts_with|A|Protein: {(Gene_A, Protein_M), (Gene_B, Protein_N)}
      Overlap = 1 (Gene_A, Protein_M)
    """

    def test_raw_results_have_nonzero_overlaps(self, pipeline_2hop):
        """At least some 2-hop paths have overlap > 0 with 1-hop targets."""
        raw_results = pipeline_2hop["raw_results"]

        overlapping = [r for r in raw_results if r['overlap'] > 0]
        assert len(overlapping) > 0, "No 2-hop paths overlap with any 1-hop target"

    def test_zero_overlap_rows_excluded(self, pipeline_2hop):
        """All rows in raw results have overlap > 0 (zero-overlap rows are skipped)."""
        raw_results = pipeline_2hop["raw_results"]

        for row in raw_results:
            assert row['overlap'] > 0, (
                f"Zero overlap in raw results: {row['predictor_path']} vs {row['predicted_path']}"
            )

    def test_overlap_leq_counts(self, pipeline_2hop):
        """overlap <= min(predictor_count, predicted_count) for all rows."""
        raw_results = pipeline_2hop["raw_results"]

        for row in raw_results:
            max_possible = min(row['predictor_count'], row['predicted_count'])
            assert row['overlap'] <= max_possible, (
                f"Overlap {row['overlap']} exceeds min(predictor={row['predictor_count']}, "
                f"predicted={row['predicted_count']}) for "
                f"{row['predictor_path']} vs {row['predicted_path']}"
            )

    def test_gene_regulates_gene_affects_disease_overlap(self, pipeline_2hop):
        """Gene→Gene→Disease via regulates+affects should overlap with Gene→Disease affects.

        2-hop: Gene_A→Gene_B (regulates), Gene_B→{Disease_P, Disease_Q} (affects)
             = {(Gene_A, Disease_P), (Gene_A, Disease_Q)}  predictor_count=2
        1-hop target: Gene→Disease affects (explicit Gene matrix only, 4 pairs)
             predicted_count=4  (pseudo-type GeneProtein_Z not included in raw result)
        Overlap = 2  (Gene_A reaches both Disease_P and Disease_Q via the 2-hop)

        Note: The path may be canonicalized (reversed) in the output.
        """
        raw_results = pipeline_2hop["raw_results"]

        matches = [
            r for r in raw_results
            if 'regulates' in r['predictor_path']
            and 'affects' in r['predictor_path']
            and 'Disease' in r['predictor_path']
            and 'affects' in r['predicted_path']
        ]

        assert len(matches) >= 1, (
            f"Expected to find Gene→Gene→Disease via regulates+affects. "
            f"Available predictor paths: {[r['predictor_path'] for r in raw_results]}"
        )

        for match in matches:
            assert match['overlap'] == 2, (
                f"Expected overlap=2 for {match['predictor_path']} vs {match['predicted_path']}, "
                f"got {match['overlap']}"
            )
            assert match['predictor_count'] == 2, (
                f"Expected predictor_count=2, got {match['predictor_count']}"
            )
            assert match['predicted_count'] == 4, (
                f"Expected predicted_count=4, got {match['predicted_count']}"
            )

    def test_gene_affects_disease_affects_gene_overlap(self, pipeline_2hop):
        """Gene→Disease→Gene via affects should overlap with Gene→Gene regulates.

        2-hop: Gene|affects|F|Disease|affects|R|Gene (palindromic path)
        Raw matrix has 4 pairs: (A,A), (A,B), (B,A), (B,B)
        After triu(k=1) dedup: only (A,B) survives → predictor_count=1

        Target: Gene|regulates|F|Gene = {(Gene_A, Gene_B)}
        Overlap = 1 (the single remaining pair matches)
        """
        raw_results = pipeline_2hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'].count('affects') == 2  # 2 hops both using affects
            and 'regulates' in r['predicted_path']
        ]

        assert len(matches) == 1, (
            "Expected exactly one Gene→Disease→Gene (affects×2) vs Gene→Gene (regulates)"
        )

        match = matches[0]
        assert match['predictor_count'] == 1, (
            f"Expected predictor_count=1 after palindromic dedup (triu), "
            f"got {match['predictor_count']}"
        )
        assert match['overlap'] == 1, (
            f"Expected overlap=1, got {match['overlap']}"
        )

    def test_gene_affects_disease_affects_protein_overlap(self, pipeline_2hop):
        """Gene→Disease→Protein via affects should overlap with Gene→Protein interacts_with.

        2-hop: Gene→Disease (affects), Disease→Protein (affects reverse)
        Disease_P→Protein_M (reverse of Protein_M→Disease_P)
        Gene_A→Disease_P→Protein_M, Gene_B→Disease_P→Protein_M
        Result: {(Gene_A, Protein_M), (Gene_B, Protein_M)}
        Target: Gene|interacts_with|A|Protein = {(Gene_A, Protein_M), (Gene_B, Protein_N)}
        Overlap = 1 (Gene_A, Protein_M in both)
        """
        raw_results = pipeline_2hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'].count('affects') == 2
            and 'Protein' in r['predictor_path']
            and 'Gene' in r['predictor_path']
            and 'interacts_with' in r['predicted_path']
        ]

        assert len(matches) >= 1, (
            "Expected Gene→Disease→Protein (affects×2) vs Gene→Protein (interacts_with)"
        )

        for match in matches:
            assert match['overlap'] == 1, (
                f"Expected overlap=1 for {match['predictor_path']} vs {match['predicted_path']}, "
                f"got {match['overlap']}"
            )

    def test_total_raw_result_count(self, pipeline_2hop):
        """Verify exact number of raw result rows as regression guard.

        The golden graph produces exactly 15 raw 2-hop result rows (with overlap > 0).
        """
        raw_results = pipeline_2hop["raw_results"]
        assert len(raw_results) == 15, (
            f"Expected 15 raw result rows, got {len(raw_results)}. "
            f"Paths: {[r['predictor_path'] for r in raw_results]}"
        )

    def test_symmetric_first_hop_protein_interacts_gene_affects_disease(self, pipeline_2hop):
        """Protein|interacts_with|A|Gene|affects|F|Disease: symmetric predicate as first hop.

        interacts_with is symmetric, so the matrix has both directions:
        Protein_M↔Gene_A, Protein_N↔Gene_B (plus Gene_B↔GeneProtein_Z, different types)
        Then Gene→Disease (affects): Gene_A→{DP,DQ}, Gene_B→{DP,DQ}
        Chain: Protein_M→{DP,DQ}, Protein_N→{DP,DQ} = 4 pairs
        predictor_count=4

        vs Protein|affects|F|Disease: Protein_M→Disease_P = 1 pair
        Overlap: (Protein_M, Disease_P) in both → overlap=1
        """
        raw_results = pipeline_2hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'] == 'Protein|interacts_with|A|Gene|affects|F|Disease'
            and r['predicted_path'] == 'Protein|affects|F|Disease'
        ]

        assert len(matches) == 1, (
            f"Expected Protein|interacts_with|A|Gene|affects|F|Disease vs Protein|affects|F|Disease"
        )
        assert matches[0]['predictor_count'] == 4
        assert matches[0]['predicted_count'] == 1
        assert matches[0]['overlap'] == 1

    def test_two_consecutive_symmetric_predicates(self, pipeline_2hop):
        """Disease|associated_with|A|Protein|interacts_with|A|Gene: two symmetric hops.

        Disease_Q→Protein_N (associated_with, symmetric) → Gene_B (interacts_with, symmetric)
        Result: {(Disease_Q, Gene_B)} — 1 pair
        predictor_count=1

        vs Disease|affects|R|Gene: 4 pairs (reverse of Gene→Disease affects)
        Overlap: (Disease_Q, Gene_B) — Gene_B→Disease_Q is in affects → overlap=1
        """
        raw_results = pipeline_2hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'] == 'Disease|associated_with|A|Protein|interacts_with|A|Gene'
        ]

        assert len(matches) == 1, (
            f"Expected Disease|associated_with|A|Protein|interacts_with|A|Gene"
        )
        assert matches[0]['predictor_count'] == 1
        assert matches[0]['predicted_count'] == 4
        assert matches[0]['overlap'] == 1

    def test_reverse_direction_regulates_gene_affects_disease(self, pipeline_2hop):
        """Gene|regulates|R|Gene|affects|F|Disease: reverse-direction non-symmetric first hop.

        Gene_A→Gene_B (regulates forward), so reverse: Gene_B→Gene_A
        Then Gene_A→{Disease_P, Disease_Q} (affects)
        Result: {(Gene_B, Disease_P), (Gene_B, Disease_Q)} — 2 pairs
        predictor_count=2

        vs Gene|affects|F|Disease: 4 pairs
        Overlap: both (Gene_B, Disease_P) and (Gene_B, Disease_Q) are in affects → overlap=2
        """
        raw_results = pipeline_2hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'] == 'Gene|regulates|R|Gene|affects|F|Disease'
        ]

        assert len(matches) == 1, (
            f"Expected Gene|regulates|R|Gene|affects|F|Disease"
        )
        assert matches[0]['predictor_count'] == 2
        assert matches[0]['predicted_count'] == 4
        assert matches[0]['overlap'] == 2

    def test_pseudo_type_source_gene_protein_affects_disease_affects_gene(self, pipeline_2hop):
        """Gene+Protein|affects|F|Disease|affects|R|Gene: pseudo-type as source.

        GeneProtein_Z→Disease_Q (affects), Disease_Q→{Gene_A, Gene_B} (reverse affects)
        Result: {(GeneProtein_Z, Gene_A), (GeneProtein_Z, Gene_B)} — 2 pairs
        src=Gene+Protein ≠ tgt=Gene, so no same-type dedup applied
        predictor_count=2

        vs Gene+Protein|interacts_with|A|Gene: {(GeneProtein_Z, Gene_B)} — 1 pair
        Overlap: (GeneProtein_Z, Gene_B) in both → overlap=1
        """
        raw_results = pipeline_2hop["raw_results"]

        matches = [
            r for r in raw_results
            if r['predictor_path'] == 'Gene+Protein|affects|F|Disease|affects|R|Gene'
        ]

        assert len(matches) == 1, (
            f"Expected Gene+Protein|affects|F|Disease|affects|R|Gene"
        )
        assert matches[0]['predictor_count'] == 2
        assert matches[0]['predicted_count'] == 1
        assert matches[0]['overlap'] == 1


class TestGrouped2HopOverlapMetrics:
    """Test overlap-derived metrics in grouped 2-hop output."""

    def test_precision_equals_overlap_over_predictor_count(self, pipeline_2hop):
        """Verify precision = overlap / predictor_count for all rows."""
        grouped_results = pipeline_2hop["grouped_results"]

        for filename, rows in grouped_results.items():
            for row in rows:
                predictor_count = row.get('predictor_count', 0)
                overlap = row.get('overlap', 0)
                precision = row.get('precision', 0)

                if predictor_count > 0:
                    expected_precision = overlap / predictor_count
                    assert abs(precision - expected_precision) < 0.0001, (
                        f"precision={precision} != overlap/predictor_count="
                        f"{overlap}/{predictor_count}={expected_precision} "
                        f"in {filename}"
                    )

    def test_f1_consistent_with_precision_recall(self, pipeline_2hop):
        """Verify F1 = 2*P*R/(P+R) for all rows with positive P and R."""
        grouped_results = pipeline_2hop["grouped_results"]

        for filename, rows in grouped_results.items():
            for row in rows:
                precision = row.get('precision', 0)
                recall = row.get('recall', 0)
                f1 = row.get('f1', 0)

                if precision > 0 and recall > 0:
                    expected_f1 = 2 * precision * recall / (precision + recall)
                    assert abs(f1 - expected_f1) < 0.0001, (
                        f"f1={f1} != 2*P*R/(P+R)={expected_f1} "
                        f"(P={precision}, R={recall}) in {filename}"
                    )



class TestGrouped2HopSpecificPaths:
    """Test specific known 2-hop overlap paths in grouped output.

    These tests verify exact values for paths we can manually compute
    from the golden graph structure.
    """

    def test_chemical_affects_gene_affects_disease_predicts_treats(self, pipeline_2hop):
        """ChemicalEntity|affects|F|Gene|affects|F|Disease predicts ChemicalEntity|treats|F|Disease.

        From the golden graph:
          SmallMolecule_X --affects--> Gene_A --affects--> {Disease_P, Disease_Q}
          SmallMolecule_Y --affects--> Gene_B --affects--> {Disease_P, Disease_Q}
        predictor_count = 4 (SM→Gene→Disease pairs)

          SmallMolecule_X --treats--> Disease_P
          SmallMolecule_Y --treats--> Disease_Q
        onehop_count = 2

        Overlap: (SM_X, Disease_P) and (SM_Y, Disease_Q) = 2
        precision = 2/4 = 0.5, recall = 2/2 = 1.0

        MCC=0: TP=2, FP=2, FN=0, TN=0 → denominator (TN+FN)=0 → MCC undefined → 0
        """
        grouped_results = pipeline_2hop["grouped_results"]

        target_file = "ChemicalEntity_treats_F_Disease.tsv.zst"
        assert target_file in grouped_results, (
            f"Expected file {target_file} not found. Available: {sorted(grouped_results.keys())}"
        )

        rows = grouped_results[target_file]
        predictor = "ChemicalEntity|affects|F|Gene|affects|F|Disease"

        matches = [r for r in rows if r.get("predictor_metapath") == predictor]
        assert len(matches) == 1, (
            f"Expected exactly one row for {predictor}, got {len(matches)}"
        )

        row = matches[0]
        assert row["overlap"] == 2, f"Expected overlap=2, got {row['overlap']}"
        assert row["predictor_count"] == 4, f"Expected predictor_count=4, got {row['predictor_count']}"
        assert abs(row["precision"] - 0.5) < 0.0001, f"Expected precision=0.5, got {row['precision']}"
        assert abs(row["recall"] - 1.0) < 0.0001, f"Expected recall=1.0, got {row['recall']}"
        assert row["mcc"] == 0.0, f"Expected mcc=0.0 (TN=0 → denominator=0), got {row['mcc']}"

    def test_biological_affects_biological_affects_biological_predicts_related_to(self, pipeline_2hop):
        """BiologicalEntity|affects|F|BiologicalEntity|affects|F|BiologicalEntity predicts
        BiologicalEntity|related_to|A|BiologicalEntity.

        The aggregated predictor (predictor_count=2) comes from:
        Gene|regulates|F|Gene|affects|F|Disease (count=2), rolling up via
        the predicate hierarchy (regulates→affects) and type hierarchy
        (Gene→BiologicalEntity, Disease→BiologicalEntity).

        Note: Gene|affects|F|Disease|affects|R|Gene does NOT contribute here
        because its directions are F,R — it rolls up to affects|F|...|affects|R,
        not affects|F|...|affects|F.

        With 7 BiologicalEntity nodes (Gene_A, Gene_B, GeneProtein_Z, Protein_M,
        Protein_N, Disease_P, Disease_Q), total_possible = 7×7 = 49.

        overlap=2, predictor_count=2 → precision=1.0
        onehop_count=15: sum of all explicit 1-hop paths that roll up to
        BiologicalEntity|related_to|A|BiologicalEntity from the raw results:
        Gene|regulates|F|Gene(1) + Protein|interacts_with|A|Gene(2) +
        Disease|affects|R|Gene(4) + Gene+Protein|interacts_with|A|Gene(1) +
        Protein|affects|F|Disease(1) + Protein|associated_with|A|Disease(1) +
        Gene+Protein|affects|F|Disease(1) + Gene|affects|F|Disease(4) = 15
        recall = 2/15 ≈ 0.1333
        """
        grouped_results = pipeline_2hop["grouped_results"]

        target_file = "BiologicalEntity_related_to_A_BiologicalEntity.tsv.zst"
        assert target_file in grouped_results, (
            f"Expected file {target_file} not found. Available: {sorted(grouped_results.keys())}"
        )

        rows = grouped_results[target_file]
        predictor = "BiologicalEntity|affects|F|BiologicalEntity|affects|F|BiologicalEntity"

        matches = [r for r in rows if r.get("predictor_metapath") == predictor]
        assert len(matches) == 1, (
            f"Expected exactly one row for {predictor}, got {len(matches)}"
        )

        row = matches[0]
        assert row["overlap"] == 2, f"Expected overlap=2, got {row['overlap']}"
        assert row["predictor_count"] == 2, f"Expected predictor_count=2, got {row['predictor_count']}"
        assert abs(row["precision"] - 1.0) < 0.0001, f"Expected precision=1.0, got {row['precision']}"
        expected_recall = 2 / 15
        assert abs(row["recall"] - expected_recall) < 0.0001, (
            f"Expected recall={expected_recall:.6f} (2/15), got {row['recall']}"
        )
        assert row["mcc"] > 0, f"Expected mcc > 0, got {row['mcc']}"

    def test_gene_product_affects_biological_affects_gene_product_predicts_interacts(self, pipeline_2hop):
        """GeneOrGeneProduct|affects|F|BiologicalEntity|affects|R|GeneOrGeneProduct predicts
        GeneOrGeneProduct|interacts_with|A|GeneOrGeneProduct.

        The aggregated predictor sums three explicit 2-hop paths:
        1. Gene|affects|F|Disease|affects|R|Gene (palindromic: count=1 after triu dedup)
        2. Protein|affects|F|Disease|affects|R|Gene (count=2, different src/tgt types)
        3. Gene+Protein|affects|F|Disease|affects|R|Gene (count=2, different src/tgt types)
        Total predictor_count = 1 + 2 + 2 = 5

        overlap=3: one overlapping pair from each explicit path
        predicted_count=4: sum of 1-hop paths rolling up to interacts_with
        """
        grouped_results = pipeline_2hop["grouped_results"]

        target_file = "GeneOrGeneProduct_interacts_with_A_GeneOrGeneProduct.tsv.zst"
        assert target_file in grouped_results, (
            f"Expected file {target_file} not found. Available: {sorted(grouped_results.keys())}"
        )

        rows = grouped_results[target_file]
        predictor = "GeneOrGeneProduct|affects|F|BiologicalEntity|affects|R|GeneOrGeneProduct"

        matches = [r for r in rows if r.get("predictor_metapath") == predictor]
        assert len(matches) == 1, (
            f"Expected exactly one row for {predictor}, got {len(matches)}"
        )

        row = matches[0]

        # Check predictor count, predicted count, and overlap independently
        assert row["predictor_count"] == 5, f"Expected predictor_count=5, got {row['predictor_count']}"
        assert row["overlap"] == 3, f"Expected overlap=3, got {row['overlap']}"

        predicted_path = "GeneOrGeneProduct|interacts_with|A|GeneOrGeneProduct"
        predicted_count = pipeline_2hop["aggregated_nhop_counts"].get(predicted_path)
        assert predicted_count == 4, f"Expected predicted_count=4, got {predicted_count}"
