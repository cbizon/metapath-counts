#!/usr/bin/env python3

import json
import os
import tempfile
import time

import numpy as np
import zstandard

from library.aggregation import (
    expand_metapath_to_typepair_variants,
    traverse_metapath_variants_for_typepair_pruned,
)
from pipeline.workers.run_grouping import (
    build_target_variant_counts,
    compute_exact_target_pair_counts,
    compute_direct_metrics,
    group_type_pair,
)


def test_build_target_variant_counts_from_explicit_typepair_shard():
    explicit_items = [
        ("SmallMolecule|treats|F|Disease", 2),
        ("SmallMolecule|affects|F|Gene|treats|F|Disease", 5),
    ]

    target_counts, _ = build_target_variant_counts(
        explicit_items,
        type1="ChemicalEntity",
        type2="Disease",
    )

    assert target_counts["ChemicalEntity|treats|F|Disease"] == 2


def test_pruned_variant_walk_cuts_off_broader_states():
    visited = []

    def visit(variant):
        visited.append(variant)
        return variant == "ChemicalEntity|treats|F|Disease"

    traverse_metapath_variants_for_typepair_pruned(
        "SmallMolecule|treats|F|Disease",
        type1="ChemicalEntity",
        type2="Disease",
        visit_variant=visit,
    )

    assert "ChemicalEntity|treats|F|Disease" in visited
    assert visited.count("ChemicalEntity|treats|F|Disease") == 1
    assert "NamedThing|treats|F|Disease" not in visited


def test_pruned_state_walk_suppresses_descendant_variants():
    visited = []

    def visit_state(_state_key):
        return True

    def visit_variant(variant):
        visited.append(variant)
        return False

    traverse_metapath_variants_for_typepair_pruned(
        "SmallMolecule|treats|F|Disease",
        type1="ChemicalEntity",
        type2="Disease",
        visit_variant=visit_variant,
        visit_state=visit_state,
    )

    assert visited == []


def test_typepair_expansion_promotes_endpoints_directly_to_job_pair():
    variants = expand_metapath_to_typepair_variants(
        "SmallMolecule|treats|F|Disease",
        type1="ChemicalEntity",
        type2="Disease",
    )

    assert "ChemicalEntity|treats|F|Disease" in variants
    assert all(v.startswith("ChemicalEntity|") for v in variants)
    assert all(v.endswith("|Disease") for v in variants)
    assert all("SmallMolecule|" not in v for v in variants)


def test_typepair_expansion_emits_both_valid_endpoint_assignments():
    variants = expand_metapath_to_typepair_variants(
        "SmallMolecule|treats|F|Drug",
        type1="NamedThing",
        type2="ChemicalEntity",
    )

    assert "ChemicalEntity|treats|F|NamedThing" in variants
    assert "ChemicalEntity|treats|R|NamedThing" in variants
    assert all(v.startswith("ChemicalEntity|") for v in variants)
    assert all(v.endswith("|NamedThing") for v in variants)


def _save_npz(directory, filename, rows, cols, nrows, ncols):
    """Save a boolean sparse matrix as NPZ (matching prebuild_matrices format)."""
    np.savez_compressed(
        os.path.join(directory, filename),
        rows=np.array(rows, dtype=np.uint64),
        cols=np.array(cols, dtype=np.uint64),
        vals=np.ones(len(rows), dtype=bool),
        nrows=nrows,
        ncols=ncols,
        nvals=len(rows),
    )


def _build_test_matrices_dir(tmpdir):
    """Create a temp matrices directory with overlapping predicate matrices.

    Returns (matrices_dir, manifest_matrices) where:
    - SmallMolecule|treats|Disease: pairs (0,0), (1,1), (2,2) -> 3 pairs
    - SmallMolecule|ameliorates|Disease: pairs (1,1), (2,2), (3,3) -> 3 pairs
      (shares pairs (1,1) and (2,2) with treats)
    - SmallMolecule|affects|Gene: pairs (0,0), (1,1), (2,2), (3,3) -> 4 pairs
    - Gene|treats|Disease: pairs (0,0), (1,1), (2,2) -> 3 pairs
    - Gene|ameliorates|Disease: pairs (2,2), (3,3) -> 2 pairs
    """
    matrices_dir = os.path.join(tmpdir, "matrices")
    os.makedirs(matrices_dir)

    _save_npz(matrices_dir, "SmallMolecule__treats__Disease.npz",
              rows=[0, 1, 2], cols=[0, 1, 2], nrows=10, ncols=5)
    _save_npz(matrices_dir, "SmallMolecule__ameliorates__Disease.npz",
              rows=[1, 2, 3], cols=[1, 2, 3], nrows=10, ncols=5)
    _save_npz(matrices_dir, "SmallMolecule__affects__Gene.npz",
              rows=[0, 1, 2, 3], cols=[0, 1, 2, 3], nrows=10, ncols=8)
    _save_npz(matrices_dir, "Gene__treats__Disease.npz",
              rows=[0, 1, 2], cols=[0, 1, 2], nrows=8, ncols=5)
    _save_npz(matrices_dir, "Gene__ameliorates__Disease.npz",
              rows=[2, 3], cols=[2, 3], nrows=8, ncols=5)

    manifest_matrices = [
        {"src_type": "SmallMolecule", "predicate": "treats", "tgt_type": "Disease",
         "nrows": 10, "ncols": 5, "nvals": 3, "filename": "SmallMolecule__treats__Disease.npz"},
        {"src_type": "SmallMolecule", "predicate": "ameliorates", "tgt_type": "Disease",
         "nrows": 10, "ncols": 5, "nvals": 3, "filename": "SmallMolecule__ameliorates__Disease.npz"},
        {"src_type": "SmallMolecule", "predicate": "affects", "tgt_type": "Gene",
         "nrows": 10, "ncols": 8, "nvals": 4, "filename": "SmallMolecule__affects__Gene.npz"},
        {"src_type": "Gene", "predicate": "treats", "tgt_type": "Disease",
         "nrows": 8, "ncols": 5, "nvals": 3, "filename": "Gene__treats__Disease.npz"},
        {"src_type": "Gene", "predicate": "ameliorates", "tgt_type": "Disease",
         "nrows": 8, "ncols": 5, "nvals": 2, "filename": "Gene__ameliorates__Disease.npz"},
    ]

    with open(os.path.join(matrices_dir, "manifest.json"), "w") as f:
        json.dump({"num_matrices": len(manifest_matrices), "matrices": manifest_matrices}, f)

    return matrices_dir, manifest_matrices


def test_compute_exact_target_pair_counts_less_than_sum():
    """Exact target counts must be less than summed counts when predicates overlap.

    SmallMolecule|treats|Disease has 3 pairs and SmallMolecule|ameliorates|Disease
    has 3 pairs, sharing 2 pairs.

    For the ancestor variant SmallMolecule|related_to|A|Disease the sum is 6 but
    exact union is 4.

    Note: ameliorates is a descendant of treats in Biolink, so the treats target
    also includes ameliorates pairs.  Sum for treats is 3 (only explicit treats
    paths counted), but exact union of treats+ameliorates base matrices is 4.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        from library.unified_index import load_base_matrices

        matrices_dir, manifest_matrices = _build_test_matrices_dir(tmpdir)
        _, base_matrices = load_base_matrices(matrices_dir)

        # Sum-based counts as build_target_variant_counts would produce.
        # Canonical form puts Disease first (alphabetically).
        target_variant_counts = {
            "Disease|treats|R|SmallMolecule": 3,
            "Disease|ameliorates|R|SmallMolecule": 3,
            "Disease|related_to|A|SmallMolecule": 6,  # sum of treats + ameliorates
        }
        counters = {}
        stage_timings = {}

        exact_counts, target_pair_sets = compute_exact_target_pair_counts(
            target_variant_counts, base_matrices, manifest_matrices,
            type1="SmallMolecule", type2="Disease",
            start_time=time.time(), progress_file=None,
            counters=counters, stage_timings=stage_timings,
        )

        # treats exact = union of base treats + ameliorates (ameliorates is
        # descendant of treats), so includes shared pairs -> 4 not 3
        assert exact_counts["Disease|treats|R|SmallMolecule"] == 4
        # ameliorates exact = just ameliorates base matrix -> 3
        assert exact_counts["Disease|ameliorates|R|SmallMolecule"] == 3

        # Ancestor predicate target: exact < sum due to overlapping pairs
        assert exact_counts["Disease|related_to|A|SmallMolecule"] == 4, (
            f"Expected 4 (union of 3+3 with 2 shared), "
            f"got {exact_counts['Disease|related_to|A|SmallMolecule']}"
        )
        assert exact_counts["Disease|related_to|A|SmallMolecule"] < target_variant_counts["Disease|related_to|A|SmallMolecule"]


def test_compute_direct_metrics_2hop():
    """Direct metrics for a 2-hop predictor against a 1-hop target."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from library.unified_index import load_base_matrices, build_unified_type_offsets

        matrices_dir, manifest_matrices = _build_test_matrices_dir(tmpdir)
        _, base_matrices = load_base_matrices(matrices_dir)

        type1, type2 = "SmallMolecule", "Disease"

        # Build target pair sets for the target variant
        target_variant_counts = {
            "Disease|treats|R|SmallMolecule": 3,
        }
        counters = {}
        stage_timings = {}
        exact_target_counts, target_pair_sets = compute_exact_target_pair_counts(
            target_variant_counts, base_matrices, manifest_matrices,
            type1=type1, type2=type2,
            start_time=time.time(), progress_file=None,
            counters=counters, stage_timings=stage_timings,
        )

        # Predictor: SM|affects|F|Gene|treats|F|Disease overlaps with target
        onehop_to_overlaps = {
            "Disease|treats|R|SmallMolecule": {
                "SmallMolecule|affects|F|Gene|treats|F|Disease": 3,
            },
        }

        results = compute_direct_metrics(
            onehop_to_overlaps, target_pair_sets, exact_target_counts,
            base_matrices, manifest_matrices,
            type1, type2,
            start_time=time.time(), progress_file=None,
            counters=counters, stage_timings=stage_timings,
        )

        target_key = "Disease|treats|R|SmallMolecule"
        assert target_key in results
        rows = results[target_key]
        assert len(rows) >= 1

        # Find the predictor row
        found = False
        for nhop_path, predictor_count, overlap, orientation_tag in rows:
            if nhop_path == "SmallMolecule|affects|F|Gene|treats|F|Disease":
                found = True
                # SM affects Gene: (0,0),(1,1),(2,2),(3,3) -> Gene treats Disease: (0,0),(1,1),(2,2)
                # SM->Gene->Disease: SM0->G0->D0, SM1->G1->D1, SM2->G2->D2 = 3 pairs
                assert predictor_count == 3
                # Target treats: SM(0,0),(1,1),(2,2) -> exact 4 (includes ameliorates)
                # But in unified space those are the same 3 SM pairs
                # Overlap: intersection of predictor pairs with target pairs
                assert overlap > 0
                assert orientation_tag in ('fwd', 'rev')
                break
        assert found, f"Expected predictor path not found in results: {rows}"


def test_compute_direct_metrics_1hop():
    """Direct metrics for a 1-hop predictor (same as target)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from library.unified_index import load_base_matrices

        matrices_dir, manifest_matrices = _build_test_matrices_dir(tmpdir)
        _, base_matrices = load_base_matrices(matrices_dir)

        type1, type2 = "SmallMolecule", "Disease"

        target_variant_counts = {
            "Disease|treats|R|SmallMolecule": 3,
        }
        counters = {}
        stage_timings = {}
        exact_target_counts, target_pair_sets = compute_exact_target_pair_counts(
            target_variant_counts, base_matrices, manifest_matrices,
            type1=type1, type2=type2,
            start_time=time.time(), progress_file=None,
            counters=counters, stage_timings=stage_timings,
        )

        # 1-hop predictor overlapping with same 1-hop target
        onehop_to_overlaps = {
            "Disease|treats|R|SmallMolecule": {
                "SmallMolecule|treats|F|Disease": 3,
            },
        }

        results = compute_direct_metrics(
            onehop_to_overlaps, target_pair_sets, exact_target_counts,
            base_matrices, manifest_matrices,
            type1, type2,
            start_time=time.time(), progress_file=None,
            counters=counters, stage_timings=stage_timings,
        )

        target_key = "Disease|treats|R|SmallMolecule"
        assert target_key in results
        rows = results[target_key]
        found = False
        for nhop_path, predictor_count, overlap, orientation_tag in rows:
            if nhop_path == "SmallMolecule|treats|F|Disease":
                found = True
                assert predictor_count == 3
                # Full overlap with itself
                assert overlap == 3
                assert orientation_tag in ('fwd', 'rev')
                break
        assert found


def test_group_type_pair_with_matrices_dir_produces_explicit_paths():
    """Integration: group_type_pair with matrices_dir produces explicit predictor paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results_2hop")
        output_dir = os.path.join(tmpdir, "grouped")
        os.makedirs(results_dir)
        os.makedirs(output_dir)

        matrices_dir, _ = _build_test_matrices_dir(tmpdir)

        # Overlap result file: predictor SM|affects|F|Gene|treats|F|Disease
        # overlapping with target SM|treats|F|Disease.
        result_file = os.path.join(results_dir, "results_matrix1_000.tsv")
        with open(result_file, "w") as f:
            f.write("predictor_metapath\tpredictor_count\tpredicted_metapath\tpredicted_count\toverlap\ttotal_possible\n")
            f.write("SmallMolecule|affects|F|Gene|treats|F|Disease\t3\tSmallMolecule|treats|F|Disease\t3\t3\t50\n")

        explicit_items = [
            ("SmallMolecule|treats|F|Disease", 3),
            ("SmallMolecule|affects|F|Gene|treats|F|Disease", 3),
        ]

        type_node_counts = {
            "SmallMolecule": 10,
            "Disease": 5,
        }

        group_type_pair(
            type1="SmallMolecule",
            type2="Disease",
            file_list=[result_file],
            output_dir=output_dir,
            n_hops=2,
            explicit_items=explicit_items,
            type_node_counts=type_node_counts,
            min_precision=0.001,
            excluded_types=set(),
            excluded_predicates=set(),
            matrices_dir=matrices_dir,
        )

        # Canonical form puts Disease first (alphabetically), so output file
        # is Disease_treats_R_SmallMolecule.tsv.zst
        output_path = os.path.join(output_dir, "Disease_treats_R_SmallMolecule.tsv.zst")
        assert os.path.exists(output_path), f"Output file not found: {output_path}"

        with zstandard.open(output_path, "rt") as f:
            rows = [line.strip() for line in f if line.strip()]

        # Should have header + at least one data row
        assert len(rows) >= 2, f"Expected at least 2 rows (header+data), got {len(rows)}"

        header = rows[0].split("\t")
        assert header[0] == "predictor_metapath"
        assert header[1] == "orientation"

        # Output should contain explicit predictor paths (not rolled-up variants)
        predictor_paths = []
        for row in rows[1:]:
            parts = row.split("\t")
            assert len(parts) == len(header), f"Row has wrong number of fields: {row}"
            predictor_paths.append(parts[0])
            assert parts[1] in ('fwd', 'rev'), f"Bad orientation: {parts[1]}"
            predictor_count = int(parts[2])
            overlap_val = int(parts[3])
            assert predictor_count > 0
            assert overlap_val > 0

        # Should have the explicit 2-hop predictor path
        assert "SmallMolecule|affects|F|Gene|treats|F|Disease" in predictor_paths
