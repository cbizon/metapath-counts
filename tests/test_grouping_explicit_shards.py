#!/usr/bin/env python3

import os
import pickle
import tempfile
import time

import zstandard

from library.aggregation import (
    canonical_variant_ancestor_closure_state_ids_for_typepair,
    canonical_variant_state_ids,
    expand_metapath_to_typepair_variants,
    promote_metapath_endpoints_to_typepair_starts,
    traverse_metapath_variants_for_typepair_pruned,
)
from pipeline.workers.run_grouping import (
    build_candidate_variants_for_targets,
    build_target_variant_counts,
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


def test_group_type_pair_uses_explicit_shard_counts_for_precision_pruning():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results_1hop")
        output_dir = os.path.join(tmpdir, "grouped")
        os.makedirs(results_dir)
        os.makedirs(output_dir)

        result_file = os.path.join(results_dir, "results_matrix1_000.tsv")
        with open(result_file, "w") as f:
            f.write("predictor_metapath\tpredictor_count\tpredicted_metapath\tpredicted_count\toverlap\ttotal_possible\n")
            f.write("SmallMolecule|affects|F|Gene|treats|F|Disease\t500\tSmallMolecule|treats|F|Disease\t1\t1\t100\n")

        explicit_items = [
            ("SmallMolecule|treats|F|Disease", 1),
            ("SmallMolecule|affects|F|Gene|treats|F|Disease", 500),
            ("ChemicalEntity|affects|F|BiologicalEntity|treats|F|Disease", 5000),
        ]

        type_node_counts = {
            "ChemicalEntity": 100,
            "Disease": 10,
        }

        group_type_pair(
            type1="ChemicalEntity",
            type2="Disease",
            file_list=[result_file],
            output_dir=output_dir,
            n_hops=2,
            explicit_items=explicit_items,
            type_node_counts=type_node_counts,
            min_precision=0.001,
            excluded_types=set(),
            excluded_predicates=set(),
        )

        output_path = os.path.join(output_dir, "ChemicalEntity_treats_F_Disease.tsv.zst")
        with zstandard.open(output_path, "rt") as f:
            rows = [line.strip() for line in f if line.strip()]

        assert len(rows) >= 2
        assert any(
            row.startswith("ChemicalEntity|affects|F|Gene|treats|F|Disease\t500\t")
            for row in rows[1:]
        )
        assert all("\t5000\t" not in row for row in rows[1:])


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


def test_pruned_states_carry_forward_across_targets_in_descending_target_count_order(monkeypatch):
    visits = []
    predictor_path = "SmallMolecule|treats|F|Disease"

    def fake_traverse(_metapath, _type1, _type2, visit_variant, visit_state=None, **kwargs):
        assert visit_state is not None
        should_prune = visit_state(("ChemicalEntity", "related_to", "A", "Disease"))
        visits.append(should_prune)
        if not should_prune:
            visit_variant(canonical_variant_state_ids("ChemicalEntity|related_to|A|Disease"))

    monkeypatch.setattr(
        "pipeline.workers.run_grouping.traverse_canonical_variants_for_typepair_pruned",
        fake_traverse,
    )

    onehop_to_overlaps = {
        "ChemicalEntity|small_target|A|Disease": {predictor_path: 1},
        "ChemicalEntity|big_target|A|Disease": {predictor_path: 1},
    }
    rolled_predictor = promote_metapath_endpoints_to_typepair_starts(
        predictor_path,
        "ChemicalEntity",
        "Disease",
    )[0]
    rolled_predictor_counts = {rolled_predictor: 150}
    global_variant_predictor_counts = {
        canonical_variant_state_ids("ChemicalEntity|related_to|A|Disease"): 150,
    }
    target_variant_counts = {
        "ChemicalEntity|big_target|A|Disease": 100,
        "ChemicalEntity|small_target|A|Disease": 50,
    }
    counters = {}
    stage_timings = {}

    candidate_rows_by_target, all_candidate_variants = build_candidate_variants_for_targets(
        onehop_to_overlaps=onehop_to_overlaps,
        rolled_predictor_counts=rolled_predictor_counts,
        global_variant_predictor_counts=global_variant_predictor_counts,
        target_variant_counts=target_variant_counts,
        type1="ChemicalEntity",
        type2="Disease",
        min_precision=1.0,
        start_time=time.time(),
        progress_file=None,
        counters=counters,
        stage_timings=stage_timings,
    )

    assert visits == [True, True]
    assert counters["targets_sorted_by_target_count"] == 2
    assert counters["candidate_states_branch_pruned"] == 1
    assert counters["candidate_state_revisits_pruned"] == 1
    assert candidate_rows_by_target["ChemicalEntity|big_target|A|Disease"] == {}
    assert candidate_rows_by_target["ChemicalEntity|small_target|A|Disease"] == {}
    assert all_candidate_variants == set()


def test_pruned_variant_ancestor_closure_reuses_pruning_across_predictors(monkeypatch):
    specific = canonical_variant_state_ids("ChemicalEntity|affects|F|Gene|treats|F|Disease")
    ancestor = canonical_variant_state_ids(
        "ChemicalEntity|related_to_at_instance_level|A|Gene|treats|F|Disease"
    )
    predictor_big = "ChemicalEntity|affects|F|Gene|treats|F|Disease"
    predictor_small = "ChemicalEntity|related_to_at_instance_level|A|Gene|treats|F|Disease"

    def fake_traverse(metapath, _type1, _type2, visit_variant, visit_state=None, **kwargs):
        assert visit_state is not None
        visit_state(("ChemicalEntity", "related_to", "A", "Disease"))
        if metapath == predictor_big:
            visit_variant(specific)
        elif metapath == predictor_small:
            visit_variant(ancestor)

    monkeypatch.setattr(
        "pipeline.workers.run_grouping.traverse_canonical_variants_for_typepair_pruned",
        fake_traverse,
    )

    onehop_to_overlaps = {
        "ChemicalEntity|treats|F|Disease": {
            predictor_big: 1,
            predictor_small: 1,
        }
    }
    rolled_predictor_counts = {
        predictor_big: 150,
        predictor_small: 1,
    }
    global_variant_predictor_counts = {
        specific: 150,
        ancestor: 151,
    }
    target_variant_counts = {
        "ChemicalEntity|treats|F|Disease": 100,
    }
    counters = {}
    stage_timings = {}

    candidate_rows_by_target, all_candidate_variants = build_candidate_variants_for_targets(
        onehop_to_overlaps=onehop_to_overlaps,
        rolled_predictor_counts=rolled_predictor_counts,
        global_variant_predictor_counts=global_variant_predictor_counts,
        target_variant_counts=target_variant_counts,
        type1="ChemicalEntity",
        type2="Disease",
        min_precision=1.0,
        start_time=time.time(),
        progress_file=None,
        counters=counters,
        stage_timings=stage_timings,
    )

    specific_closure = canonical_variant_ancestor_closure_state_ids_for_typepair(
        specific,
        type1="ChemicalEntity",
        type2="Disease",
    )
    assert ancestor in specific_closure
    assert counters["candidate_variants_branch_pruned"] == 1
    assert counters["candidate_variant_revisits_pruned"] == 1
    assert candidate_rows_by_target["ChemicalEntity|treats|F|Disease"] == {}
    assert all_candidate_variants == set()
