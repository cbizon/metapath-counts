#!/usr/bin/env python3

import os
import pickle
import tempfile

import zstandard

from pipeline.workers.run_grouping import (
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
        assert any(row.startswith("ChemicalEntity|affects|F|Gene|treats|F|Disease\t500\t") for row in rows[1:])
        assert all(
            not row.startswith("ChemicalEntity|affects|F|BiologicalEntity|treats|F|Disease\t5000\t")
            for row in rows[1:]
        )
