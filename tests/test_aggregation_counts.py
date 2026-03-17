#!/usr/bin/env python3
"""Tests for count handling in the current explicit-item grouping pipeline."""

import os
import tempfile

import zstandard

from library import calculate_metrics
from pipeline.prepare_grouping import precompute_type_node_counts
from pipeline.workers.run_grouping import (
    build_target_variant_counts,
    get_max_predictor_count_for_precision,
    group_type_pair,
)


def _read_rows(path):
    with zstandard.open(path, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


def test_metrics_use_predictor_and_target_counts_independently():
    metrics = calculate_metrics(1000, 500, 100, 1_000_000)
    assert abs(metrics["precision"] - 0.1) < 1e-4
    assert abs(metrics["recall"] - 0.2) < 1e-4


def test_max_predictor_count_for_precision():
    assert get_max_predictor_count_for_precision(1, 0.001) == 1000
    assert get_max_predictor_count_for_precision(5, 0.01) == 500
    assert get_max_predictor_count_for_precision(5, 0.0) is None


def test_build_target_variant_counts_uses_explicit_1hop_items():
    explicit_items = [
        ("SmallMolecule|treats|F|Disease", 2),
        ("Drug|treats|F|Disease", 3),
        ("SmallMolecule|affects|F|Gene|treats|F|Disease", 7),
    ]

    counts, _ = build_target_variant_counts(
        explicit_items,
        type1="ChemicalEntity",
        type2="Disease",
    )

    assert counts["ChemicalEntity|treats|F|Disease"] == 5


def test_group_type_pair_keeps_exact_predictor_count_not_multiplied():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results_2hop")
        output_dir = os.path.join(tmpdir, "grouped")
        os.makedirs(results_dir)
        os.makedirs(output_dir)

        result_file = os.path.join(results_dir, "results_matrix1_000.tsv")
        with open(result_file, "w") as f:
            f.write(
                "predictor_metapath\tpredictor_count\tpredicted_metapath\t"
                "predicted_count\toverlap\ttotal_possible\n"
            )
            f.write(
                "Gene|affects|F|Disease|treats|R|SmallMolecule\t1000\t"
                "Gene|treats|F|SmallMolecule\t200\t50\t1000000\n"
            )
            f.write(
                "Gene|affects|F|Disease|treats|R|SmallMolecule\t1000\t"
                "Gene|interacts_with|A|SmallMolecule\t150\t30\t1000000\n"
            )

        explicit_items = [
            ("Gene|affects|F|Disease|treats|R|SmallMolecule", 1000),
            ("Gene|treats|F|SmallMolecule", 200),
            ("Gene|interacts_with|A|SmallMolecule", 150),
        ]

        type_node_counts = {"Gene": 500, "SmallMolecule": 400}

        group_type_pair(
            type1="Gene",
            type2="SmallMolecule",
            file_list=[result_file],
            output_dir=output_dir,
            n_hops=2,
            explicit_items=explicit_items,
            type_node_counts=type_node_counts,
            min_count=0,
            min_precision=0.0,
            excluded_types=set(),
            excluded_predicates=set(),
        )

        header, rows = _read_rows(os.path.join(output_dir, "Gene_treats_F_SmallMolecule.tsv.zst"))
        idx = {name: i for i, name in enumerate(header)}

        matching = [
            row
            for row in rows
            if row[idx["predictor_metapath"]] == "Gene|affects|F|Disease|treats|R|SmallMolecule"
        ]
        assert matching, "Expected aggregated predictor row not found"
        assert matching[0][idx["predictor_count"]] == "1000"
        assert matching[0][idx["overlap"]] == "50"

        header, rows = _read_rows(os.path.join(output_dir, "Gene_interacts_with_A_SmallMolecule.tsv.zst"))
        idx = {name: i for i, name in enumerate(header)}
        matching = [
            row
            for row in rows
            if row[idx["predictor_metapath"]] == "Gene|affects|F|Disease|treats|R|SmallMolecule"
        ]
        assert matching, "Expected aggregated predictor row not found"
        assert matching[0][idx["predictor_count"]] == "1000"
        assert matching[0][idx["overlap"]] == "30"


def test_precompute_type_node_counts_rolls_up_manifest_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as f:
            f.write(
                "{\n"
                '  "matrices": [\n'
                '    {"src_type": "Gene", "tgt_type": "Disease", "nrows": 5, "ncols": 3},\n'
                '    {"src_type": "SmallMolecule", "tgt_type": "Gene", "nrows": 7, "ncols": 5}\n'
                "  ]\n"
                "}\n"
            )

        counts = precompute_type_node_counts(tmpdir)
        assert counts["Gene"] >= 5
        assert counts["Disease"] >= 3
        assert counts["ChemicalEntity"] >= 7
