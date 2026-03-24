#!/usr/bin/env python3
"""Tests for count handling in the current explicit-item grouping pipeline."""

import json
import os
import tempfile

import numpy as np
import zstandard

from library import calculate_metrics
from pipeline.prepare_grouping import precompute_type_node_counts
from pipeline.workers.run_grouping import (
    build_target_variant_counts,
    group_type_pair,
)


def _read_rows(path):
    with zstandard.open(path, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


def _save_npz(directory, filename, rows, cols, nrows, ncols):
    np.savez_compressed(
        os.path.join(directory, filename),
        rows=np.array(rows, dtype=np.uint64),
        cols=np.array(cols, dtype=np.uint64),
        vals=np.ones(len(rows), dtype=bool),
        nrows=nrows,
        ncols=ncols,
        nvals=len(rows),
    )


def test_metrics_use_predictor_and_target_counts_independently():
    metrics = calculate_metrics(1000, 500, 100, 1_000_000)
    assert abs(metrics["precision"] - 0.1) < 1e-4
    assert abs(metrics["recall"] - 0.2) < 1e-4


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


def test_group_type_pair_with_matrices_keeps_exact_predictor_count():
    """With matrices_dir, direct evaluation produces exact predictor counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results_2hop")
        output_dir = os.path.join(tmpdir, "grouped")
        matrices_dir = os.path.join(tmpdir, "matrices")
        os.makedirs(results_dir)
        os.makedirs(output_dir)
        os.makedirs(matrices_dir)

        # Base matrices: Gene|affects|Disease and Disease|treats|SmallMolecule
        # Gene (nrows=500), Disease (intermediate, nrows=100, ncols=100), SmallMolecule (ncols=400)
        # Gene|affects|F|Disease: 50 pairs
        gene_rows = list(range(50))
        disease_cols = list(range(50))
        _save_npz(matrices_dir, "Gene__affects__Disease.npz",
                  rows=gene_rows, cols=disease_cols, nrows=500, ncols=100)

        # Disease|treats|SmallMolecule: 30 pairs
        disease_rows = list(range(30))
        sm_cols = list(range(30))
        _save_npz(matrices_dir, "Disease__treats__SmallMolecule.npz",
                  rows=disease_rows, cols=sm_cols, nrows=100, ncols=400)

        # Gene|treats|SmallMolecule: 20 pairs (1-hop target)
        _save_npz(matrices_dir, "Gene__treats__SmallMolecule.npz",
                  rows=list(range(20)), cols=list(range(20)), nrows=500, ncols=400)

        # Gene|interacts_with|SmallMolecule: 15 pairs (another 1-hop target)
        _save_npz(matrices_dir, "Gene__interacts_with__SmallMolecule.npz",
                  rows=list(range(15)), cols=list(range(15)), nrows=500, ncols=400)

        manifest_matrices = [
            {"src_type": "Gene", "predicate": "affects", "tgt_type": "Disease",
             "nrows": 500, "ncols": 100, "nvals": 50, "filename": "Gene__affects__Disease.npz"},
            {"src_type": "Disease", "predicate": "treats", "tgt_type": "SmallMolecule",
             "nrows": 100, "ncols": 400, "nvals": 30, "filename": "Disease__treats__SmallMolecule.npz"},
            {"src_type": "Gene", "predicate": "treats", "tgt_type": "SmallMolecule",
             "nrows": 500, "ncols": 400, "nvals": 20, "filename": "Gene__treats__SmallMolecule.npz"},
            {"src_type": "Gene", "predicate": "interacts_with", "tgt_type": "SmallMolecule",
             "nrows": 500, "ncols": 400, "nvals": 15, "filename": "Gene__interacts_with__SmallMolecule.npz"},
        ]
        with open(os.path.join(matrices_dir, "manifest.json"), "w") as f:
            json.dump({"num_matrices": len(manifest_matrices), "matrices": manifest_matrices}, f)

        # Overlap file: predictor Gene|affects|F|Disease|treats|F|SmallMolecule
        # overlaps with two targets
        result_file = os.path.join(results_dir, "results_matrix1_000.tsv")
        with open(result_file, "w") as f:
            f.write(
                "predictor_metapath\tpredictor_count\tpredicted_metapath\t"
                "predicted_count\toverlap\ttotal_possible\n"
            )
            f.write(
                "Gene|affects|F|Disease|treats|F|SmallMolecule\t1000\t"
                "Gene|treats|F|SmallMolecule\t200\t50\t1000000\n"
            )
            f.write(
                "Gene|affects|F|Disease|treats|F|SmallMolecule\t1000\t"
                "Gene|interacts_with|A|SmallMolecule\t150\t30\t1000000\n"
            )

        explicit_items = [
            ("Gene|affects|F|Disease|treats|F|SmallMolecule", 1000),
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
            matrices_dir=matrices_dir,
        )

        # Check that output was written for the treats target
        header, rows = _read_rows(os.path.join(output_dir, "Gene_treats_F_SmallMolecule.tsv.zst"))
        idx = {name: i for i, name in enumerate(header)}

        matching = [
            row
            for row in rows
            if row[idx["predictor_metapath"]] == "Gene|affects|F|Disease|treats|F|SmallMolecule"
        ]
        assert matching, "Expected predictor row not found in treats target output"
        # Exact predictor count from matrix reconstruction
        predictor_count = int(matching[0][idx["predictor_count"]])
        assert predictor_count > 0
        overlap_val = int(matching[0][idx["overlap"]])
        assert overlap_val > 0


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
