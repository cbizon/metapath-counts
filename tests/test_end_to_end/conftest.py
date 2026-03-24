"""Fixtures for end-to-end integration tests.

These fixtures run the full pipeline on the golden test graph.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import zstandard

from .golden_graph import write_golden_graph, GRAPH_STATS
from library import expand_metapath_to_variants
from pipeline.workers.run_overlap import analyze_nhop_overlap
from pipeline.prebuild_matrices import load_node_types, build_matrices
from pipeline.prepare_grouping import (
    precompute_type_node_counts,
)
from pipeline.workers.run_grouping import (
    group_type_pair,
)
from library.aggregation import promote_metapath_endpoints_to_typepair_starts, original_predictor_identity


def create_matrices_dir(matrices, output_dir):
    """Create a matrices directory with manifest.json and .npz files."""
    manifest = {
        "matrices": []
    }
    for (src_type, pred, tgt_type), matrix in matrices.items():
        filename = f"{src_type}__{pred}__{tgt_type}.npz"
        rows, cols, vals = matrix.to_coo()
        np.savez_compressed(
            str(output_dir / filename),
            rows=np.array(rows, dtype=np.uint64),
            cols=np.array(cols, dtype=np.uint64),
            vals=np.array(vals, dtype=bool),
            nrows=matrix.nrows,
            ncols=matrix.ncols,
            nvals=matrix.nvals,
        )
        manifest["matrices"].append({
            "src_type": src_type,
            "predicate": pred,
            "tgt_type": tgt_type,
            "direction": "F",
            "nrows": matrix.nrows,
            "ncols": matrix.ncols,
            "nvals": matrix.nvals,
            "filename": filename,
        })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def collect_explicit_items(result_file):
    """Collect unique explicit path counts from a raw result file."""
    counts = {}
    with open(result_file, "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            for path, count in ((parts[0], int(parts[1])), (parts[2], int(parts[3]))):
                previous = counts.get(path)
                if previous is None:
                    counts[path] = count
                else:
                    assert previous == count, (
                        f"Inconsistent explicit count for {path}: {previous} vs {count}"
                    )
    return sorted(counts.items())


def build_aggregated_counts(explicit_items):
    """Build hierarchical counts for assertions from explicit items."""
    aggregated_support = {}
    for path, count in explicit_items:
        predictor_id = original_predictor_identity(path)
        for variant in expand_metapath_to_variants(path):
            aggregated_support.setdefault(variant, {})[predictor_id] = count
    return {
        variant: sum(support.values())
        for variant, support in aggregated_support.items()
    }


def filter_explicit_items_for_type_pair(explicit_items, type1, type2):
    """Keep only explicit items whose endpoints can roll up to the requested type pair."""
    return [
        (path, count)
        for path, count in explicit_items
        if promote_metapath_endpoints_to_typepair_starts(path, type1, type2)
    ]


@pytest.fixture(scope="module")
def golden_workspace(tmp_path_factory):
    """Create a workspace with the golden test graph and built matrices.

    Module-scoped to avoid repeated setup.
    """
    workspace = tmp_path_factory.mktemp("golden")

    # Write the golden graph files
    nodes_path, edges_path = write_golden_graph(workspace)

    # Build matrices (in memory)
    node_types = load_node_types(str(nodes_path))
    matrices, node_to_idx = build_matrices(str(edges_path), node_types)

    # Create a fake matrices directory with manifest for prepare_grouping
    matrices_dir = workspace / "matrices"
    matrices_dir.mkdir()
    create_matrices_dir(matrices, matrices_dir)

    return {
        "workspace": workspace,
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "matrices_dir": matrices_dir,
        "node_types": node_types,
        "matrices": matrices,
        "node_to_idx": node_to_idx,
    }


@pytest.fixture(scope="module")
def pipeline_1hop(golden_workspace):
    """Run the full 1-hop pipeline and return all results."""
    workspace = golden_workspace["workspace"]
    matrices = golden_workspace["matrices"]
    matrices_dir = golden_workspace["matrices_dir"]

    n_hops = 1
    results_dir = workspace / f"results_{n_hops}hop"
    results_dir.mkdir()

    # Step 1: Run analysis
    result_file = results_dir / "results_matrix1_000.tsv"
    analyze_nhop_overlap(matrices, str(result_file), n_hops=n_hops)

    # Step 2: Build explicit items and aggregated counts for assertions.
    explicit_items = collect_explicit_items(result_file)
    aggregated_nhop_counts = build_aggregated_counts(explicit_items)

    # Step 3: Precompute type node counts
    type_node_counts = precompute_type_node_counts(str(matrices_dir))

    # Step 4: Run grouping for exactly the type pairs present in the golden graph.
    type_pairs = [
        # Explicit leaf type pairs from the golden graph
        ("Disease", "Gene"),
        ("Disease", "Protein"),
        ("Disease", "SmallMolecule"),
        ("Gene", "Gene"),
        ("Gene", "Protein"),
        ("Gene", "SmallMolecule"),
        # Hierarchical pairs explicitly tested in test_pipeline_1hop.py
        ("BiologicalEntity", "BiologicalEntity"),
        ("BiologicalEntity", "ChemicalEntity"),
        ("BiologicalEntity", "Disease"),
        ("ChemicalEntity", "Disease"),
        ("NamedThing", "NamedThing"),
    ]

    grouped_dir = workspace / f"grouped_by_results_{n_hops}hop"
    grouped_dir.mkdir()

    file_list = [str(result_file)]

    for type1, type2 in type_pairs:
        group_type_pair(
            type1=type1,
            type2=type2,
            file_list=file_list,
            output_dir=str(grouped_dir),
            n_hops=n_hops,
            explicit_items=filter_explicit_items_for_type_pair(explicit_items, type1, type2),
            type_node_counts=type_node_counts,
            min_count=0,  # No filtering for tests
            min_precision=0.0,
            excluded_types=set(),
            excluded_predicates=set(),
            matrices_dir=str(matrices_dir),
        )

    grouped_results = parse_all_grouped_files(grouped_dir)
    raw_results = parse_raw_results(result_file)

    return {
        "n_hops": n_hops,
        "workspace": workspace,
        "result_file": result_file,
        "results_dir": results_dir,
        "raw_results": raw_results,
        "explicit_items": explicit_items,
        "aggregated_nhop_counts": aggregated_nhop_counts,
        "type_node_counts": type_node_counts,
        "grouped_dir": grouped_dir,
        "grouped_results": grouped_results,
        "type_pairs": type_pairs,
    }


@pytest.fixture(scope="module")
def pipeline_2hop(golden_workspace):
    """Run the full 2-hop pipeline and return all results."""
    workspace = golden_workspace["workspace"]
    matrices = golden_workspace["matrices"]
    matrices_dir = golden_workspace["matrices_dir"]

    n_hops = 2
    results_dir = workspace / f"results_{n_hops}hop"
    results_dir.mkdir()

    # Step 1: Run analysis
    result_file = results_dir / "results_matrix1_000.tsv"
    analyze_nhop_overlap(matrices, str(result_file), n_hops=n_hops)

    # Step 2: Build explicit items and aggregated counts for assertions.
    explicit_items = collect_explicit_items(result_file)
    aggregated_nhop_counts = build_aggregated_counts(explicit_items)

    # Step 3: Precompute type node counts
    type_node_counts = precompute_type_node_counts(str(matrices_dir))

    # Step 4: Run grouping for exactly the type pairs present in the golden graph.
    type_pairs = [
        # Explicit leaf type pairs from the golden graph
        ("Disease", "Gene"),
        ("Disease", "Protein"),
        ("Disease", "SmallMolecule"),
        ("Gene", "Gene"),
        ("Gene", "Protein"),
        ("Gene", "SmallMolecule"),
        # Hierarchical pairs explicitly tested in test_pipeline_2hop.py
        ("BiologicalEntity", "BiologicalEntity"),
        ("ChemicalEntity", "Disease"),
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
    ]

    grouped_dir = workspace / f"grouped_by_results_{n_hops}hop"
    grouped_dir.mkdir()

    file_list = [str(result_file)]

    for type1, type2 in type_pairs:
        group_type_pair(
            type1=type1,
            type2=type2,
            file_list=file_list,
            output_dir=str(grouped_dir),
            n_hops=n_hops,
            explicit_items=filter_explicit_items_for_type_pair(explicit_items, type1, type2),
            type_node_counts=type_node_counts,
            min_count=0,
            min_precision=0.0,
            excluded_types=set(),
            excluded_predicates=set(),
            matrices_dir=str(matrices_dir),
        )

    grouped_results = parse_all_grouped_files(grouped_dir)
    raw_results = parse_raw_results(result_file)

    return {
        "n_hops": n_hops,
        "workspace": workspace,
        "result_file": result_file,
        "results_dir": results_dir,
        "raw_results": raw_results,
        "explicit_items": explicit_items,
        "aggregated_nhop_counts": aggregated_nhop_counts,
        "type_node_counts": type_node_counts,
        "grouped_dir": grouped_dir,
        "grouped_results": grouped_results,
        "type_pairs": type_pairs,
    }


def parse_all_grouped_files(grouped_dir):
    """Parse all grouped output files into a dict."""
    results = {}
    for filepath in Path(grouped_dir).glob("*.tsv.zst"):
        rows = parse_grouped_output(filepath)
        results[filepath.name] = rows
    return results


def parse_grouped_output(output_file):
    """Parse a grouped output TSV file into a list of dicts.

    Columns: predictor_metapath, predictor_count, overlap, total_possible,
             precision, recall, f1, mcc, specificity, npv
    """
    rows = []
    with zstandard.open(output_file, 'rt') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= len(header):
                row = {}
                for i, col in enumerate(header):
                    val = parts[i]
                    if col.endswith('_count') or col in ('overlap', 'total_possible'):
                        row[col] = int(val)
                    elif col in ('precision', 'recall', 'f1', 'mcc', 'specificity', 'npv'):
                        row[col] = float(val)
                    else:
                        row[col] = val
                rows.append(row)
    return rows


def parse_raw_results(result_file):
    """Parse raw analysis output TSV by position.

    Columns: predictor_metapath, predictor_count, predicted_metapath,
             predicted_count, overlap, total_possible
    """
    rows = []
    with open(result_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 6:
                rows.append({
                    'predictor_path': parts[0],
                    'predictor_count': int(parts[1]),
                    'predicted_path': parts[2],
                    'predicted_count': int(parts[3]),
                    'overlap': int(parts[4]),
                    'total_possible': int(parts[5]),
                })
    return rows
