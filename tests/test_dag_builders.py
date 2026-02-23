#!/usr/bin/env python3
"""
Tests for DAG builders:
- build_onehop_dag.py (nodes/edges generation with dedup and directions)
- build_multihop_dag.py (N+1 hop DAG construction with paired join parents)
"""

import json
import sys

import pytest

import pipeline.build_onehop_dag as build_onehop_dag
import pipeline.build_multihop_dag as build_multihop_dag


def _read_tsv(path):
    with open(path, "r") as f:
        header = f.readline().strip().split("\t")
        rows = [line.strip().split("\t") for line in f if line.strip()]
    return header, rows


def test_build_onehop_nodes_edges_dedup(tmp_path, monkeypatch):
    allowed_sets = {
        "p": {("A", "B"), ("A", "B_parent")},
        "p_sym": {("A", "A")},
    }

    # Patch dependencies to avoid Biolink/Toolkit usage
    monkeypatch.setattr(build_onehop_dag, "collect_allowed_pairs_from_biolink", lambda *args, **kwargs: allowed_sets)
    monkeypatch.setattr(build_onehop_dag.bmt, "Toolkit", lambda: object())

    monkeypatch.setattr(build_onehop_dag, "get_symmetric_predicates", lambda: {"p_sym"})
    monkeypatch.setattr(build_onehop_dag, "get_predicate_variants", lambda pred: [pred])
    monkeypatch.setattr(build_onehop_dag, "parse_compound_predicate", lambda pred: (pred, []))
    monkeypatch.setattr(
        build_onehop_dag,
        "get_type_parents",
        lambda t: {"A_parent"} if t == "A" else {"B_parent"} if t == "B" else set(),
    )
    monkeypatch.setattr(build_onehop_dag, "get_predicate_parents", lambda pred: [])

    out_dir = tmp_path / "onehop"
    out_dir.mkdir()

    argv = [
        "build_onehop_dag.py",
        "--output-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    build_onehop_dag.main()

    nodes_path = out_dir / "nodes.tsv"
    edges_path = out_dir / "edges.tsv"

    _, node_rows = _read_tsv(nodes_path)
    _, edge_rows = _read_tsv(edges_path)

    nodes = {row[0] for row in node_rows}
    edges = {(row[0], row[1]) for row in edge_rows}

    assert nodes == {
        "A|p|F|B",
        "A|p|F|B_parent",
        "B|p|R|A",
        "B_parent|p|R|A",
        "A|p_sym|A|A",
    }

    assert edges == {
        ("A|p|F|B", "A|p|F|B_parent"),
        ("B|p|R|A", "B_parent|p|R|A"),
    }


def test_onehop_data_filter_ancestor_chain(monkeypatch, tmp_path):
    edges_path = tmp_path / "edges.jsonl"
    edges_path.write_text(
        json.dumps({"subject": "s1", "predicate": "p", "object": "o1"}) + "\n"
    )

    monkeypatch.setattr(build_onehop_dag, "load_node_types", lambda _: {"s1": "A", "o1": "B"})
    monkeypatch.setattr(
        build_onehop_dag,
        "get_type_ancestors",
        lambda t: {"A", "A_parent", "A_grand"} if t == "A" else {"B", "B_parent"},
    )
    monkeypatch.setattr(
        build_onehop_dag,
        "get_predicate_ancestors",
        lambda p: ["p_parent"] if p == "p" else [],
    )
    monkeypatch.setattr(
        build_onehop_dag,
        "get_type_parents",
        lambda t: {"A_parent"} if t == "A" else {"A_grand"} if t == "A_parent" else {"B_parent"} if t == "B" else set(),
    )
    monkeypatch.setattr(
        build_onehop_dag,
        "get_predicate_parents",
        lambda p: ["p_parent"] if p == "p" else [],
    )
    monkeypatch.setattr(build_onehop_dag, "get_predicate_variants", lambda p: [p])
    monkeypatch.setattr(build_onehop_dag, "parse_compound_predicate", lambda p: (p, []))
    monkeypatch.setattr(build_onehop_dag, "get_symmetric_predicates", lambda: set())

    allowed = build_onehop_dag.collect_allowed_pairs_from_data(str(edges_path), "nodes.jsonl")
    edges = set(build_onehop_dag.iter_onehop_edges(allowed))

    assert ("A|p|F|B", "A_parent|p|F|B") in edges
    assert ("A_parent|p|F|B", "A_grand|p|F|B") in edges
    assert ("A|p|F|B", "A|p|F|B_parent") in edges
    assert ("A|p|F|B", "A|p_parent|F|B") in edges


def test_onehop_symmetric_bidirectional(monkeypatch):
    allowed = {"p_sym": {("A", "B"), ("A_parent", "B"), ("A", "B_parent")}}
    monkeypatch.setattr(build_onehop_dag, "get_symmetric_predicates", lambda: {"p_sym"})
    monkeypatch.setattr(build_onehop_dag, "get_predicate_variants", lambda p: [p])
    monkeypatch.setattr(build_onehop_dag, "parse_compound_predicate", lambda p: (p, []))
    monkeypatch.setattr(
        build_onehop_dag,
        "get_type_parents",
        lambda t: {"A_parent"} if t == "A" else {"B_parent"} if t == "B" else set(),
    )
    monkeypatch.setattr(build_onehop_dag, "get_predicate_parents", lambda p: [])

    nodes = set(build_onehop_dag.iter_onehop_nodes(allowed))
    edges = set(build_onehop_dag.iter_onehop_edges(allowed))

    assert "A|p_sym|A|B" in nodes
    assert "B|p_sym|A|A" in nodes
    assert ("A|p_sym|A|B", "A_parent|p_sym|A|B") in edges
    assert ("B|p_sym|A|A", "B_parent|p_sym|A|A") in edges
    assert ("A|p_sym|A|B", "A|p_sym|A|B_parent") in edges
    assert ("B|p_sym|A|A", "B|p_sym|A|A_parent") in edges


def test_onehop_reverse_predicate_parent(monkeypatch):
    allowed = {"p": {("A", "B")}, "p_parent": {("A", "B")}}
    monkeypatch.setattr(build_onehop_dag, "get_symmetric_predicates", lambda: set())
    monkeypatch.setattr(build_onehop_dag, "get_predicate_variants", lambda p: [p])
    monkeypatch.setattr(build_onehop_dag, "parse_compound_predicate", lambda p: (p, []))
    monkeypatch.setattr(build_onehop_dag, "get_type_parents", lambda t: set())
    monkeypatch.setattr(
        build_onehop_dag,
        "get_predicate_parents",
        lambda p: ["p_parent"] if p == "p" else [],
    )

    edges = set(build_onehop_dag.iter_onehop_edges(allowed))
    assert ("B|p|R|A", "B|p_parent|R|A") in edges


def test_build_multihop_dedup_nodes(tmp_path, monkeypatch):
    nhop_dir = tmp_path / "nhop"
    onehop_dir = tmp_path / "onehop"
    nhop_dir.mkdir()
    onehop_dir.mkdir()
    nhop_nodes = nhop_dir / "nodes.tsv"
    nhop_edges = nhop_dir / "edges.tsv"
    onehop_nodes = onehop_dir / "nodes.tsv"
    onehop_edges = onehop_dir / "edges.tsv"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nhop_nodes.write_text(
        "metapath\n"
        "A|p|F|B\n"
        "A|p|F|B\n"
    )
    nhop_edges.write_text("child\tparent\n")

    onehop_nodes.write_text(
        "metapath\n"
        "B|q|F|C\n"
        "B|q|F|C\n"
    )
    onehop_edges.write_text("child\tparent\n")

    argv = [
        "build_multihop_dag.py",
        "--nhop-dir",
        str(nhop_dir),
        "--onehop-dir",
        str(onehop_dir),
        "--output-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    build_multihop_dag.main()

    _, node_rows = _read_tsv(out_dir / "nodes.tsv")
    nodes = [row[0] for row in node_rows]
    assert len(nodes) == 1
    assert nodes[0] == "A|p|F|B|q|F|C"


def test_build_onehop_requires_nodes_with_edges(monkeypatch):
    monkeypatch.setattr(build_onehop_dag.bmt, "Toolkit", lambda: object())
    argv = [
        "build_onehop_dag.py",
        "--output-dir",
        "out",
        "--edges",
        "edges.jsonl",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        build_onehop_dag.main()


def test_build_multihop_with_paired_join_parent(tmp_path, monkeypatch):
    nhop_dir = tmp_path / "nhop"
    onehop_dir = tmp_path / "onehop"
    nhop_dir.mkdir()
    onehop_dir.mkdir()
    nhop_nodes = nhop_dir / "nodes.tsv"
    nhop_edges = nhop_dir / "edges.tsv"
    onehop_nodes = onehop_dir / "nodes.tsv"
    onehop_edges = onehop_dir / "edges.tsv"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nhop_nodes.write_text(
        "metapath\n"
        "A|p|F|B\n"
        "A_parent|p|F|B\n"
        "A|p|F|B_parent\n"
    )
    nhop_edges.write_text(
        "child\tparent\n"
        "A|p|F|B\tA_parent|p|F|B\n"
        "A|p|F|B\tA|p|F|B_parent\n"
    )

    onehop_nodes.write_text(
        "metapath\n"
        "B|q|F|C\n"
        "B|q|F|C_parent\n"
        "B_parent|q|F|C\n"
    )
    onehop_edges.write_text(
        "child\tparent\n"
        "B|q|F|C\tB|q|F|C_parent\n"
        "B|q|F|C\tB_parent|q|F|C\n"
    )

    argv = [
        "build_multihop_dag.py",
        "--nhop-dir",
        str(nhop_dir),
        "--onehop-dir",
        str(onehop_dir),
        "--output-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    build_multihop_dag.main()

    _, edge_rows = _read_tsv(out_dir / "edges.tsv")
    edges = {(row[0], row[1]) for row in edge_rows}

    child = "A|p|F|B|q|F|C"
    expected_parents = {
        "A_parent|p|F|B|q|F|C",      # N-side parent (join unchanged)
        "A|p|F|B|q|F|C_parent",      # 1-hop parent (join unchanged)
        "A|p|F|B_parent|q|F|C",      # paired join parent
    }

    assert {(c, p) for (c, p) in edges if c == child} == {(child, p) for p in expected_parents}
