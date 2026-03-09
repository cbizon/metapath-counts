#!/usr/bin/env python3
"""
Build a 1-hop metapath DAG from Biolink constraints.

Outputs:
- nodes.tsv: all allowed 1-hop metapaths (src|pred|dir|tgt)
- edges.tsv: child -> parent edges between metapaths
- provenance.json: run metadata

Optionally filters predicates and types based on KGX edges/nodes.
"""

import argparse
import json
import os
import time

import bmt

from library import get_symmetric_predicates
from library.dag_compression import (
    get_allowed_predicate_parents,
    get_allowed_type_parents,
    normalize_excluded_predicates,
    normalize_excluded_types,
    normalize_predicate,
    normalize_type,
    predicate_is_excluded,
)
from library.hierarchy import (
    get_type_ancestors,
    get_predicate_ancestors,
)
from library.aggregation import get_predicate_variants, parse_compound_predicate
from library.type_assignment import assign_node_type
from pipeline.dag_edge_io import BinaryEdgeWriter, write_node_ids
from pipeline.dag_shards import materialize_left_shards, materialize_right_shards


def load_node_types(nodes_file: str) -> dict:
    node_types = {}
    with open(nodes_file, 'r') as f:
        for line in f:
            node = json.loads(line)
            node_id = node.get('id')
            categories = node.get('category', [])
            if not node_id or not categories:
                continue
            assigned = assign_node_type(categories)
            if assigned:
                node_types[node_id] = assigned
    return node_types


def collect_allowed_pairs_from_biolink(toolkit):
    # all types in model
    classes = toolkit.get_all_classes()
    if isinstance(classes, dict):
        class_names = list(classes.keys())
    else:
        class_names = list(classes)
    all_types = {normalize_type(t) for t in class_names}

    # all predicates in model
    slots = toolkit.get_all_slots()
    if isinstance(slots, dict):
        pred_names = list(slots.keys())
    else:
        pred_names = list(slots)

    allowed = {}

    for pred in pred_names:
        pred_norm = normalize_predicate(pred)
        element = toolkit.get_element(pred.replace('_', ' '))
        domain = None
        range_ = None
        if element:
            domain = getattr(element, 'domain', None)
            range_ = getattr(element, 'range', None)

        if domain:
            dom = normalize_type(domain)
            dom_desc = toolkit.get_descendants(domain.replace('_', ' ')) or []
            allowed_src = {normalize_type(d) for d in dom_desc} | {dom}
        else:
            allowed_src = set(all_types)

        if range_:
            rng = normalize_type(range_)
            rng_desc = toolkit.get_descendants(range_.replace('_', ' ')) or []
            allowed_tgt = {normalize_type(d) for d in rng_desc} | {rng}
        else:
            allowed_tgt = set(all_types)

        pairs = set()
        for src in allowed_src:
            for tgt in allowed_tgt:
                pairs.add((src, tgt))
        allowed[pred_norm] = pairs

    return allowed


def collect_allowed_pairs_from_data(edges_file: str, nodes_file: str):
    node_types = load_node_types(nodes_file)
    allowed = {}

    type_ancestor_cache = {}
    pred_ancestor_cache = {}
    edges_seen = 0
    edges_used = 0
    triples = set()
    t0 = time.time()

    with open(edges_file, 'r') as f:
        for line in f:
            edges_seen += 1
            edge = json.loads(line)
            pred = normalize_predicate(edge.get('predicate', ''))
            if not pred:
                continue
            subj = edge.get('subject')
            obj = edge.get('object')
            if subj not in node_types or obj not in node_types:
                continue
            edges_used += 1

            src = node_types[subj]
            tgt = node_types[obj]
            triples.add((src, pred, tgt))

            if edges_seen % 100000 == 0:
                elapsed = time.time() - t0
                print(
                    f"[build_onehop_dag] edges_seen={edges_seen} "
                    f"edges_used={edges_used} triples={len(triples)} "
                    f"elapsed={elapsed:.1f}s"
                )

    print(
        f"[build_onehop_dag] unique triples={len(triples)} "
        f"from edges_seen={edges_seen} edges_used={edges_used}"
    )

    for src, pred, tgt in triples:
        if pred in pred_ancestor_cache:
            pred_ancestors = pred_ancestor_cache[pred]
        else:
            pred_ancestors = [pred] + list(get_predicate_ancestors(pred))
            pred_ancestor_cache[pred] = pred_ancestors

        if src in type_ancestor_cache:
            src_ancestors = type_ancestor_cache[src]
        else:
            src_ancestors = get_type_ancestors(src)
            type_ancestor_cache[src] = src_ancestors

        if tgt in type_ancestor_cache:
            tgt_ancestors = type_ancestor_cache[tgt]
        else:
            tgt_ancestors = get_type_ancestors(tgt)
            type_ancestor_cache[tgt] = tgt_ancestors

        for p in pred_ancestors:
            pairs = allowed.setdefault(p, set())
            for s in src_ancestors:
                for t in tgt_ancestors:
                    pairs.add((s, t))

    return allowed


def iter_onehop_nodes(allowed_pairs, excluded_types=frozenset(), excluded_predicates=frozenset()):
    symmetric = get_symmetric_predicates()
    for pred, pairs in allowed_pairs.items():
        if predicate_is_excluded(pred, excluded_predicates):
            continue
        # Expand predicate variants (including qualifiers) using main aggregation logic
        pred_variants = get_predicate_variants(pred)
        for src, tgt in pairs:
            if src in excluded_types or tgt in excluded_types:
                continue
            for pred_var in pred_variants:
                if predicate_is_excluded(pred_var, excluded_predicates):
                    continue
                base = parse_compound_predicate(pred_var)[0]
                if base in symmetric:
                    yield f"{src}|{pred_var}|A|{tgt}"
                    yield f"{tgt}|{pred_var}|A|{src}"
                else:
                    yield f"{src}|{pred_var}|F|{tgt}"
                    yield f"{tgt}|{pred_var}|R|{src}"


def is_allowed(allowed_pairs, pred, src, tgt):
    if pred not in allowed_pairs:
        return False
    return (src, tgt) in allowed_pairs[pred]


def iter_onehop_edges(allowed_pairs, excluded_types=frozenset(), excluded_predicates=frozenset()):
    symmetric = get_symmetric_predicates()
    for pred, pairs in allowed_pairs.items():
        if predicate_is_excluded(pred, excluded_predicates):
            continue
        pred_variants = get_predicate_variants(pred)
        for src, tgt in pairs:
            if src in excluded_types or tgt in excluded_types:
                continue
            for pred_var in pred_variants:
                if predicate_is_excluded(pred_var, excluded_predicates):
                    continue
                base = parse_compound_predicate(pred_var)[0]
                if base in symmetric:
                    direction = 'A'
                    child = f"{src}|{pred_var}|{direction}|{tgt}"
                    forward_only = True
                else:
                    direction = 'F'
                    child = f"{src}|{pred_var}|{direction}|{tgt}"
                    forward_only = False

                # parent types
                for p_src in get_allowed_type_parents(src, excluded_types):
                    if is_allowed(allowed_pairs, pred, p_src, tgt):
                        parent = f"{p_src}|{pred_var}|{direction}|{tgt}"
                        yield child, parent
                        if forward_only:
                            a_child = f"{tgt}|{pred_var}|A|{src}"
                            a_parent = f"{tgt}|{pred_var}|A|{p_src}"
                            yield a_child, a_parent
                        else:
                            r_child = f"{tgt}|{pred_var}|R|{src}"
                            r_parent = f"{tgt}|{pred_var}|R|{p_src}"
                            yield r_child, r_parent
                for p_tgt in get_allowed_type_parents(tgt, excluded_types):
                    if is_allowed(allowed_pairs, pred, src, p_tgt):
                        parent = f"{src}|{pred_var}|{direction}|{p_tgt}"
                        yield child, parent
                        if forward_only:
                            a_child = f"{tgt}|{pred_var}|A|{src}"
                            a_parent = f"{p_tgt}|{pred_var}|A|{src}"
                            yield a_child, a_parent
                        else:
                            r_child = f"{tgt}|{pred_var}|R|{src}"
                            r_parent = f"{p_tgt}|{pred_var}|R|{src}"
                            yield r_child, r_parent

                # parent predicates
                for p_pred in get_allowed_predicate_parents(pred, excluded_predicates):
                    if is_allowed(allowed_pairs, p_pred, src, tgt):
                        for p_pred_var in get_predicate_variants(p_pred):
                            if predicate_is_excluded(p_pred_var, excluded_predicates):
                                continue
                            p_base = parse_compound_predicate(p_pred_var)[0]
                            p_dir = 'A' if p_base in symmetric else 'F'
                            parent = f"{src}|{p_pred_var}|{p_dir}|{tgt}"
                            yield child, parent
                            if forward_only:
                                a_child = f"{tgt}|{pred_var}|A|{src}"
                                a_parent = f"{tgt}|{p_pred_var}|A|{src}"
                                yield a_child, a_parent
                            else:
                                r_child = f"{tgt}|{pred_var}|R|{src}"
                                r_parent_dir = 'A' if p_base in symmetric else 'R'
                                r_parent = f"{tgt}|{p_pred_var}|{r_parent_dir}|{src}"
                                yield r_child, r_parent


def main():
    parser = argparse.ArgumentParser(description='Build 1-hop DAG from Biolink constraints')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--edges', default=None, help='Optional KGX edges.jsonl to filter predicates/types')
    parser.add_argument('--nodes', default=None, help='Optional KGX nodes.jsonl (required for type filtering)')
    parser.add_argument(
        '--exclude-types',
        default='',
        help='Comma-separated type names to exclude (compressed DAG skips through them)',
    )
    parser.add_argument(
        '--exclude-predicates',
        default='',
        help='Comma-separated predicate names to exclude (compressed DAG skips through them)',
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    toolkit = bmt.Toolkit()
    if args.edges:
        if not args.nodes:
            raise ValueError("--nodes is required when using --edges")
        allowed_sets = collect_allowed_pairs_from_data(args.edges, args.nodes)
    else:
        allowed_sets = collect_allowed_pairs_from_biolink(toolkit)
    excluded_types = normalize_excluded_types(args.exclude_types.split(','))
    excluded_predicates = normalize_excluded_predicates(args.exclude_predicates.split(','))

    nodes_path = os.path.join(args.output_dir, 'nodes.tsv')
    edges_bin_path = os.path.join(args.output_dir, 'edges.bin')
    node_ids_path = os.path.join(args.output_dir, 'node_ids.tsv')
    provenance_path = os.path.join(args.output_dir, 'provenance.json')

    seen_nodes = set()
    node_to_id = {}
    seen_edges = set()

    nodes_written = 0
    edges_written = 0

    with open(nodes_path, 'w') as f_nodes:
        f_nodes.write('metapath\n')
        for node in iter_onehop_nodes(
            allowed_sets,
            excluded_types=excluded_types,
            excluded_predicates=excluded_predicates,
        ):
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            node_to_id[node] = nodes_written
            f_nodes.write(node + '\n')
            nodes_written += 1

    late_nodes = []

    def ensure_node_id(metapath: str) -> int:
        node_id = node_to_id.get(metapath)
        if node_id is None:
            node_id = len(node_to_id)
            node_to_id[metapath] = node_id
            late_nodes.append(metapath)
        return node_id

    with BinaryEdgeWriter(edges_bin_path) as f_edges:
        for child, parent in iter_onehop_edges(
            allowed_sets,
            excluded_types=excluded_types,
            excluded_predicates=excluded_predicates,
        ):
            key = (child, parent)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            f_edges.write_ids(ensure_node_id(child), ensure_node_id(parent))
            edges_written += 1

    if late_nodes:
        with open(nodes_path, 'a') as f_nodes:
            for node in late_nodes:
                f_nodes.write(node + '\n')
        nodes_written += len(late_nodes)

    write_node_ids(node_ids_path, node_to_id)

    provenance = {
        "script": "build_onehop_dag.py",
        "output_dir": args.output_dir,
        "source_edges_file": args.edges,
        "nodes_file": args.nodes,
        "mode": "data_filtered" if args.edges else "biolink_only",
        "dag_edges_semantics": "nearest_allowed_parent" if (excluded_types or excluded_predicates) else "immediate_parent",
        "excluded_types": sorted(excluded_types),
        "excluded_predicates": sorted(excluded_predicates),
        "compression_enabled": bool(excluded_types or excluded_predicates),
        "nodes_written": nodes_written,
        "edges_written": edges_written,
        "edges_data_file": "edges.bin",
        "node_ids_file": "node_ids.tsv",
        "edges_encoding": "uint32_le_pairs",
        "timestamp": time.time(),
    }
    left_shards = materialize_left_shards(args.output_dir)
    right_shards = materialize_right_shards(args.output_dir)
    provenance["shards_left"] = {
        "path": os.path.join(args.output_dir, "shards_left"),
        "shard_count": left_shards["shard_count"],
    }
    provenance["shards_right"] = {
        "path": os.path.join(args.output_dir, "shards_right"),
        "shard_count": right_shards["shard_count"],
    }
    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
