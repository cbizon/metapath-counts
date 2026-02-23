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
from collections import defaultdict

import bmt

from library import get_symmetric_predicates
from library.hierarchy import (
    get_type_ancestors,
    get_type_parents,
    get_predicate_ancestors,
    get_predicate_parents,
)
from library.aggregation import get_predicate_variants, parse_compound_predicate
from library.type_assignment import assign_node_type, is_pseudo_type


def normalize_type(name: str) -> str:
    if not name:
        return name
    name = name.replace('biolink:', '').replace('_', ' ')
    return ''.join(word.capitalize() for word in name.split())


def normalize_predicate(name: str) -> str:
    if not name:
        return name
    return name.replace('biolink:', '').replace(' ', '_')


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


def iter_onehop_nodes(allowed_pairs):
    symmetric = get_symmetric_predicates()
    for pred, pairs in allowed_pairs.items():
        # Expand predicate variants (including qualifiers) using main aggregation logic
        pred_variants = get_predicate_variants(pred)
        for src, tgt in pairs:
            for pred_var in pred_variants:
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


def iter_onehop_edges(allowed_pairs):
    symmetric = get_symmetric_predicates()
    for pred, pairs in allowed_pairs.items():
        pred_variants = get_predicate_variants(pred)
        for src, tgt in pairs:
            for pred_var in pred_variants:
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
                for p_src in get_type_parents(src):
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
                for p_tgt in get_type_parents(tgt):
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
                for p_pred in get_predicate_parents(pred):
                    if is_allowed(allowed_pairs, p_pred, src, tgt):
                        for p_pred_var in get_predicate_variants(p_pred):
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

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    toolkit = bmt.Toolkit()
    if args.edges:
        if not args.nodes:
            raise ValueError("--nodes is required when using --edges")
        allowed_sets = collect_allowed_pairs_from_data(args.edges, args.nodes)
    else:
        allowed_sets = collect_allowed_pairs_from_biolink(toolkit)

    nodes_path = os.path.join(args.output_dir, 'nodes.tsv')
    edges_path = os.path.join(args.output_dir, 'edges.tsv')
    provenance_path = os.path.join(args.output_dir, 'provenance.json')

    seen_nodes = set()
    seen_edges = set()

    nodes_written = 0
    edges_written = 0

    with open(nodes_path, 'w') as f_nodes:
        f_nodes.write('metapath\n')
        for node in iter_onehop_nodes(allowed_sets):
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            f_nodes.write(node + '\n')
            nodes_written += 1

    with open(edges_path, 'w') as f_edges:
        f_edges.write('child\tparent\n')
        for child, parent in iter_onehop_edges(allowed_sets):
            key = (child, parent)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            f_edges.write(child + '\t' + parent + '\n')
            edges_written += 1

    provenance = {
        "script": "build_onehop_dag.py",
        "output_dir": args.output_dir,
        "edges_file": args.edges,
        "nodes_file": args.nodes,
        "mode": "data_filtered" if args.edges else "biolink_only",
        "nodes_written": nodes_written,
        "edges_written": edges_written,
        "timestamp": time.time(),
    }
    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
