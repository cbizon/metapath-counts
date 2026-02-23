#!/usr/bin/env python3
"""
Combine an N-hop DAG with a 1-hop DAG to build an (N+1)-hop DAG.

Definition:
- Nodes are all legal (N+1)-hop metapaths formed by joining an N-hop
  metapath with a 1-hop metapath on the shared join type.
- Edges are direct-parent edges: change exactly one component to its
  immediate parent (as provided by the input DAG edges), with all other
  components unchanged, and the resulting (N+1)-hop remains legal.
  This includes paired parent changes that update the join node.
Outputs:
- nodes.tsv with header "metapath"
- edges.tsv with header "child\tparent"
- provenance.json with run metadata
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from library.aggregation import parse_metapath, build_metapath

Parsed = Tuple[List[str], List[str], List[str]]


def read_nodes(path: str) -> Iterable[str]:
    with open(path, "r") as f:
        header = f.readline().strip().split("\t")
        if not header or header[0] != "metapath":
            raise ValueError(f"Invalid nodes file header in {path}: {header}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line


def read_edges(path: str) -> Dict[str, List[str]]:
    parents = defaultdict(list)
    with open(path, "r") as f:
        header = f.readline().strip().split("\t")
        if header[:2] != ["child", "parent"]:
            raise ValueError(f"Invalid edges file header in {path}: {header}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            child, parent = line.split("\t")[:2]
            parents[child].append(parent)
    return parents


def combine_parsed(nhop: Parsed, onehop: Parsed) -> Parsed:
    n_nodes, n_preds, n_dirs = nhop
    o_nodes, o_preds, o_dirs = onehop
    if n_nodes[-1] != o_nodes[0]:
        raise ValueError(
            f"Join type mismatch: N-hop ends with {n_nodes[-1]}, 1-hop starts with {o_nodes[0]}"
        )
    return (
        n_nodes + o_nodes[1:],
        n_preds + o_preds,
        n_dirs + o_dirs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine an N-hop DAG with a 1-hop DAG to build an (N+1)-hop DAG"
    )
    parser.add_argument("--nhop-dir", required=True, help="N-hop DAG directory (nodes.tsv/edges.tsv)")
    parser.add_argument("--onehop-dir", required=True, help="1-hop DAG directory (nodes.tsv/edges.tsv)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--log-every",
        type=int,
        default=100000,
        help="Log progress every N N-hop nodes (default: 100000, 0 disables)",
    )

    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    nodes_out = os.path.join(args.output_dir, "nodes.tsv")
    edges_out = os.path.join(args.output_dir, "edges.tsv")
    provenance_out = os.path.join(args.output_dir, "provenance.json")

    nhop_nodes = os.path.join(args.nhop_dir, "nodes.tsv")
    nhop_edges = os.path.join(args.nhop_dir, "edges.tsv")
    onehop_nodes = os.path.join(args.onehop_dir, "nodes.tsv")
    onehop_edges = os.path.join(args.onehop_dir, "edges.tsv")

    # Load and index 1-hop nodes by start type
    t_index = time.time()
    onehop_by_start: Dict[str, List[Tuple[str, Parsed]]] = defaultdict(list)
    onehop_cache: Dict[str, Parsed] = {}
    for metapath in read_nodes(onehop_nodes):
        parsed = parse_metapath(metapath)
        onehop_cache[metapath] = parsed
        start_type = parsed[0][0]
        onehop_by_start[start_type].append((metapath, parsed))
    onehop_count = len(onehop_cache)

    nhop_parents = read_edges(nhop_edges)
    onehop_parents = read_edges(onehop_edges)
    t_index_done = time.time()

    nhop_cache: Dict[str, Parsed] = {}

    seen_nodes = set()
    seen_edges = set()

    nhop_total = 0
    nhop_join_miss = 0
    combined_nodes_written = 0
    combined_edges_written = 0
    parent_edges_n = 0
    parent_edges_o = 0
    parent_edges_pair = 0

    with open(nodes_out, "w") as f_nodes, open(edges_out, "w") as f_edges:
        f_nodes.write("metapath\n")
        f_edges.write("child\tparent\n")

        for n_metapath in read_nodes(nhop_nodes):
            nhop_total += 1
            n_parsed = parse_metapath(n_metapath)
            nhop_cache[n_metapath] = n_parsed
            join_type = n_parsed[0][-1]

            if join_type not in onehop_by_start:
                nhop_join_miss += 1
                continue

            for o_metapath, o_parsed in onehop_by_start[join_type]:
                combined = combine_parsed(n_parsed, o_parsed)
                child_mp = build_metapath(*combined)
                if child_mp not in seen_nodes:
                    f_nodes.write(child_mp + "\n")
                    seen_nodes.add(child_mp)
                    combined_nodes_written += 1

                # N-hop parents (single-step change, join type unchanged)
                n_parents = nhop_parents.get(n_metapath, [])
                for n_parent in n_parents:
                    p_parsed = nhop_cache.get(n_parent)
                    if p_parsed is None:
                        p_parsed = parse_metapath(n_parent)
                        nhop_cache[n_parent] = p_parsed
                    if p_parsed[0][-1] != join_type:
                        continue
                    parent_comb = combine_parsed(p_parsed, o_parsed)
                    parent_mp = build_metapath(*parent_comb)
                    key = (child_mp, parent_mp)
                    if key not in seen_edges:
                        f_edges.write(f"{child_mp}\t{parent_mp}\n")
                        seen_edges.add(key)
                        combined_edges_written += 1
                        parent_edges_n += 1

                # 1-hop parents (single-step change, join type unchanged)
                o_parents = onehop_parents.get(o_metapath, [])
                for o_parent in o_parents:
                    p_parsed = onehop_cache.get(o_parent)
                    if p_parsed is None:
                        p_parsed = parse_metapath(o_parent)
                        onehop_cache[o_parent] = p_parsed
                    if p_parsed[0][0] != join_type:
                        continue
                    parent_comb = combine_parsed(n_parsed, p_parsed)
                    parent_mp = build_metapath(*parent_comb)
                    key = (child_mp, parent_mp)
                    if key not in seen_edges:
                        f_edges.write(f"{child_mp}\t{parent_mp}\n")
                        seen_edges.add(key)
                        combined_edges_written += 1
                        parent_edges_o += 1

                # Paired parent changes on the join node
                if n_parents and o_parents:
                    for n_parent in n_parents:
                        p_n = nhop_cache.get(n_parent)
                        if p_n is None:
                            p_n = parse_metapath(n_parent)
                            nhop_cache[n_parent] = p_n
                        new_join = p_n[0][-1]
                        if new_join == join_type:
                            continue
                        for o_parent in o_parents:
                            p_o = onehop_cache.get(o_parent)
                            if p_o is None:
                                p_o = parse_metapath(o_parent)
                                onehop_cache[o_parent] = p_o
                            if p_o[0][0] != new_join:
                                continue
                            parent_comb = combine_parsed(p_n, p_o)
                            parent_mp = build_metapath(*parent_comb)
                            key = (child_mp, parent_mp)
                            if key not in seen_edges:
                                f_edges.write(f"{child_mp}\t{parent_mp}\n")
                                seen_edges.add(key)
                                combined_edges_written += 1
                                parent_edges_pair += 1

            if args.log_every and nhop_total % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"[build_multihop_dag] N-hop processed={nhop_total} "
                    f"join_miss={nhop_join_miss} nodes={combined_nodes_written} "
                    f"edges={combined_edges_written} elapsed={elapsed:.1f}s"
                )

    total_elapsed = time.time() - t0
    print("[build_multihop_dag] Done")
    print(f"[build_multihop_dag] One-hop indexed={onehop_count}")
    print(f"[build_multihop_dag] N-hop processed={nhop_total}")
    print(f"[build_multihop_dag] Join misses={nhop_join_miss}")
    print(f"[build_multihop_dag] Nodes written={combined_nodes_written}")
    print(f"[build_multihop_dag] Edges written={combined_edges_written}")
    print(
        f"[build_multihop_dag] Edges by source: "
        f"N-side={parent_edges_n}, 1-hop-side={parent_edges_o}, paired={parent_edges_pair}"
    )
    print(
        "[build_multihop_dag] Timings: "
        f"index={(t_index_done - t_index):.1f}s total={total_elapsed:.1f}s"
    )

    provenance = {
        "script": "build_multihop_dag.py",
        "output_dir": args.output_dir,
        "nhop_dir": args.nhop_dir,
        "onehop_dir": args.onehop_dir,
        "nodes_written": combined_nodes_written,
        "edges_written": combined_edges_written,
        "timestamp": time.time(),
    }
    with open(provenance_out, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
