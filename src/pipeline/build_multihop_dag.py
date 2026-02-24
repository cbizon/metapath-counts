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
Optionally, shard files are written under <output_dir>/shards/
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from library.aggregation import build_metapath

Parsed = Tuple[List[str], List[str], List[str]]


def read_nodes(path: str) -> Iterable[str]:
    return read_metapath_lines(path, has_header=True)


def read_metapath_lines(path: str, has_header: bool) -> Iterable[str]:
    with open(path, "r") as f:
        if has_header:
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


def safe_shard_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace("+", "+")
    )


def end_type(metapath: str) -> str:
    return metapath.rsplit("|", 1)[-1]


def suffix_after_first(metapath: str) -> str:
    """Return the metapath string after the first type (e.g., 'pred|dir|Tgt')."""
    return metapath.split("|", 1)[1]


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
    parser.add_argument(
        "--shard-by-join",
        action="store_true",
        help="Shard N-hop nodes by join type to reduce memory pressure",
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
    onehop_by_start: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    onehop_suffix_cache: Dict[str, str] = {}
    for metapath in read_nodes(onehop_nodes):
        stype = metapath.split("|", 1)[0]
        suffix = onehop_suffix_cache.get(metapath)
        if suffix is None:
            suffix = suffix_after_first(metapath)
            onehop_suffix_cache[metapath] = suffix
        onehop_by_start[stype].append((metapath, suffix))
    onehop_count = sum(len(v) for v in onehop_by_start.values())

    nhop_parents = read_edges(nhop_edges)
    onehop_parents = read_edges(onehop_edges)
    t_index_done = time.time()

    nhop_end_cache: Dict[str, str] = {}

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

        shard_dir = None
        shard_paths = {}
        if args.shard_by_join:
            shard_dir = os.path.join(args.output_dir, "shards")
            os.makedirs(shard_dir, exist_ok=True)
            shard_handles = {}
            shard_counts = defaultdict(int)
            for n_metapath in read_nodes(nhop_nodes):
                join_type = end_type(n_metapath)
                shard_name = safe_shard_name(join_type)
                shard_path = os.path.join(shard_dir, f"nhop_{shard_name}.tsv")
                shard_paths[join_type] = shard_path
                handle = shard_handles.get(shard_path)
                if handle is None:
                    handle = open(shard_path, "w")
                    shard_handles[shard_path] = handle
                handle.write(n_metapath + "\n")
                shard_counts[join_type] += 1
            for handle in shard_handles.values():
                handle.close()
            print(f"[build_multihop_dag] Sharded N-hop into {len(shard_paths)} files")

            for join_type, shard_path in shard_paths.items():
                if join_type not in onehop_by_start:
                    continue
                shard_nhop = 0
                shard_nodes_written = 0
                shard_edges_written = 0
                shard_onehop_count = len(onehop_by_start[join_type])
                shard_t_start = time.time()
                shard_parent_edges_n = 0
                shard_parent_edges_o = 0
                shard_parent_edges_pair = 0
                shard_t_combine = 0.0
                shard_t_nparents = 0.0
                shard_t_oparents = 0.0
                shard_t_oparent_check = 0.0
                shard_t_oparent_suffix = 0.0
                shard_t_oparent_build = 0.0
                shard_t_oparent_dedup = 0.0
                shard_t_pairparents = 0.0
                seen_nodes = set()
                seen_edges = set()
                for n_metapath in read_metapath_lines(shard_path, has_header=False):
                    shard_nhop += 1
                    nhop_total += 1
                    if join_type not in onehop_by_start:
                        nhop_join_miss += 1
                        continue

                    for o_metapath, o_suffix in onehop_by_start[join_type]:
                        t_step = time.time()
                        child_mp = f"{n_metapath}|{o_suffix}"
                        if child_mp not in seen_nodes:
                            f_nodes.write(child_mp + "\n")
                            seen_nodes.add(child_mp)
                            combined_nodes_written += 1
                            shard_nodes_written += 1
                        shard_t_combine += time.time() - t_step

                        n_parents = nhop_parents.get(n_metapath, [])
                        t_step = time.time()
                        for n_parent in n_parents:
                            p_end = nhop_end_cache.get(n_parent)
                            if p_end is None:
                                p_end = end_type(n_parent)
                                nhop_end_cache[n_parent] = p_end
                            if p_end != join_type:
                                continue
                            parent_mp = f"{n_parent}|{o_suffix}"
                            key = (child_mp, parent_mp)
                            if key not in seen_edges:
                                f_edges.write(f"{child_mp}\t{parent_mp}\n")
                                seen_edges.add(key)
                                combined_edges_written += 1
                                parent_edges_n += 1
                                shard_edges_written += 1
                                shard_parent_edges_n += 1
                        shard_t_nparents += time.time() - t_step

                        o_parents = onehop_parents.get(o_metapath, [])
                        t_step = time.time()
                        for o_parent in o_parents:
                            t_step_inner = time.time()
                            if not o_parent.startswith(f"{join_type}|"):
                                shard_t_oparent_check += time.time() - t_step_inner
                                continue
                            shard_t_oparent_check += time.time() - t_step_inner

                            t_step_inner = time.time()
                            p_suffix = onehop_suffix_cache.get(o_parent)
                            if p_suffix is None:
                                p_suffix = suffix_after_first(o_parent)
                                onehop_suffix_cache[o_parent] = p_suffix
                            shard_t_oparent_suffix += time.time() - t_step_inner

                            t_step_inner = time.time()
                            parent_mp = f"{n_metapath}|{p_suffix}"
                            shard_t_oparent_build += time.time() - t_step_inner

                            t_step_inner = time.time()
                            key = (child_mp, parent_mp)
                            if key not in seen_edges:
                                f_edges.write(f"{child_mp}\t{parent_mp}\n")
                                seen_edges.add(key)
                                combined_edges_written += 1
                                parent_edges_o += 1
                                shard_edges_written += 1
                                shard_parent_edges_o += 1
                            shard_t_oparent_dedup += time.time() - t_step_inner
                        shard_t_oparents += time.time() - t_step

                        if n_parents and o_parents:
                            t_step = time.time()
                            for n_parent in n_parents:
                                new_join = nhop_end_cache.get(n_parent)
                                if new_join is None:
                                    new_join = end_type(n_parent)
                                    nhop_end_cache[n_parent] = new_join
                                if new_join == join_type:
                                    continue
                                for o_parent in o_parents:
                                    if not o_parent.startswith(f"{new_join}|"):
                                        continue
                                    p_suffix = onehop_suffix_cache.get(o_parent)
                                    if p_suffix is None:
                                        p_suffix = suffix_after_first(o_parent)
                                        onehop_suffix_cache[o_parent] = p_suffix
                                    parent_mp = f"{n_parent}|{p_suffix}"
                                    key = (child_mp, parent_mp)
                                    if key not in seen_edges:
                                        f_edges.write(f"{child_mp}\t{parent_mp}\n")
                                        seen_edges.add(key)
                                        combined_edges_written += 1
                                        parent_edges_pair += 1
                                        shard_edges_written += 1
                                        shard_parent_edges_pair += 1
                            shard_t_pairparents += time.time() - t_step
                if args.log_every:
                    elapsed = time.time() - t0
                    shard_elapsed = time.time() - shard_t_start
                    print(
                        f"[build_multihop_dag] Shard join={join_type} "
                        f"nhop={shard_nhop} onehop={shard_onehop_count} "
                        f"nodes={shard_nodes_written} edges={shard_edges_written} "
                        f"edges_n={shard_parent_edges_n} "
                        f"edges_o={shard_parent_edges_o} "
                        f"edges_pair={shard_parent_edges_pair} "
                        f"shard_elapsed={shard_elapsed:.1f}s "
                        f"t_combine={shard_t_combine:.1f}s "
                        f"t_n={shard_t_nparents:.1f}s "
                        f"t_o={shard_t_oparents:.1f}s "
                        f"t_o_check={shard_t_oparent_check:.1f}s "
                        f"t_o_suffix={shard_t_oparent_suffix:.1f}s "
                        f"t_o_build={shard_t_oparent_build:.1f}s "
                        f"t_o_dedup={shard_t_oparent_dedup:.1f}s "
                        f"t_pair={shard_t_pairparents:.1f}s "
                        f"total_nhop={nhop_total} total_nodes={combined_nodes_written} "
                        f"total_edges={combined_edges_written} elapsed={elapsed:.1f}s"
                    )
        else:
            for n_metapath in read_nodes(nhop_nodes):
                nhop_total += 1
                join_type = end_type(n_metapath)

                if join_type not in onehop_by_start:
                    nhop_join_miss += 1
                    continue

                for o_metapath, o_suffix in onehop_by_start[join_type]:
                    child_mp = f"{n_metapath}|{o_suffix}"
                    if child_mp not in seen_nodes:
                        f_nodes.write(child_mp + "\n")
                        seen_nodes.add(child_mp)
                        combined_nodes_written += 1

                    # N-hop parents (single-step change, join type unchanged)
                    n_parents = nhop_parents.get(n_metapath, [])
                    for n_parent in n_parents:
                        p_end = nhop_end_cache.get(n_parent)
                        if p_end is None:
                            p_end = end_type(n_parent)
                            nhop_end_cache[n_parent] = p_end
                        if p_end != join_type:
                            continue
                        parent_mp = f"{n_parent}|{o_suffix}"
                        key = (child_mp, parent_mp)
                        if key not in seen_edges:
                            f_edges.write(f"{child_mp}\t{parent_mp}\n")
                            seen_edges.add(key)
                            combined_edges_written += 1
                            parent_edges_n += 1

                    # 1-hop parents (single-step change, join type unchanged)
                    o_parents = onehop_parents.get(o_metapath, [])
                    for o_parent in o_parents:
                        if not o_parent.startswith(f"{join_type}|"):
                            continue
                        p_suffix = onehop_suffix_cache.get(o_parent)
                        if p_suffix is None:
                            p_suffix = suffix_after_first(o_parent)
                            onehop_suffix_cache[o_parent] = p_suffix
                        parent_mp = f"{n_metapath}|{p_suffix}"
                        key = (child_mp, parent_mp)
                        if key not in seen_edges:
                            f_edges.write(f"{child_mp}\t{parent_mp}\n")
                            seen_edges.add(key)
                            combined_edges_written += 1
                            parent_edges_o += 1

                    # Paired parent changes on the join node
                    if n_parents and o_parents:
                        for n_parent in n_parents:
                            new_join = nhop_end_cache.get(n_parent)
                            if new_join is None:
                                new_join = end_type(n_parent)
                                nhop_end_cache[n_parent] = new_join
                            if new_join == join_type:
                                continue
                            for o_parent in o_parents:
                                if not o_parent.startswith(f"{new_join}|"):
                                    continue
                                p_suffix = onehop_suffix_cache.get(o_parent)
                                if p_suffix is None:
                                    p_suffix = suffix_after_first(o_parent)
                                    onehop_suffix_cache[o_parent] = p_suffix
                                parent_mp = f"{n_parent}|{p_suffix}"
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
        "shard_by_join": bool(args.shard_by_join),
        "timestamp": time.time(),
    }
    with open(provenance_out, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
