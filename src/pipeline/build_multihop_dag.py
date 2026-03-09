#!/usr/bin/env python3
"""
Combine an N-hop DAG with a 1-hop DAG to build an (N+1)-hop DAG.

Definition:
- Nodes are all legal (N+1)-hop metapaths formed by joining an N-hop
  metapath with a 1-hop metapath on the shared join type.
- Edges are direct-parent edges: change exactly one component to its
  parent in the input DAG (which may be immediate or compressed), with all
  other components unchanged, and the resulting (N+1)-hop remains legal.
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
from contextlib import ExitStack
from typing import Dict, Iterable, List, Tuple

from library.aggregation import build_metapath
from library.dag_compression import (
    metapath_has_excluded,
    normalize_excluded_predicates,
    normalize_excluded_types,
)
from pipeline.dag_edge_io import BinaryEdgeWriter, iter_edge_pairs_auto, write_node_ids

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


def read_shard_metapath_lines(path: str) -> Iterable[str]:
    with open(path, "r") as f:
        first = f.readline().strip()
        if not first:
            return
        if first == "metapath":
            for line in f:
                line = line.strip()
                if line:
                    yield line
            return
        yield first
        for line in f:
            line = line.strip()
            if line:
                yield line


def read_edges(
    path: str,
    excluded_types: frozenset[str] = frozenset(),
    excluded_predicates: frozenset[str] = frozenset(),
) -> Dict[str, List[str]]:
    parents = defaultdict(list)
    for child, parent in iter_edge_pairs_auto(path):
        if metapath_has_excluded(child, excluded_types, excluded_predicates):
            continue
        if metapath_has_excluded(parent, excluded_types, excluded_predicates):
            continue
        parents[child].append(parent)
    return parents


def read_edges_filtered(
    path: str,
    allowed_children: set[str],
    excluded_types: frozenset[str] = frozenset(),
    excluded_predicates: frozenset[str] = frozenset(),
) -> Dict[str, List[str]]:
    parents = defaultdict(list)
    for child, parent in iter_edge_pairs_auto(path):
        if child in allowed_children:
            if metapath_has_excluded(child, excluded_types, excluded_predicates):
                continue
            if metapath_has_excluded(parent, excluded_types, excluded_predicates):
                continue
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


def edge_shard_path_for_nhop_shard(nhop_shard_path: str) -> str | None:
    base = os.path.basename(nhop_shard_path)
    if not base.startswith("nhop_") or not base.endswith(".tsv"):
        return None
    shard_dir = os.path.dirname(nhop_shard_path)
    suffix = base[len("nhop_"):-4]
    bin_path = os.path.join(shard_dir, f"nhop_edges_{suffix}.bin")
    if os.path.exists(bin_path):
        return bin_path
    return os.path.join(shard_dir, f"nhop_edges_{suffix}.tsv")


def onehop_left_shard_paths(onehop_dir: str, join_type: str) -> tuple[str, str]:
    shard_name = safe_shard_name(join_type)
    shard_dir = os.path.join(onehop_dir, "shards_left")
    edge_bin = os.path.join(shard_dir, f"onehop_edges_start_{shard_name}.bin")
    edge_path = edge_bin if os.path.exists(edge_bin) else os.path.join(shard_dir, f"onehop_edges_start_{shard_name}.tsv")
    return (
        os.path.join(shard_dir, f"onehop_start_{shard_name}.tsv"),
        edge_path,
    )


def prepare_join_shards(
    nhop_nodes_path: str,
    nhop_edges_path: str,
    shard_dir: str,
    excluded_types: frozenset[str],
    excluded_predicates: frozenset[str],
) -> tuple[dict[str, str], dict[str, int], int]:
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths: dict[str, str] = {}
    shard_counts = defaultdict(int)

    # First pass: shard N-hop nodes by join type.
    with ExitStack() as stack:
        shard_handles: dict[str, object] = {}
        for n_metapath in read_nodes(nhop_nodes_path):
            if metapath_has_excluded(n_metapath, excluded_types, excluded_predicates):
                continue
            join_type = end_type(n_metapath)
            shard_name = safe_shard_name(join_type)
            shard_path = os.path.join(shard_dir, f"nhop_{shard_name}.tsv")
            shard_paths[join_type] = shard_path
            handle = shard_handles.get(shard_path)
            if handle is None:
                handle = stack.enter_context(open(shard_path, "w"))
                shard_handles[shard_path] = handle
            handle.write(n_metapath + "\n")
            shard_counts[join_type] += 1

    # Second pass: shard N-hop edges by child join type so shard workers avoid
    # rescanning the full N-hop edge file.
    edge_shards_written = 0
    with ExitStack() as stack:
        edge_handles: dict[str, object] = {}
        for child, parent in iter_edge_pairs_auto(nhop_edges_path):
            if metapath_has_excluded(child, excluded_types, excluded_predicates):
                continue
            if metapath_has_excluded(parent, excluded_types, excluded_predicates):
                continue
            join_type = end_type(child)
            if join_type not in shard_paths:
                continue
            edge_shard_path = os.path.join(shard_dir, f"nhop_edges_{safe_shard_name(join_type)}.tsv")
            handle = edge_handles.get(edge_shard_path)
            if handle is None:
                handle = stack.enter_context(open(edge_shard_path, "w"))
                handle.write("child\tparent\n")
                edge_handles[edge_shard_path] = handle
            handle.write(f"{child}\t{parent}\n")
            edge_shards_written += 1

    return shard_paths, dict(shard_counts), edge_shards_written


def end_type(metapath: str) -> str:
    return metapath.rsplit("|", 1)[-1]


def start_type(metapath: str) -> str:
    return metapath.split("|", 1)[0]


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
    parser.add_argument(
        "--prepare-shards-only",
        action="store_true",
        help="Only write join-type shard files and shard index, then exit",
    )
    parser.add_argument(
        "--join-type",
        default=None,
        help="Process only a single join type (requires --nhop-shard-file)",
    )
    parser.add_argument(
        "--nhop-shard-file",
        default=None,
        help="Shard file containing N-hop metapaths for --join-type",
    )
    parser.add_argument(
        "--exclude-types",
        default="",
        help="Comma-separated type names to exclude",
    )
    parser.add_argument(
        "--exclude-predicates",
        default="",
        help="Comma-separated predicate names to exclude",
    )
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    nodes_out = os.path.join(args.output_dir, "nodes.tsv")
    edges_out = os.path.join(args.output_dir, "edges.bin")
    write_nodes_file = not bool(args.join_type)
    node_ids_out = os.path.join(
        args.output_dir,
        "node_ids.tsv.gz" if args.join_type else "node_ids.tsv",
    )
    provenance_out = os.path.join(args.output_dir, "provenance.json")

    nhop_nodes = os.path.join(args.nhop_dir, "nodes.tsv")
    nhop_edges = os.path.join(args.nhop_dir, "edges.bin")
    if not os.path.exists(nhop_edges):
        nhop_edges = os.path.join(args.nhop_dir, "edges.tsv")
    onehop_nodes = os.path.join(args.onehop_dir, "nodes.tsv")
    onehop_edges = os.path.join(args.onehop_dir, "edges.bin")
    if not os.path.exists(onehop_edges):
        onehop_edges = os.path.join(args.onehop_dir, "edges.tsv")
    excluded_types = normalize_excluded_types(args.exclude_types.split(","))
    excluded_predicates = normalize_excluded_predicates(args.exclude_predicates.split(","))

    if args.prepare_shards_only:
        if not args.shard_by_join:
            raise ValueError("--prepare-shards-only requires --shard-by-join")
        if args.join_type or args.nhop_shard_file:
            raise ValueError("--prepare-shards-only cannot be combined with --join-type/--nhop-shard-file")
        shard_dir = os.path.join(args.output_dir, "shards")
        shard_index_path = os.path.join(shard_dir, "index.tsv")
        shard_paths, shard_counts, edge_shards_written = prepare_join_shards(
            nhop_nodes,
            nhop_edges,
            shard_dir,
            excluded_types=excluded_types,
            excluded_predicates=excluded_predicates,
        )
        with open(shard_index_path, "w") as f_idx:
            f_idx.write("join_type\tshard_path\tnhop_count\n")
            for join_type in sorted(shard_paths):
                f_idx.write(
                    f"{join_type}\t{shard_paths[join_type]}\t{shard_counts[join_type]}\n"
                )
        print(f"[build_multihop_dag] Sharded N-hop into {len(shard_paths)} files")
        provenance = {
            "script": "build_multihop_dag.py",
            "output_dir": args.output_dir,
            "nhop_dir": args.nhop_dir,
            "onehop_dir": args.onehop_dir,
            "mode": "prepare_shards_only",
            "dag_edges_semantics": "nearest_allowed_parent"
            if (excluded_types or excluded_predicates)
            else "input_dag_parent_edges",
            "excluded_types": sorted(excluded_types),
            "excluded_predicates": sorted(excluded_predicates),
            "compression_enabled": bool(excluded_types or excluded_predicates),
            "shard_count": len(shard_paths),
            "edge_shards_written": edge_shards_written,
            "timestamp": time.time(),
        }
        with open(provenance_out, "w") as f:
            json.dump(provenance, f, indent=2, sort_keys=True)
        return

    onehop_nodes_source = onehop_nodes
    onehop_edges_shard_source = None
    if args.join_type:
        left_nodes_shard, left_edges_shard = onehop_left_shard_paths(args.onehop_dir, args.join_type)
        if os.path.exists(left_nodes_shard):
            onehop_nodes_source = left_nodes_shard
        if os.path.exists(left_edges_shard):
            onehop_edges_shard_source = left_edges_shard

    # Load and index 1-hop nodes by start type
    t_index = time.time()
    onehop_by_start: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    onehop_start_cache: Dict[str, str] = {}
    onehop_suffix_cache: Dict[str, str] = {}
    for metapath in read_nodes(onehop_nodes_source):
        if metapath_has_excluded(metapath, excluded_types, excluded_predicates):
            continue
        stype = start_type(metapath)
        onehop_start_cache[metapath] = stype
        suffix = onehop_suffix_cache.get(metapath)
        if suffix is None:
            suffix = suffix_after_first(metapath)
            onehop_suffix_cache[metapath] = suffix
        onehop_by_start[stype].append((metapath, suffix))
    onehop_count = sum(len(v) for v in onehop_by_start.values())

    shard_nhop_rows: List[str] | None = None
    if args.join_type or args.nhop_shard_file:
        if not (args.join_type and args.nhop_shard_file):
            raise ValueError("--join-type and --nhop-shard-file must be provided together")
        shard_nhop_rows = [
            mp
            for mp in read_shard_metapath_lines(args.nhop_shard_file)
            if not metapath_has_excluded(mp, excluded_types, excluded_predicates)
        ]
        edge_shard_path = edge_shard_path_for_nhop_shard(args.nhop_shard_file)
        if edge_shard_path and os.path.exists(edge_shard_path):
            nhop_parents = read_edges(
                edge_shard_path,
                excluded_types=excluded_types,
                excluded_predicates=excluded_predicates,
            )
        else:
            nhop_parents = read_edges_filtered(
                nhop_edges,
                set(shard_nhop_rows),
                excluded_types=excluded_types,
                excluded_predicates=excluded_predicates,
            )
        if onehop_edges_shard_source and os.path.exists(onehop_edges_shard_source):
            onehop_parents = read_edges(
                onehop_edges_shard_source,
                excluded_types=excluded_types,
                excluded_predicates=excluded_predicates,
            )
        else:
            onehop_children = {mp for mp, _ in onehop_by_start.get(args.join_type, [])}
            onehop_parents = read_edges_filtered(
                onehop_edges,
                onehop_children,
                excluded_types=excluded_types,
                excluded_predicates=excluded_predicates,
            )
    else:
        nhop_parents = read_edges(
            nhop_edges,
            excluded_types=excluded_types,
            excluded_predicates=excluded_predicates,
        )
        onehop_parents = read_edges(
            onehop_edges,
            excluded_types=excluded_types,
            excluded_predicates=excluded_predicates,
        )
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

    node_to_id: Dict[str, int] = {}

    with ExitStack() as stack:
        f_nodes = None
        if write_nodes_file:
            f_nodes = stack.enter_context(open(nodes_out, "w"))
            f_nodes.write("metapath\n")
        f_edges = stack.enter_context(BinaryEdgeWriter(edges_out))

        def ensure_node(metapath: str) -> bool:
            nonlocal combined_nodes_written
            if metapath not in node_to_id:
                node_to_id[metapath] = len(node_to_id)
            if metapath in seen_nodes:
                return False
            seen_nodes.add(metapath)
            if f_nodes is not None:
                f_nodes.write(metapath + "\n")
            combined_nodes_written += 1
            return True

        def ensure_node_id(metapath: str) -> int:
            node_id = node_to_id.get(metapath)
            if node_id is None:
                node_id = len(node_to_id)
                node_to_id[metapath] = node_id
            return node_id

        def write_edge(child_mp: str, parent_mp: str) -> bool:
            nonlocal combined_edges_written
            key = (child_mp, parent_mp)
            if key in seen_edges:
                return False
            child_id = ensure_node_id(child_mp)
            parent_id = ensure_node_id(parent_mp)
            seen_edges.add(key)
            f_edges.write_ids(child_id, parent_id)
            combined_edges_written += 1
            return True

        shard_dir = None
        shard_paths = {}
        if args.join_type:
            join_type = args.join_type
            if shard_nhop_rows is None:
                raise ValueError("Internal error: shard rows not loaded")
            if join_type not in onehop_by_start:
                print(f"[build_multihop_dag] No 1-hop entries for join type {join_type}; writing empty outputs")
            else:
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
                for n_metapath in shard_nhop_rows:
                    if metapath_has_excluded(n_metapath, excluded_types, excluded_predicates):
                        continue
                    shard_nhop += 1
                    nhop_total += 1
                    for o_metapath, o_suffix in onehop_by_start[join_type]:
                        t_step = time.time()
                        child_mp = f"{n_metapath}|{o_suffix}"
                        if ensure_node(child_mp):
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
                            if write_edge(child_mp, parent_mp):
                                parent_edges_n += 1
                                shard_edges_written += 1
                                shard_parent_edges_n += 1
                        shard_t_nparents += time.time() - t_step

                        o_parents = onehop_parents.get(o_metapath, [])
                        t_step = time.time()
                        for o_parent in o_parents:
                            t_step_inner = time.time()
                            p_start = onehop_start_cache.get(o_parent)
                            if p_start is None:
                                p_start = start_type(o_parent)
                                onehop_start_cache[o_parent] = p_start
                            if p_start != join_type:
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
                            if write_edge(child_mp, parent_mp):
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
                                    o_start = onehop_start_cache.get(o_parent)
                                    if o_start is None:
                                        o_start = start_type(o_parent)
                                        onehop_start_cache[o_parent] = o_start
                                    if o_start != new_join:
                                        continue
                                    p_suffix = onehop_suffix_cache.get(o_parent)
                                    if p_suffix is None:
                                        p_suffix = suffix_after_first(o_parent)
                                        onehop_suffix_cache[o_parent] = p_suffix
                                    parent_mp = f"{n_parent}|{p_suffix}"
                                    if write_edge(child_mp, parent_mp):
                                        parent_edges_pair += 1
                                        shard_edges_written += 1
                                        shard_parent_edges_pair += 1
                            shard_t_pairparents += time.time() - t_step

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
        elif args.shard_by_join:
            shard_dir = os.path.join(args.output_dir, "shards")
            shard_index_path = os.path.join(shard_dir, "index.tsv")
            if not os.path.exists(shard_index_path):
                shard_paths, shard_counts, _ = prepare_join_shards(
                    nhop_nodes,
                    nhop_edges,
                    shard_dir,
                    excluded_types=excluded_types,
                    excluded_predicates=excluded_predicates,
                )
                with open(shard_index_path, "w") as f_idx:
                    f_idx.write("join_type\tshard_path\tnhop_count\n")
                    for join_type in sorted(shard_paths):
                        f_idx.write(
                            f"{join_type}\t{shard_paths[join_type]}\t{shard_counts[join_type]}\n"
                        )
                print(f"[build_multihop_dag] Sharded N-hop into {len(shard_paths)} files")
            else:
                with open(shard_index_path, "r") as f_idx:
                    header = f_idx.readline().strip().split("\t")
                    if header[:3] != ["join_type", "shard_path", "nhop_count"]:
                        raise ValueError(f"Invalid shard index header in {shard_index_path}: {header}")
                    for line in f_idx:
                        line = line.strip()
                        if not line:
                            continue
                        join_type, shard_path, _nhop_count = line.split("\t")[:3]
                        shard_paths[join_type] = shard_path
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
                for n_metapath in read_shard_metapath_lines(shard_path):
                    if metapath_has_excluded(n_metapath, excluded_types, excluded_predicates):
                        continue
                    shard_nhop += 1
                    nhop_total += 1
                    if join_type not in onehop_by_start:
                        nhop_join_miss += 1
                        continue

                    for o_metapath, o_suffix in onehop_by_start[join_type]:
                        t_step = time.time()
                        child_mp = f"{n_metapath}|{o_suffix}"
                        if ensure_node(child_mp):
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
                            if write_edge(child_mp, parent_mp):
                                parent_edges_n += 1
                                shard_edges_written += 1
                                shard_parent_edges_n += 1
                        shard_t_nparents += time.time() - t_step

                        o_parents = onehop_parents.get(o_metapath, [])
                        t_step = time.time()
                        for o_parent in o_parents:
                            t_step_inner = time.time()
                            p_start = onehop_start_cache.get(o_parent)
                            if p_start is None:
                                p_start = start_type(o_parent)
                                onehop_start_cache[o_parent] = p_start
                            if p_start != join_type:
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
                            if write_edge(child_mp, parent_mp):
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
                                    o_start = onehop_start_cache.get(o_parent)
                                    if o_start is None:
                                        o_start = start_type(o_parent)
                                        onehop_start_cache[o_parent] = o_start
                                    if o_start != new_join:
                                        continue
                                    p_suffix = onehop_suffix_cache.get(o_parent)
                                    if p_suffix is None:
                                        p_suffix = suffix_after_first(o_parent)
                                        onehop_suffix_cache[o_parent] = p_suffix
                                    parent_mp = f"{n_parent}|{p_suffix}"
                                    if write_edge(child_mp, parent_mp):
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
                if metapath_has_excluded(n_metapath, excluded_types, excluded_predicates):
                    continue
                nhop_total += 1
                join_type = end_type(n_metapath)

                if join_type not in onehop_by_start:
                    nhop_join_miss += 1
                    continue

                for o_metapath, o_suffix in onehop_by_start[join_type]:
                    child_mp = f"{n_metapath}|{o_suffix}"
                    ensure_node(child_mp)

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
                        if write_edge(child_mp, parent_mp):
                            parent_edges_n += 1

                    # 1-hop parents (single-step change, join type unchanged)
                    o_parents = onehop_parents.get(o_metapath, [])
                    for o_parent in o_parents:
                        p_start = onehop_start_cache.get(o_parent)
                        if p_start is None:
                            p_start = start_type(o_parent)
                            onehop_start_cache[o_parent] = p_start
                        if p_start != join_type:
                            continue
                        p_suffix = onehop_suffix_cache.get(o_parent)
                        if p_suffix is None:
                            p_suffix = suffix_after_first(o_parent)
                            onehop_suffix_cache[o_parent] = p_suffix
                        parent_mp = f"{n_metapath}|{p_suffix}"
                        if write_edge(child_mp, parent_mp):
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
                                o_start = onehop_start_cache.get(o_parent)
                                if o_start is None:
                                    o_start = start_type(o_parent)
                                    onehop_start_cache[o_parent] = o_start
                                if o_start != new_join:
                                    continue
                                p_suffix = onehop_suffix_cache.get(o_parent)
                                if p_suffix is None:
                                    p_suffix = suffix_after_first(o_parent)
                                    onehop_suffix_cache[o_parent] = p_suffix
                                parent_mp = f"{n_parent}|{p_suffix}"
                                if write_edge(child_mp, parent_mp):
                                    parent_edges_pair += 1

                if args.log_every and nhop_total % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[build_multihop_dag] N-hop processed={nhop_total} "
                        f"join_miss={nhop_join_miss} nodes={combined_nodes_written} "
                        f"edges={combined_edges_written} elapsed={elapsed:.1f}s"
                    )

    write_node_ids(node_ids_out, node_to_id)

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
        "join_type": args.join_type,
        "nhop_shard_file": args.nhop_shard_file,
        "dag_edges_semantics": "nearest_allowed_parent"
        if (excluded_types or excluded_predicates)
        else "input_dag_parent_edges",
        "excluded_types": sorted(excluded_types),
        "excluded_predicates": sorted(excluded_predicates),
        "compression_enabled": bool(excluded_types or excluded_predicates),
        "nodes_written": combined_nodes_written,
        "edges_written": combined_edges_written,
        "nodes_file": "nodes.tsv" if write_nodes_file else None,
        "edges_file": "edges.bin",
        "node_ids_file": os.path.basename(node_ids_out),
        "edges_encoding": "uint32_le_pairs",
        "shard_by_join": bool(args.shard_by_join),
        "timestamp": time.time(),
    }
    with open(provenance_out, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
