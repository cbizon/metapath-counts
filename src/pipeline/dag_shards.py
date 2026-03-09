#!/usr/bin/env python3
from __future__ import annotations

import argparse
from array import array
import cProfile
import gzip
import json
import math
import os
import pstats
import resource
import shutil
import time
import zlib
from collections import OrderedDict, defaultdict
from typing import Callable

from pipeline.dag_edge_io import (
    BinaryEdgeWriter,
    copy_or_write_node_ids,
    iter_edge_id_pairs_from_bin,
    iter_edge_pairs_auto,
    load_node_ids,
    resolve_node_ids_path,
)

# Static default chosen to prevent extreme over-sharding on large DAG2+ artifacts
# while still splitting very large join types for DAG3 memory safety.
DEFAULT_RIGHT_MAX_ROWS = 250000
MAX_OPEN_SHARD_FILES = 8192
PROGRESS_EVERY_NODES = 2_000_000
PROGRESS_EVERY_EDGES = 5_000_000


def _log(msg: str) -> None:
    print(f"[dag_shards] {msg}", flush=True)


def _rate(count: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return count / elapsed_s


class PhaseProfiler:
    def __init__(self, phase: str, enabled_phases: set[str], duration_sec: float, out_dir: str):
        self.phase = phase
        self.enabled = phase in enabled_phases
        self.duration_sec = duration_sec
        self.out_dir = out_dir
        self._prof: cProfile.Profile | None = None
        self._started_at = 0.0
        self._dumped = False
        self.stats_path: str | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._prof = cProfile.Profile()
        self._prof.enable()
        self._started_at = time.time()
        _log(f"profile[{self.phase}] started duration_sec={self.duration_sec if self.duration_sec > 0 else 'full-phase'}")

    def maybe_stop_and_dump(self) -> None:
        if (not self.enabled) or self._dumped or self._prof is None:
            return
        if self.duration_sec > 0 and (time.time() - self._started_at) >= self.duration_sec:
            self._stop_and_dump(reason="duration-reached")

    def stop_and_dump(self) -> None:
        if (not self.enabled) or self._dumped or self._prof is None:
            return
        self._stop_and_dump(reason="phase-complete")

    def _stop_and_dump(self, reason: str) -> None:
        assert self._prof is not None
        self._prof.disable()
        ts = int(time.time())
        stats_path = os.path.join(self.out_dir, f"profile_{self.phase}_{ts}.pstats")
        text_path = os.path.join(self.out_dir, f"profile_{self.phase}_{ts}.txt")
        self._prof.dump_stats(stats_path)
        with open(text_path, "w") as f:
            stats = pstats.Stats(self._prof, stream=f).sort_stats("cumulative")
            stats.print_stats(60)
        self.stats_path = stats_path
        self._dumped = True
        _log(f"profile[{self.phase}] dumped reason={reason} pstats={stats_path} text={text_path}")


def _effective_open_file_cap() -> int:
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Keep headroom for other files (logs, node_ids, etc.) while allowing many shard handles.
    return max(64, min(MAX_OPEN_SHARD_FILES, soft_limit - 64))


def safe_shard_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace("+", "+")
    )


def start_type(metapath: str) -> str:
    return metapath.split("|", 1)[0]


def end_type(metapath: str) -> str:
    return metapath.rsplit("|", 1)[-1]


def _iter_nodes_with_ids(dag_dir: str):
    node_ids_path = resolve_node_ids_path(dag_dir)
    if os.path.exists(node_ids_path):
        open_fn = gzip.open if node_ids_path.endswith(".gz") else open
        with open_fn(node_ids_path, "rt") as f:
            header = f.readline().strip().split("\t")
            if header[:2] != ["node_id", "metapath"]:
                raise ValueError(f"Invalid node_ids header in {node_ids_path}: {header}")
            expected = 0
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                node_id_str, metapath = line.split("\t", 1)
                node_id = int(node_id_str)
                if node_id != expected:
                    raise ValueError(
                        f"Non-contiguous node ids in {node_ids_path}: expected {expected}, got {node_id}"
                    )
                expected += 1
                yield node_id, metapath
        return

    nodes_path = os.path.join(dag_dir, "nodes.tsv")
    with open(nodes_path, "r") as f:
        header = f.readline().strip().split("\t")
        if not header or header[0] != "metapath":
            raise ValueError(f"Invalid nodes file header in {nodes_path}: {header}")
        for node_id, line in enumerate(f):
            metapath = line.strip()
            if metapath:
                yield node_id, metapath


def _materialize_shards(
    dag_dir: str,
    shard_dir_name: str,
    node_prefix: str,
    edge_prefix: str,
    key_fn: Callable[[str], str],
    index_key_col: str,
    max_rows: int | None = None,
    profile_phases: set[str] | None = None,
    profile_duration_sec: float = 0.0,
) -> dict:
    t0 = time.time()
    _log(
        f"start dag_dir={dag_dir} shard_dir={shard_dir_name} max_rows={max_rows if max_rows is not None else 'none'}"
    )
    edges_bin_path = os.path.join(dag_dir, "edges.bin")
    edges_tsv_path = os.path.join(dag_dir, "edges.tsv")
    canonical_node_ids_path = resolve_node_ids_path(dag_dir)
    if os.path.exists(edges_bin_path):
        edges_path = edges_bin_path
    else:
        edges_path = edges_tsv_path
    shard_dir = os.path.join(dag_dir, shard_dir_name)
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)
    profile_out_dir = shard_dir
    enabled_phases = profile_phases or set()

    shard_node_paths: dict[str, str] = {}
    shard_name_to_key: dict[str, str] = {}
    shard_name_to_idx: dict[str, int] = {}
    shard_names: list[str] = []
    shard_idx_by_node_id = array("I")
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    key_counts: dict[str, int] = defaultdict(int)
    key_parts: dict[str, int] = {}

    if max_rows and max_rows > 0:
        pp = PhaseProfiler("precount", enabled_phases, profile_duration_sec, profile_out_dir)
        pp.start()
        t_count0 = time.time()
        counted = 0
        for _node_id, metapath in _iter_nodes_with_ids(dag_dir):
            key_counts[key_fn(metapath)] += 1
            counted += 1
            pp.maybe_stop_and_dump()
            if counted % PROGRESS_EVERY_NODES == 0:
                elapsed = time.time() - t_count0
                _log(f"precount nodes={counted} rate={_rate(counted, elapsed):.0f}/s elapsed={elapsed:.1f}s")
        for key, count in key_counts.items():
            key_parts[key] = max(1, math.ceil(count / max_rows))
        pp.stop_and_dump()
        elapsed = time.time() - t_count0
        _log(
            f"precount done nodes={counted} keys={len(key_counts)} "
            f"rate={_rate(counted, elapsed):.0f}/s elapsed={elapsed:.1f}s"
        )

    def shard_name_for(key: str, metapath: str) -> str:
        base = safe_shard_name(key)
        parts = key_parts.get(key, 1)
        if parts <= 1:
            return base
        part = zlib.crc32(metapath.encode("utf-8")) % parts
        return f"{base}__part{part:04d}"

    open_file_cap = _effective_open_file_cap()
    _log(f"open_file_cap={open_file_cap}")

    node_handles: OrderedDict[str, object] = OrderedDict()
    node_initialized: set[str] = set()
    node_handle_evictions = 0
    node_peak_open_handles = 0

    def get_node_handle(shard_path: str):
        nonlocal node_handle_evictions, node_peak_open_handles
        handle = node_handles.pop(shard_path, None)
        if handle is not None:
            node_handles[shard_path] = handle
            return handle
        if len(node_handles) >= open_file_cap:
            _, old = node_handles.popitem(last=False)
            old.close()
            node_handle_evictions += 1
        mode = "a" if shard_path in node_initialized else "w"
        handle = open(shard_path, mode)
        if shard_path not in node_initialized:
            handle.write("metapath\n")
            node_initialized.add(shard_path)
        node_handles[shard_path] = handle
        if len(node_handles) > node_peak_open_handles:
            node_peak_open_handles = len(node_handles)
        return handle

    pp_nodes = PhaseProfiler("node-pass", enabled_phases, profile_duration_sec, profile_out_dir)
    pp_nodes.start()
    t_nodes0 = time.time()
    nodes_processed = 0
    for node_id, metapath in _iter_nodes_with_ids(dag_dir):
        key = key_fn(metapath)
        shard_name = shard_name_for(key, metapath)
        shard_idx = shard_name_to_idx.get(shard_name)
        if shard_idx is None:
            shard_idx = len(shard_names)
            shard_name_to_idx[shard_name] = shard_idx
            shard_names.append(shard_name)
        if node_id != len(shard_idx_by_node_id):
            raise ValueError(
                f"Non-contiguous node ids while materializing shards: expected {len(shard_idx_by_node_id)}, got {node_id}"
            )
        shard_idx_by_node_id.append(shard_idx)
        shard_path = os.path.join(shard_dir, f"{node_prefix}_{shard_name}.tsv")
        shard_node_paths[shard_name] = shard_path
        shard_name_to_key[shard_name] = key
        handle = get_node_handle(shard_path)
        handle.write(metapath + "\n")
        node_counts[shard_name] += 1
        nodes_processed += 1
        pp_nodes.maybe_stop_and_dump()
        if nodes_processed % PROGRESS_EVERY_NODES == 0:
            elapsed = time.time() - t_nodes0
            _log(
                f"node-pass nodes={nodes_processed} shards={len(shard_names)} "
                f"open_handles={len(node_handles)} peak_open_handles={node_peak_open_handles} "
                f"evictions={node_handle_evictions} rate={_rate(nodes_processed, elapsed):.0f}/s elapsed={elapsed:.1f}s"
            )

    for handle in node_handles.values():
        handle.close()
    pp_nodes.stop_and_dump()
    elapsed = time.time() - t_nodes0
    _log(
        f"node-pass done nodes={nodes_processed} shards={len(shard_names)} "
        f"peak_open_handles={node_peak_open_handles} evictions={node_handle_evictions} "
        f"rate={_rate(nodes_processed, elapsed):.0f}/s elapsed={elapsed:.1f}s"
    )
    _log(f"shard_files_estimate_N={len(shard_names)} open_file_cap_M={open_file_cap} n_gt_m={len(shard_names) > open_file_cap}")

    edge_handles: OrderedDict[str, BinaryEdgeWriter] = OrderedDict()
    edge_initialized: set[str] = set()
    edge_handle_evictions = 0
    edge_peak_open_handles = 0

    def get_edge_handle(edge_shard_path: str):
        nonlocal edge_handle_evictions, edge_peak_open_handles
        handle = edge_handles.pop(edge_shard_path, None)
        if handle is not None:
            edge_handles[edge_shard_path] = handle
            return handle
        if len(edge_handles) >= open_file_cap:
            _, old = edge_handles.popitem(last=False)
            old.close()
            edge_handle_evictions += 1
        handle = BinaryEdgeWriter(edge_shard_path, append=edge_shard_path in edge_initialized)
        edge_initialized.add(edge_shard_path)
        edge_handles[edge_shard_path] = handle
        if len(edge_handles) > edge_peak_open_handles:
            edge_peak_open_handles = len(edge_handles)
        return handle

    # Fast path: canonical edges.bin + canonical node_ids.tsv means edge ids are already canonical.
    pp_edges = PhaseProfiler("edge-pass", enabled_phases, profile_duration_sec, profile_out_dir)
    pp_edges.start()
    t_edges0 = time.time()
    edges_processed = 0
    if os.path.exists(edges_bin_path) and os.path.exists(canonical_node_ids_path):
        for child_id, parent_id in iter_edge_id_pairs_from_bin(edges_bin_path):
            try:
                shard_idx = shard_idx_by_node_id[child_id]
            except IndexError as e:
                raise ValueError(f"Edge child id out of range for shard map: child={child_id}") from e
            shard_name = shard_names[shard_idx]
            edge_shard_path = os.path.join(shard_dir, f"{edge_prefix}_{shard_name}.bin")
            handle = get_edge_handle(edge_shard_path)
            handle.write_ids(child_id, parent_id)
            edge_counts[shard_name] += 1
            edges_processed += 1
            pp_edges.maybe_stop_and_dump()
            if edges_processed % PROGRESS_EVERY_EDGES == 0:
                elapsed = time.time() - t_edges0
                _log(
                    f"edge-pass(bin) edges={edges_processed} "
                    f"open_handles={len(edge_handles)} peak_open_handles={edge_peak_open_handles} "
                    f"evictions={edge_handle_evictions} rate={_rate(edges_processed, elapsed):.0f}/s elapsed={elapsed:.1f}s"
                )
    else:
        canonical_node_to_id: dict[str, int] = {}
        if os.path.exists(canonical_node_ids_path):
            canonical_node_to_id, _ = load_node_ids(canonical_node_ids_path)
        else:
            for _node_id, metapath in _iter_nodes_with_ids(dag_dir):
                canonical_node_to_id.setdefault(metapath, len(canonical_node_to_id))

        for child, parent in iter_edge_pairs_auto(edges_path):
            key = key_fn(child)
            shard_name = shard_name_for(key, child)
            edge_shard_path = os.path.join(shard_dir, f"{edge_prefix}_{shard_name}.bin")
            handle = get_edge_handle(edge_shard_path)
            child_id = canonical_node_to_id.get(child)
            if child_id is None:
                child_id = len(canonical_node_to_id)
                canonical_node_to_id[child] = child_id
            parent_id = canonical_node_to_id.get(parent)
            if parent_id is None:
                parent_id = len(canonical_node_to_id)
                canonical_node_to_id[parent] = parent_id
            handle.write_ids(child_id, parent_id)
            edge_counts[shard_name] += 1
            edges_processed += 1
            pp_edges.maybe_stop_and_dump()
            if edges_processed % PROGRESS_EVERY_EDGES == 0:
                elapsed = time.time() - t_edges0
                _log(
                    f"edge-pass(text) edges={edges_processed} "
                    f"open_handles={len(edge_handles)} peak_open_handles={edge_peak_open_handles} "
                    f"evictions={edge_handle_evictions} rate={_rate(edges_processed, elapsed):.0f}/s elapsed={elapsed:.1f}s"
                )

    for handle in edge_handles.values():
        handle.close()
    pp_edges.stop_and_dump()
    elapsed = time.time() - t_edges0
    _log(
        f"edge-pass done edges={edges_processed} files={len(edge_counts)} "
        f"peak_open_handles={edge_peak_open_handles} evictions={edge_handle_evictions} "
        f"rate={_rate(edges_processed, elapsed):.0f}/s elapsed={elapsed:.1f}s"
    )

    # Shared node-id dictionary for decoding binary edge shards in this shard directory.
    if os.path.exists(canonical_node_ids_path):
        copy_or_write_node_ids(
            os.path.join(shard_dir, "node_ids.tsv"),
            {},
            src_node_ids_path=canonical_node_ids_path,
        )
    else:
        # Legacy fallback where canonical node_ids.tsv is absent.
        copy_or_write_node_ids(
            os.path.join(shard_dir, "node_ids.tsv"),
            canonical_node_to_id,
            src_node_ids_path=None,
        )

    index_path = os.path.join(shard_dir, "index.tsv")
    with open(index_path, "w") as f_idx:
        # Keep first 3 columns compatible with existing multihop shard submitter for right shards.
        f_idx.write(f"{index_key_col}\tshard_path\tnode_count\tedge_shard_path\tedge_count\n")
        for shard_name in sorted(shard_node_paths):
            key = shard_name_to_key[shard_name]
            node_shard = shard_node_paths[shard_name]
            edge_shard = os.path.join(shard_dir, f"{edge_prefix}_{shard_name}.bin")
            f_idx.write(
                f"{key}\t{node_shard}\t{node_counts[shard_name]}\t{edge_shard}\t{edge_counts.get(shard_name, 0)}\n"
            )

    prov = {
        "dag_dir": dag_dir,
        "shard_dir": shard_dir,
        "node_prefix": node_prefix,
        "edge_prefix": edge_prefix,
        "index_key_col": index_key_col,
        "shard_count": len(shard_node_paths),
        "node_count_total": int(sum(node_counts.values())),
        "edge_count_total": int(sum(edge_counts.values())),
        "max_rows": max_rows,
        "profile_phases": sorted(enabled_phases),
        "profile_duration_sec": profile_duration_sec,
        "timestamp": time.time(),
    }
    with open(os.path.join(shard_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
    _log(
        f"done shard_count={len(shard_node_paths)} node_total={prov['node_count_total']} "
        f"edge_total={prov['edge_count_total']} elapsed={time.time() - t0:.1f}s"
    )
    return prov


def materialize_right_shards(dag_dir: str, max_rows: int | None = None) -> dict:
    return _materialize_shards(
        dag_dir=dag_dir,
        shard_dir_name="shards_right",
        node_prefix="nhop",
        edge_prefix="nhop_edges",
        key_fn=end_type,
        index_key_col="join_type",
        max_rows=DEFAULT_RIGHT_MAX_ROWS if max_rows is None else max_rows,
        profile_phases=None,
        profile_duration_sec=0.0,
    )


def materialize_left_shards(dag_dir: str, max_rows: int | None = None) -> dict:
    return _materialize_shards(
        dag_dir=dag_dir,
        shard_dir_name="shards_left",
        node_prefix="onehop_start",
        edge_prefix="onehop_edges_start",
        key_fn=start_type,
        index_key_col="start_type",
        max_rows=max_rows,
        profile_phases=None,
        profile_duration_sec=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize persistent DAG shard layouts")
    parser.add_argument("--dag-dir", required=True, help="DAG directory containing nodes.tsv and edges.tsv")
    parser.add_argument("--right", action="store_true", help="Write right-sharded nhop artifacts")
    parser.add_argument("--left", action="store_true", help="Write left-sharded onehop artifacts")
    parser.add_argument("--max-rows", type=int, default=None, help=f"Max nodes per shard (split into parts if exceeded). Default for right shards: {DEFAULT_RIGHT_MAX_ROWS}")
    parser.add_argument(
        "--profile-phase",
        action="append",
        choices=["precount", "node-pass", "edge-pass", "all"],
        default=[],
        help="Enable cProfile for selected phase(s). Repeatable; use 'all' for all phases.",
    )
    parser.add_argument(
        "--profile-duration-sec",
        type=float,
        default=0.0,
        help="If >0, stop profiling after this many seconds within each profiled phase and dump stats immediately.",
    )
    args = parser.parse_args()
    if not args.right and not args.left:
        raise ValueError("Specify at least one of --right or --left")

    if "all" in args.profile_phase:
        profile_phases = {"precount", "node-pass", "edge-pass"}
    else:
        profile_phases = set(args.profile_phase)

    results = {}
    if args.right:
        results["right"] = _materialize_shards(
            dag_dir=args.dag_dir,
            shard_dir_name="shards_right",
            node_prefix="nhop",
            edge_prefix="nhop_edges",
            key_fn=end_type,
            index_key_col="join_type",
            max_rows=DEFAULT_RIGHT_MAX_ROWS if args.max_rows is None else args.max_rows,
            profile_phases=profile_phases,
            profile_duration_sec=args.profile_duration_sec,
        )
    if args.left:
        results["left"] = _materialize_shards(
            dag_dir=args.dag_dir,
            shard_dir_name="shards_left",
            node_prefix="onehop_start",
            edge_prefix="onehop_edges_start",
            key_fn=start_type,
            index_key_col="start_type",
            max_rows=args.max_rows,
            profile_phases=profile_phases,
            profile_duration_sec=args.profile_duration_sec,
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
