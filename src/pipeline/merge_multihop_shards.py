#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time

from pipeline.dag_edge_io import (
    BinaryEdgeWriter,
    iter_edge_id_pairs_from_bin,
    iter_edge_pairs_auto,
    load_node_ids,
    resolve_node_ids_path,
    write_node_ids,
)


def shard_dirs_from_manifest(out_dir: str):
    manifest = os.path.join(out_dir, "shards", "submitted_jobs.tsv")
    shard_root = os.path.join(out_dir, "shard_jobs")
    with open(manifest, "r") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row:
                continue
            if row[0] == "job_id":
                continue
            _job_id, _join_type, shard_path, _nhop_count = row[:4]
            shard_base = os.path.basename(shard_path)
            if shard_base.endswith(".tsv"):
                shard_base = shard_base[:-4]
            shard_key = shard_base[5:] if shard_base.startswith("nhop_") else shard_base
            yield os.path.join(shard_root, shard_key)


def _log(msg: str) -> None:
    print(f"[merge_multihop_shards] {msg}", flush=True)


def merge_shards(out_dir: str) -> None:
    t0 = time.time()
    edges_out = os.path.join(out_dir, "edges.bin")
    node_ids_out = os.path.join(out_dir, "node_ids.tsv.gz")

    shard_dirs = list(shard_dirs_from_manifest(out_dir))
    _log(f"start out_dir={out_dir} shards={len(shard_dirs)}")

    node_to_id: dict[str, int] = {}
    shard_local_to_global: dict[str, list[int]] = {}
    for i, shard_dir in enumerate(shard_dirs, start=1):
        _local_node_to_id, local_id_to_node = load_node_ids(resolve_node_ids_path(shard_dir))
        local_to_global = [0] * len(local_id_to_node)
        for local_id, metapath in enumerate(local_id_to_node):
            global_id = node_to_id.get(metapath)
            if global_id is None:
                global_id = len(node_to_id)
                node_to_id[metapath] = global_id
            local_to_global[local_id] = global_id
        if os.path.exists(os.path.join(shard_dir, "edges.bin")):
            shard_local_to_global[shard_dir] = local_to_global
        if i % 10 == 0 or i == len(shard_dirs):
            _log(f"indexed shard node_ids {i}/{len(shard_dirs)} unique_nodes={len(node_to_id)}")

    write_node_ids(node_ids_out, node_to_id)
    _log(f"wrote node_ids {node_ids_out} nodes={len(node_to_id)}")

    added_from_edges = False
    edge_count = 0
    with BinaryEdgeWriter(edges_out) as bw:
        for i, shard_dir in enumerate(shard_dirs, start=1):
            shard_edges_bin = os.path.join(shard_dir, "edges.bin")
            shard_edges_tsv = os.path.join(shard_dir, "edges.tsv")
            if os.path.exists(shard_edges_bin):
                local_to_global = shard_local_to_global[shard_dir]
                for child_local_id, parent_local_id in iter_edge_id_pairs_from_bin(shard_edges_bin):
                    bw.write_ids(local_to_global[child_local_id], local_to_global[parent_local_id])
                    edge_count += 1
                    if edge_count % 5_000_000 == 0:
                        _log(f"merged edges={edge_count}")
                if i % 10 == 0 or i == len(shard_dirs):
                    _log(f"processed shard edges {i}/{len(shard_dirs)} total_edges={edge_count}")
                continue

            # Legacy TSV fallback.
            for child, parent in iter_edge_pairs_auto(shard_edges_tsv):
                child_id = node_to_id.get(child)
                if child_id is None:
                    child_id = len(node_to_id)
                    node_to_id[child] = child_id
                    added_from_edges = True
                parent_id = node_to_id.get(parent)
                if parent_id is None:
                    parent_id = len(node_to_id)
                    node_to_id[parent] = parent_id
                    added_from_edges = True
                bw.write_ids(child_id, parent_id)
                edge_count += 1
                if edge_count % 5_000_000 == 0:
                    _log(f"merged edges={edge_count}")
            if i % 10 == 0 or i == len(shard_dirs):
                _log(f"processed shard edges {i}/{len(shard_dirs)} total_edges={edge_count}")

    # If parent-only nodes were discovered from shard edges, rewrite node_ids once.
    if added_from_edges:
        write_node_ids(node_ids_out, node_to_id)
        _log(f"rewrote node_ids due to legacy TSV edge nodes nodes={len(node_to_id)}")
    _log(f"done edges={edge_count} elapsed={time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multihop shard outputs to canonical node_ids.tsv + edges.bin")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    merge_shards(args.out_dir)


if __name__ == "__main__":
    main()
