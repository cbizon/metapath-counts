#!/usr/bin/env python3
"""
Pass A reduce worker: merge first-seen shards into explicit counts and build
type-pair explicit path shards for grouping jobs.
"""

import argparse
import json
import os
import pickle
import glob
import time
from datetime import datetime
from collections import defaultdict

from library import get_type_ancestors, parse_metapath


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare grouping Pass A reducer")
    parser.add_argument("--tmp-dir", required=True, help="Temp directory with shard files")
    parser.add_argument("--results-dir", required=True, help="Results directory for metadata")
    parser.add_argument("--n-hops", type=int, required=True, help="Number of hops")
    return parser.parse_args()


def load_files_list(tmp_dir):
    files_list_path = os.path.join(tmp_dir, "files.txt")
    with open(files_list_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def get_rss_mb():
    """Return resident memory in MB from /proc/self/status when available."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def log_progress(stage, start_time, **metrics):
    """Emit a timestamped progress line with elapsed time and RSS."""
    elapsed = time.time() - start_time
    rss_mb = get_rss_mb()
    metric_parts = [f"{key}={value}" for key, value in metrics.items()]
    metric_parts.append(f"elapsed_s={elapsed:.1f}")
    if rss_mb is not None:
        metric_parts.append(f"rss_mb={rss_mb:.1f}")
    print(f"[reduceA] {stage}: " + " ".join(metric_parts), flush=True)


def write_progress_file(tmp_dir, stage, start_time, **metrics):
    """Write a small JSON progress record so OOM/failure leaves breadcrumbs."""
    payload = {
        "stage": stage,
        "updated_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - start_time, 1),
        "rss_mb": None if get_rss_mb() is None else round(get_rss_mb(), 1),
    }
    payload.update(metrics)
    with open(os.path.join(tmp_dir, "reduceA_progress.json"), "w") as f:
        json.dump(payload, f, indent=2)


def get_path_endpoint_typepairs(path):
    """Return all ancestor endpoint type pairs for a metapath.

    Each explicit path is assigned to every grouping job whose endpoint types are
    ancestors of the explicit endpoint types. This is much smaller than expanding
    the full metapath hierarchy, and gives grouping jobs exact local explicit counts.
    """
    nodes, _, _ = parse_metapath(path)
    src_type = nodes[0]
    tgt_type = nodes[-1]

    src_ancestors = get_type_ancestors(src_type)
    tgt_ancestors = get_type_ancestors(tgt_type)

    pairs = set()
    for src_ancestor in src_ancestors:
        for tgt_ancestor in tgt_ancestors:
            pairs.add(tuple(sorted((src_ancestor, tgt_ancestor))))
    return pairs


def main():
    args = parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)
    start_time = time.time()

    shard_files = sorted(glob.glob(os.path.join(args.tmp_dir, "first_seen_*.pkl")))
    if not shard_files:
        raise FileNotFoundError(f"No first_seen shards found in {args.tmp_dir}")
    log_progress("start", start_time, shard_files=len(shard_files))
    write_progress_file(args.tmp_dir, "start", start_time, shard_files=len(shard_files))

    merged = {}
    for idx, shard_path in enumerate(shard_files, start=1):
        with open(shard_path, "rb") as f:
            shard = pickle.load(f)
        for path, (file_index, count) in shard.items():
            if path not in merged or file_index < merged[path][0]:
                merged[path] = (file_index, count)
        if idx % 10 == 0 or idx == len(shard_files):
            log_progress("merge_first_seen", start_time, shards_done=idx, merged_paths=len(merged))
            write_progress_file(args.tmp_dir, "merge_first_seen", start_time, shards_done=idx, merged_paths=len(merged))

    explicit_counts = {path: count for path, (_, count) in merged.items()}
    log_progress("explicit_counts_ready", start_time, explicit_paths=len(explicit_counts))
    write_progress_file(args.tmp_dir, "explicit_counts_ready", start_time, explicit_paths=len(explicit_counts))
    explicit_counts_path = os.path.join(args.tmp_dir, "explicit_counts.pkl")
    with open(explicit_counts_path, "wb") as f:
        pickle.dump(explicit_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    log_progress("explicit_counts_written", start_time, explicit_counts_path=explicit_counts_path)
    write_progress_file(args.tmp_dir, "explicit_counts_written", start_time, explicit_counts_path=explicit_counts_path)

    typepair_dir = os.path.join(args.tmp_dir, "typepair_explicit_paths")
    os.makedirs(typepair_dir, exist_ok=True)

    typepair_to_items = defaultdict(list)
    total_assignments = 0
    for idx, (path, count) in enumerate(explicit_counts.items(), start=1):
        for typepair in get_path_endpoint_typepairs(path):
            typepair_to_items[typepair].append((path, count))
            total_assignments += 1
        if idx % 100000 == 0 or idx == len(explicit_counts):
            log_progress(
                "fanout_typepairs",
                start_time,
                explicit_done=idx,
                explicit_total=len(explicit_counts),
                unique_typepairs=len(typepair_to_items),
                total_assignments=total_assignments,
            )
            write_progress_file(
                args.tmp_dir,
                "fanout_typepairs",
                start_time,
                explicit_done=idx,
                explicit_total=len(explicit_counts),
                unique_typepairs=len(typepair_to_items),
                total_assignments=total_assignments,
            )

    typepairs = []
    for idx, ((type1, type2), items) in enumerate(typepair_to_items.items(), start=1):
        shard_path = os.path.join(typepair_dir, f"{type1}__{type2}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        typepairs.append((type1, type2, len(items)))
        if idx % 100 == 0 or idx == len(typepair_to_items):
            log_progress("write_typepair_shards", start_time, typepairs_written=idx, typepairs_total=len(typepair_to_items))
            write_progress_file(
                args.tmp_dir,
                "write_typepair_shards",
                start_time,
                typepairs_written=idx,
                typepairs_total=len(typepair_to_items),
            )

    files_list = load_files_list(args.tmp_dir)
    meta = {
        "created_at": datetime.now().isoformat(),
        "n_hops": args.n_hops,
        "num_result_files": len(files_list),
        "num_explicit_paths": len(explicit_counts),
        "num_typepairs": len(typepairs),
    }
    meta_path = os.path.join(args.tmp_dir, "explicit_counts_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    typepairs_path = os.path.join(args.tmp_dir, "typepairs.json")
    with open(typepairs_path, "w") as f:
        json.dump(
            [
                {"type1": type1, "type2": type2, "num_paths": num_paths}
                for type1, type2, num_paths in sorted(typepairs)
            ],
            f,
            indent=2,
        )

    log_progress("complete", start_time, explicit_paths=len(explicit_counts), typepairs=len(typepairs))
    write_progress_file(args.tmp_dir, "complete", start_time, explicit_paths=len(explicit_counts), typepairs=len(typepairs))
    print(f"✓ Reduced {len(shard_files)} shards into {len(explicit_counts)} explicit paths")
    print(f"✓ Wrote explicit type-pair shards for {len(typepairs)} grouping jobs to {typepair_dir}")


if __name__ == "__main__":
    main()
