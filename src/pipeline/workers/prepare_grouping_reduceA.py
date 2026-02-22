#!/usr/bin/env python3
"""
Pass A reduce worker: merge first-seen shards into explicit counts and split
into Pass B shards.
"""

import argparse
import json
import math
import os
import pickle
import glob
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare grouping Pass A reducer")
    parser.add_argument("--tmp-dir", required=True, help="Temp directory with shard files")
    parser.add_argument("--passb-shards", type=int, required=True, help="Shard count for Pass B")
    parser.add_argument("--results-dir", required=True, help="Results directory for metadata")
    parser.add_argument("--n-hops", type=int, required=True, help="Number of hops")
    return parser.parse_args()


def load_files_list(tmp_dir):
    files_list_path = os.path.join(tmp_dir, "files.txt")
    with open(files_list_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def main():
    args = parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)

    shard_files = sorted(glob.glob(os.path.join(args.tmp_dir, "first_seen_*.pkl")))
    if not shard_files:
        raise FileNotFoundError(f"No first_seen shards found in {args.tmp_dir}")

    merged = {}
    for shard_path in shard_files:
        with open(shard_path, "rb") as f:
            shard = pickle.load(f)
        for path, (file_index, count) in shard.items():
            if path not in merged or file_index < merged[path][0]:
                merged[path] = (file_index, count)

    explicit_counts = {path: count for path, (_, count) in merged.items()}
    explicit_counts_path = os.path.join(args.tmp_dir, "explicit_counts.pkl")
    with open(explicit_counts_path, "wb") as f:
        pickle.dump(explicit_counts, f, protocol=pickle.HIGHEST_PROTOCOL)

    items = list(explicit_counts.items())
    if args.passb_shards <= 0:
        raise ValueError("passb_shards must be > 0")
    shard_size = math.ceil(len(items) / args.passb_shards) if items else 1

    for i in range(args.passb_shards):
        start = i * shard_size
        end = min(start + shard_size, len(items))
        shard_items = items[start:end]
        shard_path = os.path.join(args.tmp_dir, f"explicit_counts_shard_{i:05d}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(shard_items, f, protocol=pickle.HIGHEST_PROTOCOL)

    files_list = load_files_list(args.tmp_dir)
    meta = {
        "created_at": datetime.now().isoformat(),
        "n_hops": args.n_hops,
        "num_result_files": len(files_list),
        "num_explicit_paths": len(explicit_counts),
        "passb_shards": args.passb_shards,
    }
    meta_path = os.path.join(args.tmp_dir, "explicit_counts_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ Reduced {len(shard_files)} shards into {len(explicit_counts)} explicit paths")
    print(f"✓ Wrote {args.passb_shards} Pass B shards to {args.tmp_dir}")


if __name__ == "__main__":
    main()
