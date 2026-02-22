#!/usr/bin/env python3
"""
Pass B reduce worker: merge aggregated shards and write aggregated_nhop_counts.json.
"""

import argparse
import json
import os
import pickle
import glob
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare grouping Pass B reducer")
    parser.add_argument("--tmp-dir", required=True, help="Temp directory with shard files")
    parser.add_argument("--results-dir", required=True, help="Results directory for output")
    parser.add_argument("--n-hops", type=int, required=True, help="Number of hops")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    meta_path = os.path.join(args.tmp_dir, "explicit_counts_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    shard_files = sorted(glob.glob(os.path.join(args.tmp_dir, "agg_*.pkl")))
    if not shard_files:
        raise FileNotFoundError(f"No agg shards found in {args.tmp_dir}")

    aggregated_counts = {}
    for shard_path in shard_files:
        with open(shard_path, "rb") as f:
            shard = pickle.load(f)
        for path, count in shard.items():
            aggregated_counts[path] = aggregated_counts.get(path, 0) + count

    result = {
        "_metadata": {
            "created_at": datetime.now().isoformat(),
            "n_hops": args.n_hops,
            "num_result_files": meta.get("num_result_files"),
            "num_explicit_paths": meta.get("num_explicit_paths"),
            "num_aggregated_paths": len(aggregated_counts),
            "num_agg_shards": len(shard_files),
        },
        "counts": aggregated_counts,
    }

    output_path = os.path.join(args.results_dir, "aggregated_nhop_counts.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✓ Wrote aggregated counts to {output_path}")


if __name__ == "__main__":
    main()
