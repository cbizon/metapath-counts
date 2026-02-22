#!/usr/bin/env python3
"""
Pass B map worker: expand explicit paths to hierarchical variants and
accumulate aggregated counts for a shard.
"""

import argparse
import os
import pickle

from library import expand_metapath_to_variants


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare grouping Pass B map worker")
    parser.add_argument("--tmp-dir", required=True, help="Temp directory with shard files")
    parser.add_argument("--shard-index", type=int, required=True, help="Shard index")
    return parser.parse_args()


def main():
    args = parse_args()
    shard_path = os.path.join(args.tmp_dir, f"explicit_counts_shard_{args.shard_index:05d}.pkl")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Explicit counts shard not found: {shard_path}")

    with open(shard_path, "rb") as f:
        items = pickle.load(f)

    aggregated_counts = {}
    for path, count in items:
        variants = expand_metapath_to_variants(path)
        for variant in variants:
            aggregated_counts[variant] = aggregated_counts.get(variant, 0) + count

    output_path = os.path.join(args.tmp_dir, f"agg_{args.shard_index:05d}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(aggregated_counts, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
