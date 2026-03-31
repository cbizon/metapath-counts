#!/usr/bin/env python3
"""
Prepare distributed grouping by type pairs.

This stage no longer precomputes global hierarchical path counts. Instead it:
1. Loads explicit path counts produced by Pass A reduce
2. Saves hierarchical type node counts for total_possible calculations
3. Builds one grouping job per endpoint type pair shard
"""

import argparse
import json
import os
from datetime import datetime

from library.hierarchy import get_type_ancestors


def precompute_type_node_counts(matrices_dir):
    """Precompute node counts per type (explicit and hierarchical)."""
    manifest_path = os.path.join(matrices_dir, "manifest.json")

    with open(manifest_path, 'r') as f:
        matrix_manifest = json.load(f)

    explicit_type_counts = {}
    for matrix_info in matrix_manifest.get("matrices", []):
        src_type = matrix_info.get("src_type")
        tgt_type = matrix_info.get("tgt_type")
        nrows = matrix_info.get("nrows", 0)
        ncols = matrix_info.get("ncols", 0)

        if src_type:
            explicit_type_counts[src_type] = max(explicit_type_counts.get(src_type, 0), nrows)
        if tgt_type:
            explicit_type_counts[tgt_type] = max(explicit_type_counts.get(tgt_type, 0), ncols)

    aggregated_type_counts = {}
    for explicit_type, count in explicit_type_counts.items():
        for ancestor in get_type_ancestors(explicit_type):
            aggregated_type_counts[ancestor] = aggregated_type_counts.get(ancestor, 0) + count

    return aggregated_type_counts


def load_type_pairs(tmp_dir):
    """Load type pairs from Reduce A output."""
    typepairs_path = os.path.join(tmp_dir, "typepairs.json")
    if not os.path.exists(typepairs_path):
        raise FileNotFoundError(
            f"Type pair metadata not found: {typepairs_path}\n"
            "Run precompute_aggregated_counts_slurm.py first."
        )

    with open(typepairs_path, "r") as f:
        data = json.load(f)

    return [(item["type1"], item["type2"]) for item in data]


def create_grouping_manifest(type_pairs, results_dir, n_hops):
    """Create manifest for distributed grouping."""
    manifest_path = os.path.join(results_dir, "grouping_manifest.json")

    manifest = {
        "_metadata": {
            "n_hops": n_hops,
            "results_dir": results_dir,
            "total_jobs": len(type_pairs),
            "granularity": "type_pairs",
            "created_at": datetime.now().isoformat()
        }
    }

    for idx, (type1, type2) in enumerate(type_pairs):
        job_id = f"typepair_{idx:04d}"
        manifest[job_id] = {
            "index": idx,
            "type1": type1,
            "type2": type2,
            "status": "pending",
            "job_id": None,
            "attempts": 0,
            "memory_tier": 64,
            "cpu_tier": 1,
            "last_update": None,
            "error_type": None
        }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created {len(type_pairs)} grouping jobs")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description='Prepare distributed grouping by type pairs')
    parser.add_argument('--n-hops', type=int, required=True, help='Number of hops (e.g., 1, 2, 3)')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory (default: results_{n_hops}hop)')
    parser.add_argument('--matrices-dir', type=str, default='matrices', help='Matrices directory (default: matrices)')
    parser.add_argument('--tmp-dir', type=str, default=None, help='Temp directory from explicit-count precompute')

    args = parser.parse_args()

    results_dir = args.results_dir or f"results_{args.n_hops}hop"
    tmp_dir = args.tmp_dir or os.path.join(results_dir, "_tmp_prepare_grouping")

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not os.path.exists(args.matrices_dir):
        raise FileNotFoundError(f"Matrices directory not found: {args.matrices_dir}")
    if not os.path.exists(tmp_dir):
        raise FileNotFoundError(f"Precompute temp directory not found: {tmp_dir}")

    print("=" * 80)
    print(f"PREPARING DISTRIBUTED GROUPING FOR {args.n_hops}-HOP RESULTS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Matrices directory: {args.matrices_dir}")
    print(f"Precompute temp dir: {tmp_dir}")
    print(f"Granularity: Type pairs (src_type, tgt_type)")

    print("\nPrecomputing type node counts...")
    type_node_counts = precompute_type_node_counts(args.matrices_dir)
    type_counts_path = os.path.join(results_dir, "type_node_counts.json")
    with open(type_counts_path, 'w') as f:
        json.dump(type_node_counts, f, indent=2)
    print(f"  ✓ Saved type node counts to {type_counts_path}")

    print("\nLoading type pairs from explicit-count shards...")
    type_pairs = load_type_pairs(tmp_dir)
    print(f"  ✓ Loaded {len(type_pairs)} type pairs")

    manifest_path = create_grouping_manifest(type_pairs, results_dir, args.n_hops)

    print("\n" + "=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  uv run python src/pipeline/orchestrate_grouping.py --n-hops {args.n_hops}")
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
