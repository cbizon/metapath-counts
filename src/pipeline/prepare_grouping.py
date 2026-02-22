#!/usr/bin/env python3
"""
Prepare distributed grouping by type pairs.

Creates one job per (src_type, tgt_type) pair, which will process
all predicates and directions between those types.

Also precomputes aggregated path counts from matrix metadata for use
during grouping (so we don't need to read all result files to get
global counts for aggregated paths like Entity|related_to|F|Entity).

Usage:
    uv run python scripts/prepare_grouping.py --n-hops 1
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

from library import expand_metapath_to_variants, is_pseudo_type


def extract_type_pairs_from_aggregated_paths(aggregated_counts):
    """Extract unique (src_type, tgt_type) pairs from aggregated path counts.

    This extracts type pairs from ALL hierarchical variants, not just explicit types.
    For example, if explicit data has SmallMolecule|treats|F|Disease, this will
    also create type pairs for (ChemicalEntity, Disease), (ChemicalEntity, DiseaseOrPhenotypicFeature), etc.

    Args:
        aggregated_counts: Dict mapping path string to count (from precompute_aggregated_counts)

    Returns:
        Sorted list of (type1, type2) tuples (alphabetically ordered)
    """
    print(f"\nExtracting type pairs from {len(aggregated_counts)} aggregated paths...")

    type_pairs = set()

    for path in aggregated_counts.keys():
        parts = path.split('|')
        if len(parts) != 4:
            continue

        src_type, pred, direction, tgt_type = parts

        # Skip pseudo-types: their counts are already captured by constituent type jobs
        if is_pseudo_type(src_type) or is_pseudo_type(tgt_type):
            continue

        # Store as sorted tuple to treat (A,B) same as (B,A)
        pair = tuple(sorted([src_type, tgt_type]))
        type_pairs.add(pair)

    print(f"Found {len(type_pairs)} unique type pairs (including hierarchical)")

    return sorted(type_pairs)


def precompute_type_node_counts(matrices_dir):
    """Precompute node counts per type (explicit and hierarchical).

    Each node is assigned to exactly one explicit type (or pseudo-type),
    so summing child type counts gives parent type counts without double-counting.

    Returns:
        Dict mapping type name to node count
    """
    manifest_path = os.path.join(matrices_dir, "manifest.json")

    with open(manifest_path, 'r') as f:
        matrix_manifest = json.load(f)

    # Collect explicit type node counts from matrix metadata
    # Each matrix has (src_type, nrows) and (tgt_type, ncols)
    explicit_type_counts = {}  # type -> node count

    for matrix_info in matrix_manifest.get("matrices", []):
        src_type = matrix_info.get("src_type")
        tgt_type = matrix_info.get("tgt_type")
        nrows = matrix_info.get("nrows", 0)
        ncols = matrix_info.get("ncols", 0)

        if src_type:
            # Take max in case of inconsistencies (shouldn't happen)
            explicit_type_counts[src_type] = max(explicit_type_counts.get(src_type, 0), nrows)
        if tgt_type:
            explicit_type_counts[tgt_type] = max(explicit_type_counts.get(tgt_type, 0), ncols)

    print(f"  Found {len(explicit_type_counts)} explicit types with node counts")

    # Aggregate to hierarchical types
    # For each explicit type, add its count to all ancestor types
    aggregated_type_counts = defaultdict(int)

    for explicit_type, count in explicit_type_counts.items():
        # Get all ancestors (includes self)
        from library.hierarchy import get_type_ancestors
        ancestors = get_type_ancestors(explicit_type)

        for ancestor in ancestors:
            aggregated_type_counts[ancestor] += count

    print(f"  Aggregated to {len(aggregated_type_counts)} hierarchical types")

    return dict(aggregated_type_counts)


def precompute_aggregated_nhop_counts(results_dir, output_path, n_hops):
    """Precompute aggregated path counts from result files.

    Collects explicit counts for both the N-hop predictor paths (col 0/1) and
    the 1-hop predicted paths (col 2/3) from result files, then expands each
    to all hierarchical variants and sums counts.

    This single file serves both lookups during grouping:
      - predictor count: aggregated_nhop_counts[nhop_path]
      - target count:    aggregated_nhop_counts[onehop_path]

    Using result files (not matrix metadata) ensures pseudo-type paths are
    counted once, not once per constituent leaf type.

    Args:
        results_dir: Directory containing result files (results_matrix1_*.tsv)
        output_path: Path to write the aggregated counts JSON
        n_hops: Number of hops (for logging)

    Returns:
        Dict mapping path variant to global count (covers both N-hop and 1-hop paths)
    """
    import glob

    print(f"\nPrecomputing aggregated path counts from result files...")

    # Find all result files
    pattern = os.path.join(results_dir, "results_matrix1_*.tsv")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        print(f"  WARNING: No result files found matching {pattern}")
        return {}

    print(f"  Found {len(result_files)} result files")

    # Collect explicit counts from both predictor (col 0/1) and predicted (col 2/3)
    explicit_counts = {}  # path -> count

    files_processed = 0
    rows_processed = 0

    for file_path in result_files:
        files_processed += 1

        with open(file_path, 'r') as f:
            f.readline()  # skip header

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue

                nhop_path = parts[0]
                nhop_count = int(parts[1])
                onehop_path = parts[2]
                onehop_count = int(parts[3])
                rows_processed += 1

                if nhop_path not in explicit_counts:
                    explicit_counts[nhop_path] = nhop_count
                if onehop_path not in explicit_counts:
                    explicit_counts[onehop_path] = onehop_count

        if files_processed % 500 == 0:
            print(f"  Processed {files_processed}/{len(result_files)} files...")

    print(f"  Found {len(explicit_counts)} unique explicit paths (predictor + predicted)")
    print(f"  (from {rows_processed:,} total rows)")

    # Expand each path to hierarchical variants and aggregate
    aggregated_counts = defaultdict(int)

    for i, (path, count) in enumerate(explicit_counts.items()):
        if (i + 1) % 10000 == 0:
            print(f"  Expanding path {i+1}/{len(explicit_counts)}...")

        variants = expand_metapath_to_variants(path)

        for variant in variants:
            aggregated_counts[variant] += count

    print(f"  Expanded to {len(aggregated_counts)} aggregated paths")

    # Write to file
    result = {
        "_metadata": {
            "created_at": datetime.now().isoformat(),
            "n_hops": n_hops,
            "num_result_files": len(result_files),
            "num_explicit_paths": len(explicit_counts),
            "num_aggregated_paths": len(aggregated_counts)
        },
        "counts": dict(aggregated_counts)
    }

    print(f"  Writing to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  ✓ Precomputed {len(aggregated_counts)} aggregated path counts")

    return dict(aggregated_counts)


def create_grouping_manifest(type_pairs, results_dir, n_hops):
    """Create manifest for distributed grouping.

    Args:
        type_pairs: List of (type1, type2) tuples
        results_dir: Directory containing result files
        n_hops: Number of hops
    """
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

    # Create job entry for each type pair
    for idx, (type1, type2) in enumerate(type_pairs):
        job_id = f"typepair_{idx:04d}"
        manifest[job_id] = {
            "index": idx,
            "type1": type1,
            "type2": type2,
            "status": "pending",
            "job_id": None,
            "attempts": 0,
            "memory_tier": 32,  # Start with 32GB
            "last_update": None,
            "error_type": None
        }

    # Save manifest
    print(f"\nWriting manifest to {manifest_path}...")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created {len(type_pairs)} grouping jobs")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare distributed grouping by type pairs'
    )
    parser.add_argument(
        '--n-hops',
        type=int,
        required=True,
        help='Number of hops (e.g., 1, 2, 3)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Results directory (default: results_{n_hops}hop)'
    )
    parser.add_argument(
        '--matrices-dir',
        type=str,
        default='matrices',
        help='Matrices directory (default: matrices)'
    )
    parser.add_argument(
        '--aggregated-counts-path',
        type=str,
        default=None,
        help='Path to precomputed aggregated_nhop_counts.json (skip precompute)'
    )
    parser.add_argument(
        '--skip-aggregated-precompute',
        action='store_true',
        help='Use existing aggregated_nhop_counts.json in results dir'
    )

    args = parser.parse_args()

    results_dir = args.results_dir or f"results_{args.n_hops}hop"

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if not os.path.exists(args.matrices_dir):
        raise FileNotFoundError(f"Matrices directory not found: {args.matrices_dir}")

    print("=" * 80)
    print(f"PREPARING DISTRIBUTED GROUPING FOR {args.n_hops}-HOP RESULTS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Matrices directory: {args.matrices_dir}")
    print(f"Granularity: Type pairs (src_type, tgt_type)")

    # Precompute type node counts (for total_possible calculation)
    print("\nPrecomputing type node counts...")
    type_node_counts = precompute_type_node_counts(args.matrices_dir)

    # Save type node counts
    type_counts_path = os.path.join(results_dir, "type_node_counts.json")
    with open(type_counts_path, 'w') as f:
        json.dump(type_node_counts, f, indent=2)
    print(f"  ✓ Saved type node counts to {type_counts_path}")

    # Precompute aggregated path counts from result files
    # (used for both predictor and target count lookups during grouping)
    aggregated_nhop_counts_path = os.path.join(results_dir, "aggregated_nhop_counts.json")
    if args.aggregated_counts_path or args.skip_aggregated_precompute:
        counts_path = args.aggregated_counts_path or aggregated_nhop_counts_path
        if not os.path.exists(counts_path):
            raise FileNotFoundError(f"Aggregated counts file not found: {counts_path}")
        with open(counts_path, 'r') as f:
            data = json.load(f)
        aggregated_counts = data.get("counts", data)
        print(f"  ✓ Loaded aggregated counts from {counts_path} ({len(aggregated_counts)} paths)")
    else:
        aggregated_counts = precompute_aggregated_nhop_counts(results_dir, aggregated_nhop_counts_path, args.n_hops)

    # Extract type pairs from aggregated paths (includes all hierarchical types)
    type_pairs = extract_type_pairs_from_aggregated_paths(aggregated_counts)

    # Create manifest
    manifest_path = create_grouping_manifest(type_pairs, results_dir, args.n_hops)

    print("\n" + "=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  uv run python scripts/orchestrate_grouping.py --n-hops {args.n_hops}")
    print()


if __name__ == "__main__":
    main()
