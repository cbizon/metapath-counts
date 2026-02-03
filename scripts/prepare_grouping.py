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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metapath_counts import expand_metapath_to_variants


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
        from metapath_counts.hierarchy import get_type_ancestors
        ancestors = get_type_ancestors(explicit_type)

        for ancestor in ancestors:
            aggregated_type_counts[ancestor] += count

    print(f"  Aggregated to {len(aggregated_type_counts)} hierarchical types")

    return dict(aggregated_type_counts)


def precompute_aggregated_counts(matrices_dir, output_path):
    """Precompute aggregated path counts from matrix metadata.

    For each matrix (src_type, pred, dir, tgt_type) with count N,
    expand to all hierarchical variants and sum counts.

    This allows grouping workers to look up global counts for aggregated
    paths like Entity|related_to|F|Entity without reading all result files.

    Args:
        matrices_dir: Directory containing matrices/manifest.json
        output_path: Path to write the aggregated counts JSON

    Returns:
        Dict mapping path string to global count
    """
    manifest_path = os.path.join(matrices_dir, "manifest.json")

    print(f"\nPrecomputing aggregated path counts from {manifest_path}...")

    with open(manifest_path, 'r') as f:
        matrix_manifest = json.load(f)

    # First pass: collect explicit path counts
    explicit_counts = {}  # path -> count
    matrices = matrix_manifest.get("matrices", [])

    print(f"  Processing {len(matrices)} matrices...")

    for matrix_info in matrices:
        src_type = matrix_info.get("src_type")
        pred = matrix_info.get("predicate")
        direction = matrix_info.get("direction", "F")  # Default to forward
        tgt_type = matrix_info.get("tgt_type")
        count = matrix_info.get("nvals", 0)

        if not all([src_type, pred, tgt_type]):
            continue

        # Build explicit path string
        path = f"{src_type}|{pred}|{direction}|{tgt_type}"
        explicit_counts[path] = count

    print(f"  Found {len(explicit_counts)} explicit paths")

    # Second pass: expand each path to variants and aggregate
    aggregated_counts = defaultdict(int)

    for i, (path, count) in enumerate(explicit_counts.items()):
        if (i + 1) % 1000 == 0:
            print(f"  Expanding path {i+1}/{len(explicit_counts)}...")

        # Expand to hierarchical variants
        variants = expand_metapath_to_variants(path)

        for variant in variants:
            aggregated_counts[variant] += count

    print(f"  Expanded to {len(aggregated_counts)} aggregated paths")

    # Write to file
    result = {
        "_metadata": {
            "created_at": datetime.now().isoformat(),
            "num_explicit_paths": len(explicit_counts),
            "num_aggregated_paths": len(aggregated_counts)
        },
        "counts": dict(aggregated_counts)
    }

    print(f"  Writing to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  ✓ Precomputed {len(aggregated_counts)} aggregated path counts")

    return aggregated_counts


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

    # Precompute aggregated path counts (needed for type pair extraction)
    aggregated_counts_path = os.path.join(results_dir, "aggregated_path_counts.json")
    aggregated_counts = precompute_aggregated_counts(args.matrices_dir, aggregated_counts_path)

    # Extract type pairs from AGGREGATED paths (includes all hierarchical types)
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
