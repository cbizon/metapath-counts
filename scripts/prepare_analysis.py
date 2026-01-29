#!/usr/bin/env python3
"""
Initialize manifest and directories for parallel N-hop metapath analysis.

This script:
1. Reads pre-built matrix manifest
2. Creates output directories
3. Initializes manifest.json with all jobs in "pending" status

Usage:
    # 3-hop analysis (default)
    uv run python scripts/prepare_analysis.py --matrices-dir matrices

    # Custom N-hop analysis
    uv run python scripts/prepare_analysis.py --matrices-dir matrices --n-hops 2
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from metapath_counts import get_symmetric_predicates


def build_matrix_list_from_manifest(matrices_dir: str):
    """
    Build directed matrix list from pre-built matrix manifest WITHOUT loading matrices.

    Returns list of (src_type, pred, tgt_type, nvals, direction) tuples.

    Args:
        matrices_dir: Directory containing matrices and manifest.json

    Returns:
        List of (src_type, pred, tgt_type, nvals, direction) tuples
    """
    manifest_path = Path(matrices_dir) / 'manifest.json'

    if not manifest_path.exists():
        raise FileNotFoundError(f"Matrix manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    symmetric_predicates = get_symmetric_predicates()

    all_matrices = []
    for mat_info in manifest['matrices']:
        src_type = mat_info['src_type']
        pred = mat_info['predicate']
        tgt_type = mat_info['tgt_type']
        nvals = mat_info['nvals']

        # Forward direction
        all_matrices.append((src_type, pred, tgt_type, nvals, 'F'))

        # Reverse direction (if not symmetric)
        if pred not in symmetric_predicates:
            all_matrices.append((tgt_type, pred, src_type, nvals, 'R'))

    return all_matrices


def prepare_analysis(matrices_dir: str, n_hops: int = 3):
    """Prepare for parallel analysis run.

    Args:
        matrices_dir: Directory with pre-built matrices
        n_hops: Number of hops to analyze (default: 3)
    """
    print("=" * 80)
    print(f"PREPARING PARALLEL {n_hops}-HOP ANALYSIS")
    print("=" * 80)
    print(f"\nInput:")
    print(f"  Pre-built matrices: {matrices_dir}")
    print(f"  N-hops: {n_hops}")

    # Read matrix manifest (fast, no matrix loading)
    print("\nReading pre-built matrix manifest...")
    all_matrices_metadata = build_matrix_list_from_manifest(matrices_dir)
    num_matrices = len(all_matrices_metadata)
    print(f"Found {num_matrices} directed matrices in manifest")

    # For 1-hop analysis, pre-filter to only canonical directions
    # This cuts the job count roughly in half
    if n_hops == 1:
        print(f"\n1-hop analysis: filtering to canonical directions only (src_type <= tgt_type)...")
        original_count = len(all_matrices_metadata)
        all_matrices_metadata = [
            (src, pred, tgt, nvals, d)
            for src, pred, tgt, nvals, d in all_matrices_metadata
            if src <= tgt  # Canonical direction
        ]
        filtered_count = len(all_matrices_metadata)
        print(f"Filtered from {original_count} to {filtered_count} jobs ({100*filtered_count/original_count:.1f}%)")
        num_matrices = filtered_count

    print(f"\nTotal Matrix1 jobs to process: {num_matrices}")

    # Create output directories with n_hops suffix
    results_dir = f"results_{n_hops}hop"
    logs_dir = f"logs_{n_hops}hop"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Created directories:")
    print(f"  - {results_dir}")
    print(f"  - {logs_dir}")

    # Initialize manifest
    manifest_path = os.path.join(results_dir, "manifest.json")

    print(f"\nInitializing manifest at {manifest_path}...")
    manifest = {
        "_metadata": {
            "matrices_dir": matrices_dir,
            "n_hops": n_hops,
            "created_at": datetime.now().isoformat(),
            "total_jobs": num_matrices
        }
    }

    for i in range(num_matrices):
        src_type, pred, tgt_type, matrix_nvals, direction = all_matrices_metadata[i]

        manifest[f"matrix1_{i:03d}"] = {
            "status": "pending",
            "memory_tier": 180,  # Matches most cluster nodes (~191GB available)
            "attempts": 0,
            "job_id": None,
            "last_update": datetime.now().isoformat(),
            "error_type": None,
            "matrix_nvals": matrix_nvals,
            "src_type": src_type,
            "pred": pred,
            "tgt_type": tgt_type,
            "direction": direction,
            "paths_completed": 0,
            "paths_failed": 0
        }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Initialized manifest for {num_matrices} jobs")

    # Print summary
    print("\n" + "=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal jobs: {num_matrices}")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. Review manifest.json")
    print(f"  2. Run: uv run python scripts/orchestrate_hop_analysis.py --n-hops {n_hops}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Initialize manifest and directories for parallel N-hop metapath analysis'
    )
    parser.add_argument('--matrices-dir', required=True,
                        help='Directory with pre-built matrices')
    parser.add_argument('--n-hops', type=int, default=3,
                        help='Number of hops to analyze (default: 3)')

    args = parser.parse_args()

    prepare_analysis(matrices_dir=args.matrices_dir, n_hops=args.n_hops)


if __name__ == "__main__":
    main()
