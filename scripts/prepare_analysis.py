#!/usr/bin/env python3
"""
Initialize manifest and directories for parallel 3-hop metapath analysis.

This script:
1. Loads matrices to determine total count
2. Creates output directories
3. Initializes manifest.json with all jobs in "pending" status

Usage:
    uv run python scripts/prepare_analysis.py
"""

import os
import json
from datetime import datetime
import sys

# Add parent directory to path to import from analyze_3hop_overlap
sys.path.insert(0, os.path.dirname(__file__))
from analyze_3hop_overlap import load_node_types, build_matrices, build_matrix_list


def prepare_analysis():
    """Prepare for parallel analysis run."""
    # Input data paths - SINGLE SOURCE OF TRUTH
    #nodes_file = "/projects/sequence_analysis/vol3/bizon/sub/pathfilter/scripts/metapaths/input/nodes.jsonl"
    #edges_file = "/projects/sequence_analysis/vol3/bizon/sub/pathfilter/scripts/metapaths/input/edges.jsonl"

    nodes_file = "/projects/sequence_analysis/vol3/bizon/graphs/rbn_6f3_human_curated_merged_cliques/nodes.jsonl"
    edges_file = "/projects/sequence_analysis/vol3/bizon/graphs/rbn_6f3_human_curated_merged_cliques/edges.jsonl"

    print("=" * 80)
    print("PREPARING PARALLEL 3-HOP ANALYSIS")
    print("=" * 80)
    print(f"\nInput data:")
    print(f"  Nodes: {nodes_file}")
    print(f"  Edges: {edges_file}")

    # Load matrices using existing infrastructure from analyze_3hop_overlap.py
    print("\nLoading graph data and building matrices...")
    node_types = load_node_types(nodes_file)
    matrices = build_matrices(edges_file, node_types)

    # Build extended matrix list with forward and reverse directions
    all_matrices, _ = build_matrix_list(matrices)

    num_matrices = len(all_matrices)
    print(f"\nTotal Matrix1 jobs to process: {num_matrices}")

    # Create output directories
    results_dir = "results"
    logs_dir = "logs"

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
            "nodes_file": nodes_file,
            "edges_file": edges_file,
            "created_at": datetime.now().isoformat(),
            "total_jobs": num_matrices
        }
    }

    for i in range(num_matrices):
        src_type, pred, tgt_type, matrix, direction = all_matrices[i]
        matrix_nvals = matrix.nvals

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
            "direction": direction
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
    print(f"  2. Run: uv run python scripts/orchestrate_3hop_analysis.py")
    print()


if __name__ == "__main__":
    prepare_analysis()
