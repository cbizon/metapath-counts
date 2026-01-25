#!/usr/bin/env python3
"""
Initialize manifest and directories for parallel N-hop metapath analysis.

This script:
1. Loads matrices to determine total count
2. Creates output directories
3. Initializes manifest.json with all jobs in "pending" status

Usage:
    # 3-hop analysis (default)
    uv run python scripts/prepare_analysis.py --edges /path/to/edges.jsonl --nodes /path/to/nodes.jsonl

    # Custom N-hop analysis
    uv run python scripts/prepare_analysis.py --edges /path/to/edges.jsonl --nodes /path/to/nodes.jsonl --n-hops 2
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path
import sys
import yaml

# Add parent directory to path to import from analyze_hop_overlap
sys.path.insert(0, os.path.dirname(__file__))
from analyze_hop_overlap import load_node_types, build_matrices, build_matrix_list, load_prebuilt_matrices


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = 'config/type_expansion.yaml'

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config.get('type_expansion', {})


def prepare_analysis(nodes_file: str, edges_file: str, n_hops: int = 3, matrices_dir: str = None, config_path: str = None):
    """Prepare for parallel analysis run.

    Args:
        nodes_file: Path to KGX nodes file
        edges_file: Path to KGX edges file
        n_hops: Number of hops to analyze (default: 3)
        matrices_dir: Optional directory with pre-built matrices (faster)
        config_path: Optional path to configuration YAML file
    """
    # Load configuration
    type_config = load_config(config_path)

    print("=" * 80)
    print(f"PREPARING PARALLEL {n_hops}-HOP ANALYSIS")
    print("=" * 80)
    print(f"\nInput data:")
    print(f"  Nodes: {nodes_file}")
    print(f"  Edges: {edges_file}")
    print(f"  N-hops: {n_hops}")
    print(f"  Config: {config_path or 'config/type_expansion.yaml'}")
    if matrices_dir:
        print(f"  Pre-built matrices: {matrices_dir}")
    if type_config:
        print(f"  Type expansion: exclude {len(type_config.get('exclude_types', []))} types")

    # Load matrices (either prebuilt or from edges)
    if matrices_dir:
        print("\nLoading pre-built matrices...")
        matrices = load_prebuilt_matrices(matrices_dir)
    else:
        print("\nLoading graph data and building matrices...")
        node_types = load_node_types(nodes_file, config=type_config)
        matrices = build_matrices(edges_file, node_types)

    # Build extended matrix list with forward and reverse directions
    all_matrices, _ = build_matrix_list(matrices)

    num_matrices = len(all_matrices)
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
            "nodes_file": nodes_file,
            "edges_file": edges_file,
            "n_hops": n_hops,
            "created_at": datetime.now().isoformat(),
            "total_jobs": num_matrices
        }
    }

    # Add matrices_dir to metadata if provided
    if matrices_dir:
        manifest["_metadata"]["matrices_dir"] = matrices_dir

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
    parser.add_argument('--edges', required=True,
                        help='Path to edges.jsonl file')
    parser.add_argument('--nodes', required=True,
                        help='Path to nodes.jsonl file')
    parser.add_argument('--n-hops', type=int, default=3,
                        help='Number of hops to analyze (default: 3)')
    parser.add_argument('--matrices-dir', default=None,
                        help='Directory with pre-built matrices (for faster preparation)')
    parser.add_argument('--config', default='config/type_expansion.yaml',
                        help='Path to configuration YAML file (default: config/type_expansion.yaml)')

    args = parser.parse_args()

    prepare_analysis(nodes_file=args.nodes, edges_file=args.edges, n_hops=args.n_hops,
                    matrices_dir=args.matrices_dir, config_path=args.config)


if __name__ == "__main__":
    main()
