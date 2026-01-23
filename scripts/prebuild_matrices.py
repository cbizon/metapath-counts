#!/usr/bin/env python3
"""
Pre-build and serialize GraphBLAS matrices for faster analysis startup.

This script:
1. Loads the full knowledge graph
2. Builds all (src_type, predicate, tgt_type) matrices
3. Serializes matrices to disk in compressed format
4. Saves node indexing metadata

This avoids re-loading edges and re-building matrices in every SLURM job.

Usage:
    uv run python scripts/prebuild_matrices.py \
        --edges edges.jsonl \
        --nodes nodes.jsonl \
        --output matrices/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import yaml

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))
from analyze_hop_overlap import load_node_types, build_matrices


def serialize_matrix(matrix, output_path):
    """
    Serialize a GraphBLAS matrix to npz format.

    Args:
        matrix: GraphBLAS matrix
        output_path: Path to save .npz file
    """
    # Extract sparse representation
    rows, cols, vals = matrix.to_coo()

    # Save as compressed npz
    np.savez_compressed(
        output_path,
        rows=rows,
        cols=cols,
        vals=vals,
        nrows=matrix.nrows,
        ncols=matrix.ncols,
        nvals=matrix.nvals
    )


def save_node_indexing(node_to_idx, output_dir):
    """
    Save node-to-index mappings for each type.

    Args:
        node_to_idx: Dict of {type: {node_id: index}}
        output_dir: Directory to save metadata
    """
    metadata_path = os.path.join(output_dir, 'node_indexing.json')

    # Convert to serializable format
    serializable = {}
    for node_type, mapping in node_to_idx.items():
        serializable[node_type] = {
            'count': len(mapping),
            # We don't save the full mapping to save space
            # Jobs can rebuild from nodes.jsonl if needed
        }

    with open(metadata_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved node indexing metadata to {metadata_path}")


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


def main():
    parser = argparse.ArgumentParser(
        description='Pre-build and serialize matrices for fast loading'
    )
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--output', required=True, help='Output directory for matrices')
    parser.add_argument(
        '--config',
        default='config/type_expansion.yaml',
        help='Path to configuration YAML file (default: config/type_expansion.yaml)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    type_config = load_config(args.config)

    print("=" * 80)
    print("PRE-BUILDING MATRICES")
    print("=" * 80)
    print(f"Nodes file: {args.nodes}")
    print(f"Edges file: {args.edges}")
    print(f"Output dir: {args.output}")
    print(f"Config file: {args.config}")
    if type_config:
        print(f"Type expansion enabled: {type_config.get('enabled', True)}")
        print(f"Excluded types: {len(type_config.get('exclude_types', []))}")
    print()

    # Load node types
    print("[1/3] Loading node types...")
    start_time = time.time()
    node_types = load_node_types(args.nodes, config=type_config)
    node_load_time = time.time() - start_time
    print(f"Loaded {len(node_types):,} nodes in {node_load_time:.1f}s")
    print()

    # Build matrices
    print("[2/3] Building matrices from edges...")
    start_time = time.time()
    matrices = build_matrices(args.edges, node_types)
    build_time = time.time() - start_time
    print(f"Built {len(matrices):,} matrices in {build_time:.1f}s ({build_time/60:.1f}min)")
    print()

    # Serialize matrices
    print("[3/3] Serializing matrices to disk...")
    start_time = time.time()

    total_bytes = 0
    for i, ((src_type, pred, tgt_type), matrix) in enumerate(matrices.items(), 1):
        # Create safe filename
        filename = f"{src_type}__{pred}__{tgt_type}.npz"
        output_path = output_dir / filename

        # Serialize
        serialize_matrix(matrix, output_path)

        # Track size
        file_size = output_path.stat().st_size
        total_bytes += file_size

        if i % 10 == 0:
            print(f"  Serialized {i}/{len(matrices)} matrices ({total_bytes / 1e9:.2f} GB so far)")

    serialize_time = time.time() - start_time

    # Create manifest
    manifest = {
        'source_files': {
            'nodes': args.nodes,
            'edges': args.edges
        },
        'num_matrices': len(matrices),
        'total_size_bytes': total_bytes,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'type_expansion': {
            'enabled': True,
            'config_file': args.config,
            'exclude_types': list(type_config.get('exclude_types', [])) if type_config else []
        },
        'matrices': []
    }

    # Add matrix metadata
    for (src_type, pred, tgt_type), matrix in matrices.items():
        manifest['matrices'].append({
            'src_type': src_type,
            'predicate': pred,
            'tgt_type': tgt_type,
            'nrows': int(matrix.nrows),
            'ncols': int(matrix.ncols),
            'nvals': int(matrix.nvals),
            'filename': f"{src_type}__{pred}__{tgt_type}.npz"
        })

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 80)
    print("PREBUILD COMPLETE")
    print("=" * 80)
    print(f"Matrices saved: {len(matrices)}")
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    print(f"Compression ratio: {(84e9 / total_bytes):.1f}× (vs 84GB edges)")
    print(f"Serialization time: {serialize_time:.1f}s ({serialize_time/60:.1f}min)")
    print(f"Manifest: {manifest_path}")
    print()
    print("Next steps:")
    print(f"  Use --matrices-dir {args.output} in analyze_hop_overlap.py")
    print()


if __name__ == '__main__':
    main()
