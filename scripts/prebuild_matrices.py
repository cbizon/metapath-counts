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
from collections import defaultdict
import numpy as np
import graphblas as gb
from metapath_counts import assign_node_type, get_symmetric_predicates, is_pseudo_type


def load_node_types(nodes_file: str, config: dict = None) -> dict:
    """
    Load node types from KGX nodes file.

    Each node is assigned to exactly ONE type or pseudo-type.

    Args:
        nodes_file: Path to KGX nodes.jsonl file
        config: Unused (kept for backwards compatibility)

    Returns:
        dict: Mapping of node_id -> str (single type or pseudo-type)
    """
    print(f"Loading node types from {nodes_file}...", flush=True)

    node_types = {}
    pseudo_type_count = 0

    with open(nodes_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1_000_000 == 0:
                print(f"  Loaded {line_num:,} nodes", flush=True)

            node = json.loads(line)
            node_id = node['id']
            categories = node.get('category', [])

            if categories:
                # Assign single type or pseudo-type
                assigned_type = assign_node_type(categories)

                if assigned_type:
                    node_types[node_id] = assigned_type

                    # Track pseudo-types
                    if is_pseudo_type(assigned_type):
                        pseudo_type_count += 1

    print(f"Loaded {len(node_types):,} nodes with assigned types", flush=True)
    print(f"  Pseudo-types (multi-leaf): {pseudo_type_count:,} nodes ({100*pseudo_type_count/len(node_types):.1f}%)", flush=True)

    return node_types


def build_matrices(edges_file: str, node_types: dict):
    """
    Build sparse matrices for each (source_type, predicate, target_type) triple.

    Each node has exactly one assigned type, so each edge creates exactly one
    matrix entry (or two for symmetric predicates).

    Args:
        edges_file: Path to KGX edges.jsonl file
        node_types: dict mapping node_id -> str (single type or pseudo-type)

    Returns:
        dict: Mapping of (src_type, pred, tgt_type) -> GraphBLAS Matrix
        dict: Mapping of type_name -> {node_id: index}
    """
    print(f"\nCollecting edge types from {edges_file}...", flush=True)

    # Get symmetric predicates from biolink model
    symmetric_predicates = get_symmetric_predicates()
    print(f"Loaded {len(symmetric_predicates)} symmetric predicates from Biolink Model", flush=True)

    edge_triples = defaultdict(list)
    node_to_idx = defaultdict(dict)

    skipped_subclass = 0
    skipped_no_types = 0
    edges_processed = 0

    with open(edges_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1_000_000 == 0:
                print(f"  Processed {line_num:,} edges", flush=True)

            edge = json.loads(line)
            predicate = edge.get('predicate', '')

            if predicate == 'biolink:subclass_of':
                skipped_subclass += 1
                continue

            subject = edge['subject']
            obj = edge['object']
            src_type = node_types.get(subject)  # Now returns single str
            tgt_type = node_types.get(obj)      # Now returns single str

            if not src_type or not tgt_type:
                skipped_no_types += 1
                continue

            pred = predicate.replace('biolink:', '')

            # For symmetric predicates, add edge in both directions
            is_symmetric = pred in symmetric_predicates

            # Add forward direction
            triple = (src_type, pred, tgt_type)
            edge_triples[triple].append((subject, obj))

            # Track node indices for both types
            if subject not in node_to_idx[src_type]:
                idx = len(node_to_idx[src_type])
                node_to_idx[src_type][subject] = idx

            if obj not in node_to_idx[tgt_type]:
                idx = len(node_to_idx[tgt_type])
                node_to_idx[tgt_type][obj] = idx

            # Add reverse direction for symmetric predicates
            if is_symmetric:
                reverse_triple = (tgt_type, pred, src_type)
                edge_triples[reverse_triple].append((obj, subject))

            edges_processed += 1

    print(f"\nEdge statistics:", flush=True)
    print(f"  Processed: {edges_processed:,} edges", flush=True)
    print(f"  Skipped (subclass_of): {skipped_subclass:,}", flush=True)
    print(f"  Skipped (no types): {skipped_no_types:,}", flush=True)
    print(f"  Unique edge type triples: {len(edge_triples):,}", flush=True)

    # Build matrices
    print(f"\nBuilding GraphBLAS matrices...", flush=True)
    matrices = {}

    # Track matrix size distribution
    matrix_sizes = []

    for triple, edges in edge_triples.items():
        src_type, pred, tgt_type = triple
        rows = [node_to_idx[src_type][src_id] for src_id, _ in edges]
        cols = [node_to_idx[tgt_type][tgt_id] for _, tgt_id in edges]

        nrows = len(node_to_idx[src_type])
        ncols = len(node_to_idx[tgt_type])

        matrix = gb.Matrix.from_coo(
            rows, cols, [1] * len(rows),
            nrows=nrows, ncols=ncols,
            dtype=gb.dtypes.BOOL,
            dup_op=gb.binary.any
        )

        matrices[triple] = matrix
        matrix_sizes.append((nrows * ncols, len(edges)))

    print(f"Built {len(matrices):,} matrices", flush=True)

    # Print matrix size statistics
    if matrix_sizes:
        total_dims = sum(size for size, _ in matrix_sizes)
        total_entries = sum(entries for _, entries in matrix_sizes)
        avg_dim = total_dims / len(matrix_sizes)
        avg_entries = total_entries / len(matrix_sizes)
        print(f"  Average matrix dimensions: {avg_dim:.0f}", flush=True)
        print(f"  Average entries per matrix: {avg_entries:.0f}", flush=True)

    return matrices, node_to_idx


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


def main():
    parser = argparse.ArgumentParser(
        description='Pre-build and serialize matrices for fast loading'
    )
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--output', required=True, help='Output directory for matrices')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PRE-BUILDING MATRICES")
    print("=" * 80)
    print(f"Nodes file: {args.nodes}")
    print(f"Edges file: {args.edges}")
    print(f"Output dir: {args.output}")
    print(f"Method: Single type or pseudo-type assignment")
    print()

    # Load node types
    print("[1/3] Loading node types...")
    start_time = time.time()
    node_types = load_node_types(args.nodes)
    node_load_time = time.time() - start_time
    print(f"Loaded {len(node_types):,} nodes in {node_load_time:.1f}s")
    print()

    # Build matrices
    print("[2/3] Building matrices from edges...")
    start_time = time.time()
    matrices, node_to_idx = build_matrices(args.edges, node_types)
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
        'type_assignment': {
            'method': 'single_type_or_pseudo',
            'description': 'Each node assigned to single type or pseudo-type for multi-leaf nodes'
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
