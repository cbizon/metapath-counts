#!/usr/bin/env python3
"""
Analyze overlap between 3-hop metapaths and 1-hop edges.

Computes all 3-hop metapaths via matrix multiplication, then calculates
overlap with 1-hop edges to identify which 1-hop edges appear in 3-hop paths.

Output format (TSV):
  3hop_metapath          | 3hop_count | 1hop_metapath    | 1hop_count | overlap | total_possible
  SmallMolecule|affects|F|Gene|affects|R|SmallMolecule|affects|F|Gene | 6170000000 | SmallMolecule|regulates|F|Gene | 500000 | 450000 | 201000000000

Metapath format: NodeType|predicate|direction|NodeType|...
  - Pipe separated
  - F = forward, R = reverse, A = any (symmetric predicates)
  - Example: Disease|treats|R|SmallMolecule|affects|F|Gene
  - Symmetric example: Gene|directly_physically_interacts_with|A|Gene

Usage:
    uv run python analyze_3hop_overlap.py \
        --edges ../SimplePredictions/input_graphs/robokop_base_nonredundant/edges.jsonl \
        --nodes ../SimplePredictions/input_graphs/robokop_base_nonredundant/nodes.jsonl \
        --output 3hop_1hop_overlap.tsv
"""

import argparse
import json
from collections import defaultdict
import psutil
import os
import time
import gc
import graphblas as gb
from metapath_counts import get_most_specific_type, get_symmetric_predicates


def get_memory_mb():
    """Get current process memory usage in MB (including C libraries like GraphBLAS)."""
    import subprocess
    pid = os.getpid()

    # Use ps command to get actual RSS (includes C library allocations)
    try:
        # macOS ps: rss is in KB
        result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)],
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            rss_kb = int(result.stdout.strip())
            return rss_kb / 1024  # Convert KB to MB
    except:
        pass

    # Fallback: use psutil
    process = psutil.Process(pid)
    return process.memory_info().rss / 1024 / 1024


def is_canonical_direction(src_type: str, tgt_type: str) -> bool:
    """
    Check if a 1-hop path is in canonical direction.

    For 1-hop paths, we can't use size-based duplicate elimination (there's only
    one matrix). Instead, we use alphabetical ordering of node types.

    Returns True if src_type <= tgt_type (alphabetically), meaning this is the
    canonical direction to process.

    Args:
        src_type: Source node type
        tgt_type: Target node type

    Returns:
        True if this is the canonical direction to process
    """
    return src_type <= tgt_type


def should_process_path(n_hops: int, first_matrix_nvals: int, last_matrix_nvals: int,
                        src_type: str, tgt_type: str) -> bool:
    """
    Determine whether to process a path to avoid duplicate computation.

    Each N-hop path can be computed from either direction (A→...→Z or Z→...→A).
    This function determines which direction to process:

    - For N=1: Use canonical direction (alphabetical ordering of node types)
    - For N>1: Use size-based rule (last_matrix.nvals >= first_matrix.nvals)

    Args:
        n_hops: Number of hops in the path
        first_matrix_nvals: Number of values in the first matrix
        last_matrix_nvals: Number of values in the last matrix
        src_type: Source node type (start of path)
        tgt_type: Target node type (end of path)

    Returns:
        True if this path should be processed (not a duplicate)
    """
    if n_hops == 1:
        return is_canonical_direction(src_type, tgt_type)
    else:
        return last_matrix_nvals >= first_matrix_nvals


def format_metapath(node_types, predicates, directions):
    """
    Format metapath as parsable string.

    For symmetric predicates, uses 'A' (any) direction instead of 'F' or 'R'.

    Example: ['Disease', 'SmallMolecule', 'Gene'], ['treats', 'regulates'], ['R', 'F']
    Returns: 'Disease|treats|R|SmallMolecule|regulates|F|Gene'

    Symmetric example: ['Gene', 'Gene'], ['directly_physically_interacts_with'], ['F']
    Returns: 'Gene|directly_physically_interacts_with|A|Gene'
    """
    # Get symmetric predicates from biolink model
    symmetric_predicates = get_symmetric_predicates()

    parts = []
    for i, node_type in enumerate(node_types):
        parts.append(node_type)
        if i < len(predicates):
            pred = predicates[i]
            parts.append(pred)

            # Use 'A' for symmetric predicates, otherwise use the provided direction
            if pred in symmetric_predicates:
                parts.append('A')
            else:
                parts.append(directions[i])

    return '|'.join(parts)


def load_node_types(nodes_file: str) -> dict:
    """Load node types from KGX nodes file."""
    print(f"Loading node types from {nodes_file}...", flush=True)
    node_types = {}

    with open(nodes_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1_000_000 == 0:
                print(f"  Loaded {line_num:,} nodes", flush=True)

            node = json.loads(line)
            node_id = node['id']
            categories = node.get('category', [])

            if categories:
                most_specific = get_most_specific_type(categories)
                primary_type = most_specific.replace('biolink:', '')
                node_types[node_id] = primary_type

    print(f"Loaded {len(node_types):,} node types", flush=True)
    return node_types


def build_matrices(edges_file: str, node_types: dict):
    """Build sparse matrices for each (source_type, predicate, target_type) triple."""
    print(f"\nCollecting edge types from {edges_file}...", flush=True)

    # Get symmetric predicates from biolink model
    symmetric_predicates = get_symmetric_predicates()
    print(f"Loaded {len(symmetric_predicates)} symmetric predicates from Biolink Model", flush=True)

    edge_triples = defaultdict(list)
    node_to_idx = defaultdict(dict)

    skipped_subclass = 0
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
            src_type = node_types.get(subject)
            tgt_type = node_types.get(obj)

            if not src_type or not tgt_type:
                continue

            pred = predicate.replace('biolink:', '')

            # For symmetric predicates, add edge in both directions
            is_symmetric = pred in symmetric_predicates

            # Add forward direction
            triple = (src_type, pred, tgt_type)
            edge_triples[triple].append((subject, obj))

            # Add reverse direction for symmetric predicates
            if is_symmetric:
                reverse_triple = (tgt_type, pred, src_type)
                edge_triples[reverse_triple].append((obj, subject))

            # Track node indices for both types
            if subject not in node_to_idx[src_type]:
                idx = len(node_to_idx[src_type])
                node_to_idx[src_type][subject] = idx

            if obj not in node_to_idx[tgt_type]:
                idx = len(node_to_idx[tgt_type])
                node_to_idx[tgt_type][obj] = idx

            edges_processed += 1

    print(f"\nEdge statistics:", flush=True)
    print(f"  Processed: {edges_processed:,}", flush=True)
    print(f"  Unique edge type triples: {len(edge_triples):,}", flush=True)

    # Build matrices
    print(f"\nBuilding GraphBLAS matrices...", flush=True)
    matrices = {}

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

    print(f"Built {len(matrices):,} matrices", flush=True)
    return matrices


def build_matrix_list(matrices):
    """Build extended matrix list with forward and reverse directions."""
    all_matrices = []
    matrix_metadata = {}  # (src_type, pred, tgt_type) -> (matrix, direction)

    # Get symmetric predicates from biolink model
    symmetric_predicates = get_symmetric_predicates()

    for (src_type, pred, tgt_type), matrix in matrices.items():
        # Forward
        all_matrices.append((src_type, pred, tgt_type, matrix, 'F'))
        matrix_metadata[(src_type, pred, tgt_type)] = (matrix, 'F')

        # Reverse (if not symmetric)
        is_symmetric = pred in symmetric_predicates
        if not is_symmetric:
            all_matrices.append((tgt_type, pred, src_type, matrix.T, 'R'))
            matrix_metadata[(tgt_type, pred, src_type)] = (matrix.T, 'R')

    return all_matrices, matrix_metadata


def analyze_nhop_overlap(matrices, output_file, n_hops=3, matrix1_index=None):
    """Compute N-hop metapaths and calculate overlap with 1-hop edges.

    This is a generalized version that handles any number of hops (1, 2, 3, ..., N).

    NOTE: This function does NOT filter out paths with repeated nodes. The counts
    include all paths regardless of whether nodes are revisited. Filtering for
    distinct node paths should be done in post-processing if needed.

    Args:
        matrices: Dict of (src_type, pred, tgt_type) -> GraphBLAS matrix
        output_file: Path to output TSV file
        n_hops: Number of hops for the metapath (default: 3)
        matrix1_index: Optional index to process only a single Matrix1 (for parallelization)
    """
    print(f"\n{'=' * 80}", flush=True)
    print(f"ANALYZING {n_hops}-HOP TO 1-HOP OVERLAP", flush=True)
    print(f"{'=' * 80}", flush=True)

    start_time = time.time()

    # Build extended matrix list with inverses
    all_matrices, matrix_metadata = build_matrix_list(matrices)

    print(f"Total matrices (with inverses): {len(all_matrices):,}", flush=True)
    print(f"Memory: {get_memory_mb():.0f} MB", flush=True)

    if matrix1_index is not None:
        print(f"Processing ONLY Matrix1 index: {matrix1_index}", flush=True)

    # Build aggregated 1-hop matrices
    print(f"\nBuilding aggregated 1-hop matrices...", flush=True)
    aggregated_1hop = {}
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        key = (src_type, tgt_type)
        if key not in aggregated_1hop:
            aggregated_1hop[key] = matrix.dup()
        else:
            aggregated_1hop[key] = aggregated_1hop[key].ewise_add(matrix, gb.binary.any).new()

    print(f"Created {len(aggregated_1hop):,} aggregated type-pair matrices", flush=True)

    # Group by source type for efficient lookup
    by_source_type = defaultdict(list)
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        by_source_type[src_type].append((src_type, pred, tgt_type, matrix, direction))

    # Open output file
    with open(output_file, 'w') as f:
        f.write(f"{n_hops}hop_metapath\t{n_hops}hop_count\t1hop_metapath\t1hop_count\toverlap\ttotal_possible\n")

        rows_written = 0
        paths_computed = 0

        # Determine which Matrix1 indices to process
        if matrix1_index is not None:
            if matrix1_index < 0 or matrix1_index >= len(all_matrices):
                raise ValueError(f"matrix1_index {matrix1_index} out of range [0, {len(all_matrices)})")
            matrix1_list = [(matrix1_index, all_matrices[matrix1_index])]
        else:
            matrix1_list = list(enumerate(all_matrices))

        # Recursive function to build N-hop paths
        def process_path(depth, accumulated_matrix, node_types, predicates, directions, first_matrix_nvals):
            """
            Recursively build N-hop paths by matrix multiplication.

            Args:
                depth: Current hop depth (1 to n_hops)
                accumulated_matrix: Result of multiplying matrices so far
                node_types: List of node types in the path so far
                predicates: List of predicates in the path so far
                directions: List of directions in the path so far
                first_matrix_nvals: Number of values in the first matrix (for duplicate elimination)
            """
            nonlocal rows_written, paths_computed

            if depth == n_hops:
                # We've completed an N-hop path, now compare with 1-hop matrices
                src_type_final = node_types[0]
                tgt_type_final = node_types[-1]

                # Check if we should process this path (duplicate elimination)
                if not should_process_path(n_hops, first_matrix_nvals, accumulated_matrix.nvals,
                                          src_type_final, tgt_type_final):
                    return

                # Format N-hop metapath
                nhop_metapath = format_metapath(node_types, predicates, directions)
                nhop_count = accumulated_matrix.nvals
                total_possible = accumulated_matrix.nrows * accumulated_matrix.ncols

                paths_computed += 1
                if paths_computed % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Paths computed: {paths_computed:,} | Rows written: {rows_written:,} | "
                          f"Elapsed: {elapsed/60:.1f}min | Mem: {get_memory_mb():.0f}MB", flush=True)

                # Compare with all 1-hop matrices
                for (onehop_src, onehop_pred, onehop_tgt), (onehop_matrix, onehop_dir) in matrix_metadata.items():
                    if onehop_src != src_type_final or onehop_tgt != tgt_type_final:
                        continue

                    if accumulated_matrix.nrows != onehop_matrix.nrows or accumulated_matrix.ncols != onehop_matrix.ncols:
                        continue

                    # Calculate overlap
                    overlap_matrix = accumulated_matrix.ewise_mult(onehop_matrix, gb.binary.pair).new()
                    overlap_count = overlap_matrix.nvals
                    onehop_count = onehop_matrix.nvals
                    del overlap_matrix

                    # Format 1-hop metapath
                    onehop_metapath = format_metapath([onehop_src, onehop_tgt], [onehop_pred], [onehop_dir])

                    # Write row
                    f.write(f"{nhop_metapath}\t{nhop_count}\t{onehop_metapath}\t{onehop_count}\t{overlap_count}\t{total_possible}\n")
                    rows_written += 1

                    if rows_written % 10000 == 0:
                        f.flush()

                # Also compare with aggregated 1-hop
                agg_key = (src_type_final, tgt_type_final)
                if agg_key in aggregated_1hop:
                    agg_matrix = aggregated_1hop[agg_key]

                    if accumulated_matrix.nrows == agg_matrix.nrows and accumulated_matrix.ncols == agg_matrix.ncols:
                        overlap_matrix = accumulated_matrix.ewise_mult(agg_matrix, gb.binary.pair).new()
                        overlap_count = overlap_matrix.nvals
                        agg_count = agg_matrix.nvals
                        del overlap_matrix

                        agg_metapath = f"{src_type_final}|ANY|A|{tgt_type_final}"

                        f.write(f"{nhop_metapath}\t{nhop_count}\t{agg_metapath}\t{agg_count}\t{overlap_count}\t{total_possible}\n")
                        rows_written += 1

                        if rows_written % 10000 == 0:
                            f.flush()

                return

            # Recursive case: continue building the path
            current_target_type = node_types[-1]

            if current_target_type not in by_source_type:
                return

            for src_type, pred, tgt_type, matrix, direction in by_source_type[current_target_type]:
                # Check dimension compatibility
                if accumulated_matrix.ncols != matrix.nrows:
                    continue

                # Multiply matrices
                next_result = accumulated_matrix.mxm(matrix, gb.semiring.any_pair).new()

                if next_result.nvals == 0:
                    del next_result
                    continue

                # Recurse to next depth
                process_path(
                    depth + 1,
                    next_result,
                    node_types + [tgt_type],
                    predicates + [pred],
                    directions + [direction],
                    first_matrix_nvals
                )

                # Clean up
                del next_result
                if depth % 2 == 0:  # Periodic garbage collection
                    gc.collect()

        # Main loop: iterate over first matrices
        for idx1, (src_type1, pred1, tgt_type1, matrix1, dir1) in matrix1_list:
            print(f"\nProcessing Matrix1 {idx1}/{len(all_matrices)}: {src_type1}|{pred1}|{dir1}|{tgt_type1}", flush=True)

            # Start recursive path building
            process_path(
                depth=1,
                accumulated_matrix=matrix1,
                node_types=[src_type1, tgt_type1],
                predicates=[pred1],
                directions=[dir1],
                first_matrix_nvals=matrix1.nvals
            )

    print(f"\nDone! Wrote {rows_written:,} rows to {output_file}", flush=True)
    print(f"Total {n_hops}-hop paths computed: {paths_computed:,}", flush=True)


def analyze_3hop_overlap(matrices, output_file, matrix1_index=None):
    """Compute 3-hop metapaths and calculate overlap with 1-hop edges.

    This is a convenience wrapper around analyze_nhop_overlap with n_hops=3.
    Kept for backwards compatibility.

    Args:
        matrices: Dict of (src_type, pred, tgt_type) -> GraphBLAS matrix
        output_file: Path to output TSV file
        matrix1_index: Optional index to process only a single Matrix1 (for parallelization)
    """
    return analyze_nhop_overlap(matrices, output_file, n_hops=3, matrix1_index=matrix1_index)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze overlap between N-hop metapaths and 1-hop edges'
    )
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--output', required=True, help='Output TSV file path')
    parser.add_argument('--n-hops', type=int, default=3,
                        help='Number of hops for metapath analysis (default: 3)')
    parser.add_argument('--matrix1-index', type=int, default=None,
                        help='Process only this Matrix1 index (for parallelization)')

    args = parser.parse_args()

    # Load data and build matrices
    node_types = load_node_types(args.nodes)
    matrices = build_matrices(args.edges, node_types)

    # Analyze overlap
    analyze_nhop_overlap(matrices, args.output, n_hops=args.n_hops, matrix1_index=args.matrix1_index)

    print(f"\nFinal memory: {get_memory_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
