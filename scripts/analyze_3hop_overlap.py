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


def build_matrices(edges_file: str, node_types: dict, enable_zeroing: bool = True):
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

        # DIAGONAL ZEROING #1: Remove self-loops from input edges (A→A)
        # This prevents paths like: NodeA -treats-> NodeA
        if enable_zeroing and nrows == ncols:  # Only square matrices have diagonals
            matrix = matrix.select(gb.select.offdiag).new()

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


def analyze_3hop_overlap(matrices, output_file, matrix1_index=None, enable_zeroing=True):
    """Compute 3-hop metapaths and calculate overlap with 1-hop edges.

    IMPORTANT: This function can optionally implement three-level diagonal zeroing
    to reduce paths that loop back to the starting node:
      1. Input matrices: removes self-loops from edges.jsonl (A→A)
      2. After A @ B: prevents paths where start node reappears at position 2
      3. After (A @ B) @ C: prevents paths where start node == end node (if same type)

    NOTE: Diagonal zeroing does NOT prevent all repeated nodes, only those that
    loop back to the start. See DIAGONAL_ZEROING.md for details.

    Args:
        matrices: Dict of (src_type, pred, tgt_type) -> GraphBLAS matrix
        output_file: Path to output TSV file
        matrix1_index: Optional index to process only a single Matrix1 (for parallelization)
        enable_zeroing: If True, apply diagonal zeroing (default). If False, count all paths.
    """
    print(f"\n{'=' * 80}", flush=True)
    print("ANALYZING 3-HOP TO 1-HOP OVERLAP", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"Diagonal zeroing: {'ENABLED' if enable_zeroing else 'DISABLED'}", flush=True)
    if not enable_zeroing:
        print("  WARNING: All paths will be counted, including those with repeated nodes", flush=True)

    start_time = time.time()

    # Build extended matrix list with inverses
    all_matrices, matrix_metadata = build_matrix_list(matrices)

    print(f"Total matrices (with inverses): {len(all_matrices):,}", flush=True)
    print(f"Memory: {get_memory_mb():.0f} MB", flush=True)

    if matrix1_index is not None:
        print(f"Processing ONLY Matrix1 index: {matrix1_index}", flush=True)

    # Build aggregated 1-hop matrices (sum all predicates AND directions per node type pair)
    print(f"\nBuilding aggregated 1-hop matrices...", flush=True)
    aggregated_1hop = {}
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        key = (src_type, tgt_type)
        if key not in aggregated_1hop:
            aggregated_1hop[key] = matrix.dup()
        else:
            # Element-wise OR (union of edges)
            aggregated_1hop[key] = aggregated_1hop[key].ewise_add(matrix, gb.binary.any).new()

    print(f"Created {len(aggregated_1hop):,} aggregated type-pair matrices (both F and R)", flush=True)
    for (src_type, tgt_type), agg_matrix in list(aggregated_1hop.items())[:5]:
        print(f"  {src_type} -> {tgt_type}: {agg_matrix.nvals:,} edges", flush=True)

    # Group by source type
    by_source_type = defaultdict(list)
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        by_source_type[src_type].append((src_type, pred, tgt_type, matrix, direction))

    # Open output file
    with open(output_file, 'w') as f:
        f.write("3hop_metapath\t3hop_count\t1hop_metapath\t1hop_count\toverlap\ttotal_possible\n")

        total_comparisons = 0
        rows_written = 0
        matrix2_count = 0
        matrix3_count = 0
        total_matrix2_needed = sum(
            len(by_source_type.get(tgt_type1, []))
            for _, _, tgt_type1, _, _ in all_matrices
        )

        # Estimate total Matrix3 iterations
        total_matrix3_needed = 0
        for src_type1, pred1, tgt_type1, matrix1, dir1 in all_matrices:
            if tgt_type1 in by_source_type:
                for src_type2, pred2, tgt_type2, matrix2, dir2 in by_source_type[tgt_type1]:
                    if tgt_type2 in by_source_type:
                        total_matrix3_needed += len(by_source_type[tgt_type2])

        # Generate 3-hop metapaths
        print(f"\nProcessing 3-hop metapaths...", flush=True)
        print(f"Total Matrix2 iterations needed: ~{total_matrix2_needed:,}", flush=True)
        print(f"Total Matrix3 iterations needed: ~{total_matrix3_needed:,}", flush=True)

        # Determine which Matrix1 indices to process
        if matrix1_index is not None:
            if matrix1_index < 0 or matrix1_index >= len(all_matrices):
                raise ValueError(f"matrix1_index {matrix1_index} out of range [0, {len(all_matrices)})")
            matrix1_list = [(matrix1_index, all_matrices[matrix1_index])]
        else:
            matrix1_list = list(enumerate(all_matrices))

        for idx1, (src_type1, pred1, tgt_type1, matrix1, dir1) in matrix1_list:
            if tgt_type1 not in by_source_type:
                continue

            m1_nvals = matrix1.nvals  # Store for filtering

            for idx2, (src_type2, pred2, tgt_type2, matrix2, dir2) in enumerate(by_source_type[tgt_type1]):
                # Filter Matrix3 candidates: only process if M3.nvals >= M1.nvals
                # This eliminates duplicate computation (path computed from both ends)
                if tgt_type2 in by_source_type:
                    valid_matrix3s = [
                        (src_type3, pred3, tgt_type3, matrix3, dir3)
                        for src_type3, pred3, tgt_type3, matrix3, dir3 in by_source_type[tgt_type2]
                        if matrix3.nvals >= m1_nvals
                    ]
                else:
                    valid_matrix3s = []

                # Skip this M2 if no valid M3s remain after filtering
                if not valid_matrix3s:
                    continue

                matrix2_count += 1

                # Progress after each Matrix2
                elapsed = time.time() - start_time
                rate = matrix2_count / elapsed if elapsed > 0 else 0
                eta_sec = (total_matrix2_needed - matrix2_count) / rate if rate > 0 else 0
                eta_min = eta_sec / 60
                eta_hr = eta_min / 60

                print(f"  Matrix2: {matrix2_count:,}/{total_matrix2_needed:,} | Elapsed: {elapsed/60:.1f}min | ETA: {eta_hr:.1f}hr | Rows: {rows_written:,} | Mem: {get_memory_mb():.0f}MB", flush=True)
                if matrix1.ncols != matrix2.nrows:
                    continue

                # Compute A @ B
                result_AB = matrix1.mxm(matrix2, gb.semiring.any_pair).new()

                # DIAGONAL ZEROING #2: Prevent paths where start node reappears at position 2
                # This prevents paths like: NodeA -affects-> NodeB -regulates-> NodeA -treats-> NodeC
                if enable_zeroing and result_AB.nrows == result_AB.ncols:
                    result_AB = result_AB.select(gb.select.offdiag).new()

                if result_AB.nvals == 0:
                    del result_AB  # Clean up before continuing to next Matrix2
                    continue

                for src_type3, pred3, tgt_type3, matrix3, dir3 in valid_matrix3s:
                    matrix3_count += 1
                    iter_start = time.time()

                    if result_AB.ncols != matrix3.nrows:
                        continue

                    # TIMING: Compute (A @ B) @ C
                    mult_start = time.time()
                    result_ABC = result_AB.mxm(matrix3, gb.semiring.any_pair).new()

                    # DIAGONAL ZEROING #3: Prevent paths where start node == end node (if same type)
                    # Only applies when start and end node types match (square matrix)
                    if enable_zeroing and result_ABC.nrows == result_ABC.ncols:
                        result_ABC = result_ABC.select(gb.select.offdiag).new()

                    mult_time = time.time() - mult_start

                    if result_ABC.nvals == 0:
                        del result_ABC  # Clean up even when empty
                        continue

                    # Format 3-hop metapath
                    threehop_metapath = format_metapath(
                        [src_type1, tgt_type1, tgt_type2, tgt_type3],
                        [pred1, pred2, pred3],
                        [dir1, dir2, dir3]
                    )
                    threehop_count = result_ABC.nvals
                    total_possible = result_ABC.nrows * result_ABC.ncols

                    # TIMING: Compare with all 1-hop matrices
                    overlap_start = time.time()
                    overlap_comparisons = 0
                    for (onehop_src, onehop_pred, onehop_tgt), (onehop_matrix, onehop_dir) in matrix_metadata.items():
                        # Check dimension compatibility
                        if onehop_src != src_type1 or onehop_tgt != tgt_type3:
                            continue

                        if result_ABC.nrows != onehop_matrix.nrows or result_ABC.ncols != onehop_matrix.ncols:
                            continue

                        # Calculate overlap
                        overlap_matrix = result_ABC.ewise_mult(onehop_matrix, gb.binary.pair).new()
                        overlap_count = overlap_matrix.nvals
                        onehop_count = onehop_matrix.nvals
                        overlap_comparisons += 1

                        # Explicitly delete overlap matrix to free memory
                        del overlap_matrix

                        # Format 1-hop metapath
                        onehop_metapath = format_metapath(
                            [onehop_src, onehop_tgt],
                            [onehop_pred],
                            [onehop_dir]
                        )

                        # Write row
                        f.write(f"{threehop_metapath}\t{threehop_count}\t{onehop_metapath}\t{onehop_count}\t{overlap_count}\t{total_possible}\n")
                        rows_written += 1
                        total_comparisons += 1

                        if rows_written % 10000 == 0:
                            f.flush()

                    overlap_time = time.time() - overlap_start

                    # TIMING: Aggregated comparison
                    agg_start = time.time()
                    agg_key = (src_type1, tgt_type3)
                    if agg_key in aggregated_1hop:
                        agg_matrix = aggregated_1hop[agg_key]

                        if result_ABC.nrows == agg_matrix.nrows and result_ABC.ncols == agg_matrix.ncols:
                            # Calculate overlap with aggregated
                            overlap_matrix = result_ABC.ewise_mult(agg_matrix, gb.binary.pair).new()
                            overlap_count = overlap_matrix.nvals
                            agg_count = agg_matrix.nvals

                            # Explicitly delete overlap matrix to free memory
                            del overlap_matrix

                            # Format aggregated 1-hop metapath
                            agg_metapath = f"{src_type1}|ANY|A|{tgt_type3}"

                            # Write row
                            f.write(f"{threehop_metapath}\t{threehop_count}\t{agg_metapath}\t{agg_count}\t{overlap_count}\t{total_possible}\n")
                            rows_written += 1

                            if rows_written % 10000 == 0:
                                f.flush()
                    agg_time = time.time() - agg_start

                    # Explicitly delete result_ABC to free memory before next iteration
                    del result_ABC

                    # Print timing breakdown
                    iter_time = time.time() - iter_start
                    elapsed = time.time() - start_time
                    rate = matrix3_count / elapsed if elapsed > 0 else 0
                    eta_sec = (total_matrix3_needed - matrix3_count) / rate if rate > 0 else 0
                    eta_hr = eta_sec / 3600

                    print(f"    Matrix3: {matrix3_count:,}/{total_matrix3_needed:,} | {threehop_metapath} | "
                          f"Total: {iter_time:.1f}s (Mult: {mult_time:.1f}s, Overlap: {overlap_time:.1f}s [{overlap_comparisons} comps], Agg: {agg_time:.1f}s) | "
                          f"ETA: {eta_hr:.1f}hr | Mem: {get_memory_mb():.0f}MB", flush=True)

                    # Force garbage collection to free C memory from GraphBLAS matrices
                    # More aggressive: collect every 2 iterations instead of every 5
                    if matrix3_count % 2 == 0:
                        gc.collect()

                # Explicitly delete result_AB after processing all Matrix3s for this Matrix2
                del result_AB
                gc.collect()

    print(f"\nDone! Wrote {rows_written:,} rows to {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze overlap between 3-hop metapaths and 1-hop edges'
    )
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--output', required=True, help='Output TSV file path')
    parser.add_argument('--matrix1-index', type=int, default=None,
                        help='Process only this Matrix1 index (for parallelization)')
    parser.add_argument('--no-zeroing', action='store_true',
                        help='Disable diagonal zeroing (count all paths including those with repeated nodes)')

    args = parser.parse_args()

    # Load data and build matrices
    node_types = load_node_types(args.nodes)
    matrices = build_matrices(args.edges, node_types, enable_zeroing=not args.no_zeroing)

    # Analyze overlap
    analyze_3hop_overlap(matrices, args.output, matrix1_index=args.matrix1_index, enable_zeroing=not args.no_zeroing)

    print(f"\nFinal memory: {get_memory_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
