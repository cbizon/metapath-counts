#!/usr/bin/env python3
"""
Analyze overlap between N-hop metapaths and 1-hop edges.

Computes all N-hop metapaths via matrix multiplication, then calculates
overlap with 1-hop edges to identify which 1-hop edges appear in N-hop paths.

Output format (TSV):
  Nhop_metapath          | Nhop_count | 1hop_metapath    | 1hop_count | overlap | total_possible
  SmallMolecule|affects|F|Gene|affects|R|SmallMolecule|affects|F|Gene | 6170000000 | SmallMolecule|regulates|F|Gene | 500000 | 450000 | 201000000000

Metapath format: NodeType|predicate|direction|NodeType|...
  - Pipe separated
  - F = forward, R = reverse, A = any (symmetric predicates)
  - Example: Disease|treats|R|SmallMolecule|affects|F|Gene
  - Symmetric example: Gene|directly_physically_interacts_with|A|Gene

Usage:
    # 3-hop analysis (default)
    uv run python analyze_hop_overlap.py \
        --edges edges.jsonl \
        --nodes nodes.jsonl \
        --output results_3hop/results.tsv

    # 2-hop analysis
    uv run python analyze_hop_overlap.py \
        --edges edges.jsonl \
        --nodes nodes.jsonl \
        --n-hops 2 \
        --output results_2hop/results.tsv
"""

import argparse
import json
from collections import defaultdict
import psutil
import os
import sys
import time
import gc
import numpy as np
from pathlib import Path
import graphblas as gb
from metapath_counts import get_most_specific_type, get_symmetric_predicates, get_all_types

# Add path_tracker module
sys.path.insert(0, os.path.dirname(__file__))
from path_tracker import (
    generate_path_id,
    load_completed_paths,
    load_failed_paths,
    record_completed_path,
    record_failed_path,
    record_path_in_progress,
    clear_path_in_progress,
    enumerate_downstream_paths
)


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


def load_node_types(nodes_file: str, config: dict = None) -> dict:
    """
    Load node types from KGX nodes file.

    Returns dict mapping node_id -> list[str] of types (without biolink: prefix).
    Uses hierarchical type expansion if configured.

    Args:
        nodes_file: Path to KGX nodes.jsonl file
        config: Configuration dict with type_expansion settings:
                - exclude_types: set/list of types to exclude
                - max_depth: max hierarchy depth (None = unlimited)
                - include_most_specific: always include most specific type

    Returns:
        dict: Mapping of node_id -> list of type names (without biolink: prefix)
    """
    print(f"Loading node types from {nodes_file}...", flush=True)

    # Default config if none provided
    if config is None:
        config = {
            'exclude_types': {
                'ThingWithTaxon',
                'SubjectOfInvestigation',
                'PhysicalEssenceOrOccurrent',
                'PhysicalEssence',
                'OntologyClass',
                'Occurrent',
                'InformationContentEntity',
                'Attribute'
            },
            'max_depth': None,
            'include_most_specific': True
        }

    node_types = {}
    type_count_distribution = {}  # Track how many nodes have N types

    with open(nodes_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1_000_000 == 0:
                print(f"  Loaded {line_num:,} nodes", flush=True)

            node = json.loads(line)
            node_id = node['id']
            categories = node.get('category', [])

            if categories:
                # Get all types (hierarchical expansion)
                all_types = get_all_types(
                    categories,
                    exclude_types=config.get('exclude_types', set()),
                    max_depth=config.get('max_depth'),
                    include_most_specific=config.get('include_most_specific', True)
                )

                if all_types:
                    node_types[node_id] = all_types

                    # Track distribution
                    num_types = len(all_types)
                    type_count_distribution[num_types] = type_count_distribution.get(num_types, 0) + 1

    print(f"Loaded {len(node_types):,} nodes with hierarchical types", flush=True)

    # Print type distribution statistics
    if type_count_distribution:
        print(f"\nType count distribution:", flush=True)
        total_nodes = len(node_types)
        for count in sorted(type_count_distribution.keys()):
            num_nodes = type_count_distribution[count]
            pct = 100 * num_nodes / total_nodes
            print(f"  {count} types: {num_nodes:,} nodes ({pct:.1f}%)", flush=True)

        # Calculate average
        total_type_assignments = sum(k * v for k, v in type_count_distribution.items())
        avg_types = total_type_assignments / total_nodes
        print(f"  Average types per node: {avg_types:.2f}", flush=True)

    return node_types


def build_matrices(edges_file: str, node_types: dict):
    """
    Build sparse matrices for each (source_type, predicate, target_type) triple.

    With hierarchical types, each edge may appear in multiple matrices
    (one for each type combination).

    Args:
        edges_file: Path to KGX edges.jsonl file
        node_types: dict mapping node_id -> list[str] of types

    Returns:
        dict: Mapping of (src_type, pred, tgt_type) -> GraphBLAS Matrix
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
    total_matrix_entries = 0  # Track explosion factor

    with open(edges_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1_000_000 == 0:
                print(f"  Processed {line_num:,} edges, {total_matrix_entries:,} matrix entries", flush=True)

            edge = json.loads(line)
            predicate = edge.get('predicate', '')

            if predicate == 'biolink:subclass_of':
                skipped_subclass += 1
                continue

            subject = edge['subject']
            obj = edge['object']
            src_types = node_types.get(subject)  # Now returns list[str]
            tgt_types = node_types.get(obj)      # Now returns list[str]

            if not src_types or not tgt_types:
                skipped_no_types += 1
                continue

            pred = predicate.replace('biolink:', '')

            # For symmetric predicates, add edge in both directions
            is_symmetric = pred in symmetric_predicates

            # Iterate over all type combinations (Cartesian product)
            for src_type in src_types:
                for tgt_type in tgt_types:
                    # Add forward direction
                    triple = (src_type, pred, tgt_type)
                    edge_triples[triple].append((subject, obj))
                    total_matrix_entries += 1

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
                        total_matrix_entries += 1

            edges_processed += 1

    print(f"\nEdge statistics:", flush=True)
    print(f"  Processed: {edges_processed:,} edges", flush=True)
    print(f"  Skipped (no types): {skipped_no_types:,}", flush=True)
    print(f"  Total matrix entries: {total_matrix_entries:,}", flush=True)
    print(f"  Unique edge type triples: {len(edge_triples):,}", flush=True)
    if edges_processed > 0:
        expansion_factor = total_matrix_entries / edges_processed
        print(f"  Type expansion factor: {expansion_factor:.2f}x", flush=True)

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

    return matrices


def load_prebuilt_matrices(matrices_dir: str):
    """
    Load pre-built matrices from disk.

    Args:
        matrices_dir: Directory containing serialized matrices

    Returns:
        Dict of (src_type, pred, tgt_type) -> GraphBLAS matrix
    """
    print(f"Loading pre-built matrices from {matrices_dir}...", flush=True)
    start_time = time.time()

    matrices_path = Path(matrices_dir)
    manifest_path = matrices_path / 'manifest.json'

    if not manifest_path.exists():
        raise FileNotFoundError(f"Matrix manifest not found: {manifest_path}")

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Found {manifest['num_matrices']} pre-built matrices ({manifest['total_size_bytes']/1e9:.2f} GB)", flush=True)

    # Load each matrix
    matrices = {}
    for i, mat_info in enumerate(manifest['matrices'], 1):
        src_type = mat_info['src_type']
        pred = mat_info['predicate']
        tgt_type = mat_info['tgt_type']
        filename = mat_info['filename']

        # Load npz file
        npz_path = matrices_path / filename
        data = np.load(npz_path)

        # Reconstruct GraphBLAS matrix
        matrix = gb.Matrix.from_coo(
            data['rows'],
            data['cols'],
            data['vals'],
            nrows=int(data['nrows']),
            ncols=int(data['ncols']),
            dtype=gb.dtypes.BOOL,
            dup_op=gb.binary.any
        )

        matrices[(src_type, pred, tgt_type)] = matrix

        if i % 20 == 0:
            print(f"  Loaded {i}/{len(manifest['matrices'])} matrices", flush=True)

    load_time = time.time() - start_time
    print(f"Loaded {len(matrices):,} matrices in {load_time:.1f}s", flush=True)
    print(f"Memory: {get_memory_mb():.0f} MB", flush=True)
    print()

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


def analyze_nhop_overlap(matrices, output_file, n_hops=3, matrix1_index=None,
                         current_memory_gb=180, enable_path_tracking=True):
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
        current_memory_gb: Current memory tier (for path tracking)
        enable_path_tracking: Enable per-path completion tracking (default: True)
    """
    print(f"\n{'=' * 80}", flush=True)
    print(f"ANALYZING {n_hops}-HOP TO 1-HOP OVERLAP", flush=True)
    print(f"{'=' * 80}", flush=True)

    start_time = time.time()

    # Build extended matrix list with inverses
    print(f"[TIMING] Building matrix list...", flush=True)
    list_build_start = time.time()
    all_matrices, matrix_metadata = build_matrix_list(matrices)
    list_build_time = time.time() - list_build_start

    print(f"[TIMING] Matrix list built in {list_build_time:.1f}s", flush=True)
    print(f"Total matrices (with inverses): {len(all_matrices):,}", flush=True)
    print(f"Memory: {get_memory_mb():.0f} MB", flush=True)

    if matrix1_index is not None:
        print(f"Processing ONLY Matrix1 index: {matrix1_index}", flush=True)

    # Build aggregated 1-hop matrices
    print(f"\n[TIMING] Building aggregated 1-hop matrices...", flush=True)
    agg_start = time.time()
    aggregated_1hop = {}
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        key = (src_type, tgt_type)
        if key not in aggregated_1hop:
            aggregated_1hop[key] = matrix.dup()
        else:
            aggregated_1hop[key] = aggregated_1hop[key].ewise_add(matrix, gb.binary.any).new()
    agg_time = time.time() - agg_start

    print(f"[TIMING] Aggregated matrices built in {agg_time:.1f}s", flush=True)
    print(f"Created {len(aggregated_1hop):,} aggregated type-pair matrices", flush=True)

    # Group by source type for efficient lookup
    by_source_type = defaultdict(list)
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        by_source_type[src_type].append((src_type, pred, tgt_type, matrix, direction))

    # Load path tracking data (if enabled)
    completed_paths = set()
    failed_at_current_tier = set()
    results_dir = str(Path(output_file).parent)

    if enable_path_tracking and matrix1_index is not None:
        print(f"\n[TRACKING] Loading path tracking data...", flush=True)
        completed_paths = load_completed_paths(results_dir, matrix1_index)
        failed_at_current_tier = load_failed_paths(results_dir, matrix1_index, current_memory_gb)

        print(f"[TRACKING] Completed paths: {len(completed_paths):,}", flush=True)
        print(f"[TRACKING] Failed at {current_memory_gb}GB: {len(failed_at_current_tier):,}", flush=True)

        if len(completed_paths) > 0:
            print(f"[TRACKING] Will skip {len(completed_paths):,} already-completed paths", flush=True)
        if len(failed_at_current_tier) > 0:
            print(f"[TRACKING] Will skip {len(failed_at_current_tier):,} paths that failed at this tier", flush=True)

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

        print(f"\n[TIMING] Starting main computation loop for {len(matrix1_list)} starting matrices...", flush=True)
        computation_start = time.time()

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

                # Generate path ID for tracking
                if enable_path_tracking and matrix1_index is not None:
                    path_id = generate_path_id(node_types, predicates, directions)

                    # Check if already completed
                    if path_id in completed_paths:
                        return  # Skip, already have results

                    # Check if failed at this memory tier
                    if path_id in failed_at_current_tier:
                        return  # Skip, will retry at higher tier

                    # Record that we're computing this path
                    record_path_in_progress(path_id, results_dir, matrix1_index, n_hops, current_memory_gb)

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

                # CRITICAL: Flush after every path (not every 10k rows)
                f.flush()

                # Record path completion
                if enable_path_tracking and matrix1_index is not None:
                    record_completed_path(path_id, results_dir, matrix1_index)
                    clear_path_in_progress(results_dir, matrix1_index)

                return

            # Recursive case: continue building the path
            current_target_type = node_types[-1]

            if current_target_type not in by_source_type:
                return

            for src_type, pred, tgt_type, matrix, direction in by_source_type[current_target_type]:
                # Check dimension compatibility
                if accumulated_matrix.ncols != matrix.nrows:
                    continue

                # Build next path segment for tracking
                next_node_types = node_types + [tgt_type]
                next_predicates = predicates + [pred]
                next_directions = directions + [direction]

                try:
                    # Record in-progress for this branch (before potentially OOM operation)
                    if enable_path_tracking and matrix1_index is not None:
                        partial_path_id = generate_path_id(next_node_types, next_predicates, next_directions)
                        record_path_in_progress(partial_path_id, results_dir, matrix1_index,
                                              depth + 1, current_memory_gb)

                    # Multiply matrices (may OOM here)
                    next_result = accumulated_matrix.mxm(matrix, gb.semiring.any_pair).new()

                    if next_result.nvals == 0:
                        del next_result
                        continue

                    # Recurse to next depth
                    process_path(
                        depth + 1,
                        next_result,
                        next_node_types,
                        next_predicates,
                        next_directions,
                        first_matrix_nvals
                    )

                    # Clean up
                    del next_result
                    if depth % 2 == 0:  # Periodic garbage collection
                        gc.collect()

                except (MemoryError, gb.exceptions.OutOfMemory) as e:
                    # OOM during intermediate multiplication
                    # This means ALL downstream N-hop paths are impossible to compute at this tier
                    print(f"  [WARNING] OOM at depth {depth+1}, enumerating downstream paths...", flush=True)

                    if enable_path_tracking and matrix1_index is not None:
                        # Generate partial path ID up to this point
                        partial_path_id = generate_path_id(next_node_types, next_predicates, next_directions)

                        # Enumerate all possible N-hop completions of this partial path
                        downstream_paths = enumerate_downstream_paths(
                            partial_path_id,
                            all_matrices,
                            n_hops,
                            depth + 1
                        )

                        # Mark all downstream paths as failed at this tier
                        print(f"  [WARNING] Marking {len(downstream_paths)} downstream paths as failed at {current_memory_gb}GB", flush=True)
                        for complete_path_id in downstream_paths:
                            record_failed_path(
                                complete_path_id,
                                results_dir,
                                matrix1_index,
                                current_memory_gb,
                                depth=depth + 1,
                                reason="branch_oom"
                            )

                    # Continue to next matrix (don't crash entire job)
                    continue

                except Exception as e:
                    # Unexpected error - log but continue
                    print(f"  [ERROR] Unexpected error at depth {depth+1}: {type(e).__name__}: {e}", flush=True)
                    continue

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

        computation_time = time.time() - computation_start
        print(f"\n[TIMING] Main computation loop completed in {computation_time:.1f}s ({computation_time/60:.1f}min)", flush=True)

    print(f"\nDone! Wrote {rows_written:,} rows to {output_file}", flush=True)
    print(f"Total {n_hops}-hop paths computed: {paths_computed:,}", flush=True)

    total_analysis_time = time.time() - start_time
    print(f"\n[TIMING] Total analysis function time: {total_analysis_time:.1f}s ({total_analysis_time/60:.1f}min)", flush=True)
    print(f"[TIMING] Breakdown:", flush=True)
    print(f"  Matrix list build:  {list_build_time:7.1f}s ({list_build_time/total_analysis_time*100:5.1f}%)", flush=True)
    print(f"  Aggregation:        {agg_time:7.1f}s ({agg_time/total_analysis_time*100:5.1f}%)", flush=True)
    print(f"  Main computation:   {computation_time:7.1f}s ({computation_time/total_analysis_time*100:5.1f}%)", flush=True)


def analyze_3hop_overlap(matrices, output_file, matrix1_index=None):
    """Compute 3-hop metapaths and calculate overlap with 1-hop edges.

    This is a convenience wrapper around analyze_nhop_overlap with n_hops=3.
    Kept for backwards compatibility.

    Args:
        matrices: Dict of (src_type, pred, tgt_type) -> GraphBLAS matrix
        output_file: Path to output TSV file
        matrix1_index: Optional index to process only a single Matrix1 (for parallelization)
    """
    return analyze_nhop_overlap(matrices, output_file, n_hops=3, matrix1_index=matrix1_index,
                                current_memory_gb=180, enable_path_tracking=True)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze overlap between N-hop metapaths and 1-hop edges'
    )
    parser.add_argument('--edges', help='Path to edges.jsonl (required if --matrices-dir not provided)')
    parser.add_argument('--nodes', help='Path to nodes.jsonl (required if --matrices-dir not provided)')
    parser.add_argument('--matrices-dir', help='Directory with pre-built matrices (faster startup)')
    parser.add_argument('--output', required=True, help='Output TSV file path')
    parser.add_argument('--n-hops', type=int, default=3,
                        help='Number of hops for metapath analysis (default: 3)')
    parser.add_argument('--matrix1-index', type=int, default=None,
                        help='Process only this Matrix1 index (for parallelization)')
    parser.add_argument('--memory-gb', type=int, default=180,
                        help='Current SLURM memory tier in GB (for path tracking, default: 180)')
    parser.add_argument('--disable-path-tracking', action='store_true',
                        help='Disable per-path tracking (for testing or benchmarking)')

    args = parser.parse_args()

    # Validate arguments
    if not args.matrices_dir and (not args.edges or not args.nodes):
        parser.error("Either --matrices-dir OR (--edges AND --nodes) must be provided")

    print(f"\n{'=' * 80}", flush=True)
    print(f"STARTING {args.n_hops}-HOP ANALYSIS", flush=True)
    print(f"{'=' * 80}", flush=True)
    overall_start = time.time()

    # Load matrices (either prebuilt or from edges)
    if args.matrices_dir:
        print(f"\n[TIMING] Using pre-built matrices from {args.matrices_dir}...", flush=True)
        load_start = time.time()
        matrices = load_prebuilt_matrices(args.matrices_dir)
        load_time = time.time() - load_start

        node_load_time = 0
        matrix_build_time = load_time

        print(f"[TIMING] Pre-built matrix loading completed in {load_time:.1f}s ({load_time/60:.1f}min)", flush=True)
        print(f"[TIMING] Memory after loading: {get_memory_mb():.0f} MB", flush=True)
    else:
        print(f"\n[TIMING] Loading node types...", flush=True)
        node_load_start = time.time()
        node_types = load_node_types(args.nodes)
        node_load_time = time.time() - node_load_start
        print(f"[TIMING] Node loading completed in {node_load_time:.1f}s ({node_load_time/60:.1f}min)", flush=True)
        print(f"[TIMING] Memory after node loading: {get_memory_mb():.0f} MB", flush=True)

        print(f"\n[TIMING] Building matrices from edges...", flush=True)
        matrix_build_start = time.time()
        matrices = build_matrices(args.edges, node_types)
        matrix_build_time = time.time() - matrix_build_start
        print(f"[TIMING] Matrix building completed in {matrix_build_time:.1f}s ({matrix_build_time/60:.1f}min)", flush=True)
        print(f"[TIMING] Memory after matrix building: {get_memory_mb():.0f} MB", flush=True)

    # Analyze overlap
    print(f"\n[TIMING] Starting overlap analysis...", flush=True)
    analysis_start = time.time()
    enable_path_tracking = not args.disable_path_tracking
    if enable_path_tracking and args.matrix1_index is not None:
        print(f"[TRACKING] Path tracking ENABLED (memory tier: {args.memory_gb}GB)", flush=True)
    analyze_nhop_overlap(matrices, args.output, n_hops=args.n_hops, matrix1_index=args.matrix1_index,
                        current_memory_gb=args.memory_gb, enable_path_tracking=enable_path_tracking)
    analysis_time = time.time() - analysis_start
    print(f"\n[TIMING] Overlap analysis completed in {analysis_time:.1f}s ({analysis_time/60:.1f}min)", flush=True)

    overall_time = time.time() - overall_start
    print(f"\n{'=' * 80}", flush=True)
    print(f"TIMING SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"Node loading:      {node_load_time:8.1f}s ({node_load_time/overall_time*100:5.1f}%)", flush=True)
    print(f"Matrix building:   {matrix_build_time:8.1f}s ({matrix_build_time/overall_time*100:5.1f}%)", flush=True)
    print(f"Overlap analysis:  {analysis_time:8.1f}s ({analysis_time/overall_time*100:5.1f}%)", flush=True)
    print(f"{'-' * 80}", flush=True)
    print(f"Total:             {overall_time:8.1f}s ({overall_time/60:.1f}min)", flush=True)
    print(f"{'=' * 80}", flush=True)

    print(f"\nFinal memory: {get_memory_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
