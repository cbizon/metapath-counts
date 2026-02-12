#!/usr/bin/env python3
"""
Analyze overlap between N-hop metapaths and 1-hop edges.

Computes all N-hop metapaths via matrix multiplication, then calculates
overlap with 1-hop edges to identify which 1-hop edges appear in N-hop paths.

Output format (TSV):
  predictor_metapath     | predictor_count | predicted_metapath | predicted_count | overlap | total_possible
  SmallMolecule|affects|F|Gene|affects|R|SmallMolecule|affects|F|Gene | 6170000000 | SmallMolecule|regulates|F|Gene | 500000 | 450000 | 201000000000

Metapath format: NodeType|predicate|direction|NodeType|...
  - Pipe separated
  - F = forward, R = reverse, A = any (symmetric predicates)
  - Example: Disease|treats|R|SmallMolecule|affects|F|Gene
  - Symmetric example: Gene|directly_physically_interacts_with|A|Gene

Usage:
    # 3-hop analysis (default)
    uv run python analyze_hop_overlap.py \
        --matrices-dir matrices \
        --output results_3hop/results.tsv

    # 2-hop analysis
    uv run python analyze_hop_overlap.py \
        --matrices-dir matrices \
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
import itertools
from metapath_counts import (
    get_symmetric_predicates,
    is_pseudo_type,
    parse_pseudo_type,
    get_type_ancestors,
    get_predicate_ancestors,
    parse_metapath,
    build_metapath,
    generate_metapath_variants
)

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


def aggregate_explicit_results(explicit_file: str, aggregated_file: str):
    """
    Aggregate explicit results to all hierarchical variants.

    Reads explicit results, expands to variants, and writes aggregated results.
    """
    print(f"\n[AGGREGATION] Starting hierarchical aggregation...", flush=True)
    agg_start = time.time()

    # Read explicit results
    print(f"[AGGREGATION] Reading explicit results from {explicit_file}...", flush=True)
    explicit_results = []

    with open(explicit_file, 'r') as f:
        header = f.readline().strip()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 6:
                continue
            nhop_path, nhop_count, onehop_path, onehop_count, overlap, total_possible = parts
            explicit_results.append((
                nhop_path,
                int(nhop_count),
                onehop_path,
                int(onehop_count),
                int(overlap),
                int(total_possible)
            ))

    print(f"[AGGREGATION] Read {len(explicit_results):,} explicit results", flush=True)

    # Extract unique paths and their counts
    # Each explicit nhop/onehop path should only be counted ONCE, not once per comparison row
    unique_nhop_counts = {}
    unique_onehop_counts = {}
    for nhop_path, nhop_count, onehop_path, onehop_count, overlap, total_possible in explicit_results:
        if nhop_path not in unique_nhop_counts:
            unique_nhop_counts[nhop_path] = nhop_count
        if onehop_path not in unique_onehop_counts:
            unique_onehop_counts[onehop_path] = onehop_count

    print(f"[AGGREGATION] Found {len(unique_nhop_counts):,} unique N-hop paths, {len(unique_onehop_counts):,} unique 1-hop paths", flush=True)

    # Pre-compute variants for each unique path
    print(f"[AGGREGATION] Pre-computing variants for N-hop paths...", flush=True)
    nhop_variants_cache = {}
    for path in unique_nhop_counts:
        nhop_variants_cache[path] = list(generate_metapath_variants(path))

    print(f"[AGGREGATION] Pre-computing variants for 1-hop paths...", flush=True)
    onehop_variants_cache = {}
    for path in unique_onehop_counts:
        onehop_variants_cache[path] = list(generate_metapath_variants(path))

    # Step 1: Compute aggregated counts for each nhop variant (independent of onehop)
    print(f"[AGGREGATION] Computing N-hop variant counts...", flush=True)
    nhop_variant_counts = defaultdict(int)
    for nhop_path, nhop_count in unique_nhop_counts.items():
        for nhop_variant in nhop_variants_cache[nhop_path]:
            nhop_variant_counts[nhop_variant] += nhop_count

    # Step 2: Compute aggregated counts for each onehop variant (independent of nhop)
    print(f"[AGGREGATION] Computing 1-hop variant counts...", flush=True)
    onehop_variant_counts = defaultdict(int)
    for onehop_path, onehop_count in unique_onehop_counts.items():
        for onehop_variant in onehop_variants_cache[onehop_path]:
            onehop_variant_counts[onehop_variant] += onehop_count

    # Step 3: Aggregate overlap and total_possible for each (nhop_variant, onehop_variant) pair
    print(f"[AGGREGATION] Aggregating {len(explicit_results):,} pair results...", flush=True)
    pair_data = defaultdict(lambda: [0, 0])  # [overlap, total_possible]

    for i, (nhop_path, nhop_count, onehop_path, onehop_count, overlap, total_possible) in enumerate(explicit_results, 1):
        if i % 1000 == 0:
            print(f"[AGGREGATION] Processed {i:,}/{len(explicit_results):,} ({100*i/len(explicit_results):.1f}%)", flush=True)

        for nhop_variant in nhop_variants_cache[nhop_path]:
            for onehop_variant in onehop_variants_cache[onehop_path]:
                key = (nhop_variant, onehop_variant)
                pair_data[key][0] += overlap
                pair_data[key][1] += total_possible

    # Step 4: Combine into final aggregated results
    print(f"[AGGREGATION] Building final aggregated results...", flush=True)
    aggregated = {}
    for (nhop_variant, onehop_variant), (overlap, total_possible) in pair_data.items():
        aggregated[(nhop_variant, onehop_variant)] = [
            nhop_variant_counts[nhop_variant],
            onehop_variant_counts[onehop_variant],
            overlap,
            total_possible
        ]

    # Write aggregated results
    print(f"[AGGREGATION] Writing {len(aggregated):,} aggregated results to {aggregated_file}...", flush=True)
    with open(aggregated_file, 'w') as f:
        f.write(header + '\n')
        for (nhop_path, onehop_path), (nhop_count, onehop_count, overlap, total_possible) in aggregated.items():
            f.write(f"{nhop_path}\t{nhop_count}\t{onehop_path}\t{onehop_count}\t{overlap}\t{total_possible}\n")

    agg_time = time.time() - agg_start
    print(f"[AGGREGATION] Completed in {agg_time:.1f}s ({agg_time/60:.1f}min)", flush=True)
    print(f"[AGGREGATION] Expansion: {len(explicit_results):,} explicit → {len(aggregated):,} aggregated ({len(aggregated)/len(explicit_results):.1f}x)", flush=True)


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
        if last_matrix_nvals != first_matrix_nvals:
            return last_matrix_nvals > first_matrix_nvals
        # Tie-break equal sizes with canonical direction (same as 1-hop)
        return is_canonical_direction(src_type, tgt_type)


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


def determine_needed_matrices(manifest_path: Path, n_hops: int, matrix1_spec: tuple = None):
    """
    Determine which matrices are needed for N-hop analysis by simulating path enumeration.

    This uses the same chain-forming logic as the analysis, but on metadata only
    (without loading actual matrices) to determine which triples will participate.

    Args:
        manifest_path: Path to manifest.json
        n_hops: Number of hops
        matrix1_spec: If provided, (src_type, pred, direction, tgt_type) tuple for single starting matrix

    Returns:
        Set of (src_type, pred, tgt_type) tuples to load
    """
    print(f"Analyzing manifest to determine needed matrices for {n_hops}-hop...", flush=True)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    symmetric_predicates = get_symmetric_predicates()

    # Build directed matrix list (same as build_matrix_list but metadata only)
    all_directed = []
    for mat_info in manifest['matrices']:
        src_type = mat_info['src_type']
        pred = mat_info['predicate']
        tgt_type = mat_info['tgt_type']

        # Forward
        all_directed.append((src_type, pred, 'F', tgt_type))

        # Reverse (if not symmetric)
        if pred not in symmetric_predicates:
            all_directed.append((tgt_type, pred, 'R', src_type))

    print(f"Total directed matrices: {len(all_directed):,}", flush=True)

    # Group by source type (same as analysis code)
    by_source_type = defaultdict(list)
    for src_type, pred, direction, tgt_type in all_directed:
        by_source_type[src_type].append((src_type, pred, direction, tgt_type))

    # Track which base triples are needed
    needed_base_triples = set()
    final_type_pairs = set()  # Track (src_type, tgt_type) pairs for comparison matrices

    # Determine which matrix1s to process
    if matrix1_spec is not None:
        matrix1_list = [matrix1_spec]
        print(f"Processing single Matrix1: {matrix1_spec[0]}|{matrix1_spec[1]}|{matrix1_spec[2]}|{matrix1_spec[3]}", flush=True)
    else:
        matrix1_list = all_directed
        print(f"Processing all {len(matrix1_list):,} starting matrices", flush=True)

    # Recursive enumeration of paths (metadata only)
    def enumerate_paths(depth, node_types, predicates, directions):
        """Recursively enumerate N-hop paths without loading matrices."""
        nonlocal needed_base_triples, final_type_pairs

        if depth == n_hops:
            # Record final type pair for 1-hop comparison matrices
            src_type_final = node_types[0]
            tgt_type_final = node_types[-1]
            final_type_pairs.add((src_type_final, tgt_type_final))
            return

        # Continue building path
        current_target_type = node_types[-1]

        if current_target_type not in by_source_type:
            return

        for src_type, pred, direction, tgt_type in by_source_type[current_target_type]:
            # Record that we need this base triple
            if direction == 'F':
                needed_base_triples.add((src_type, pred, tgt_type))
            else:  # 'R'
                needed_base_triples.add((tgt_type, pred, src_type))

            # Recurse
            enumerate_paths(
                depth + 1,
                node_types + [tgt_type],
                predicates + [pred],
                directions + [direction]
            )

    # Enumerate all paths starting from each matrix1
    paths_enumerated = 0
    for src_type1, pred1, dir1, tgt_type1 in matrix1_list:
        # Record matrix1 itself
        if dir1 == 'F':
            needed_base_triples.add((src_type1, pred1, tgt_type1))
        else:
            needed_base_triples.add((tgt_type1, pred1, src_type1))

        # Enumerate downstream paths
        enumerate_paths(
            depth=1,
            node_types=[src_type1, tgt_type1],
            predicates=[pred1],
            directions=[dir1]
        )

        paths_enumerated += 1
        if paths_enumerated % 1000 == 0:
            print(f"  Enumerated paths from {paths_enumerated}/{len(matrix1_list)} starting matrices...", flush=True)

    # Add 1-hop comparison matrices
    print(f"Found {len(final_type_pairs):,} final (src_type, tgt_type) pairs for comparison", flush=True)
    for src_type, tgt_type in final_type_pairs:
        # Add all 1-hop matrices with this type pair
        for mat_src, mat_pred, mat_dir, mat_tgt in all_directed:
            if mat_src == src_type and mat_tgt == tgt_type:
                if mat_dir == 'F':
                    needed_base_triples.add((mat_src, mat_pred, mat_tgt))
                else:
                    needed_base_triples.add((mat_tgt, mat_pred, mat_src))

    print(f"Need to load {len(needed_base_triples)}/{manifest['num_matrices']} matrices", flush=True)
    print(f"Reduction: {100*(1 - len(needed_base_triples)/manifest['num_matrices']):.1f}% fewer matrices", flush=True)

    return needed_base_triples


def load_prebuilt_matrices(matrices_dir: str, needed_triples: set = None):
    """
    Load pre-built matrices from disk.

    Args:
        matrices_dir: Directory containing serialized matrices
        needed_triples: Optional set of (src_type, pred, tgt_type) to load.
                       If None, loads all matrices.

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

    if needed_triples is None:
        print(f"Loading ALL {manifest['num_matrices']} matrices ({manifest['total_size_bytes']/1e9:.2f} GB)", flush=True)
        to_load = manifest['matrices']
    else:
        print(f"Loading {len(needed_triples)} of {manifest['num_matrices']} matrices", flush=True)
        # Filter manifest to only needed matrices
        to_load = [m for m in manifest['matrices']
                   if (m['src_type'], m['predicate'], m['tgt_type']) in needed_triples]

        # Print reduction percentage
        reduction_pct = 100 * (1 - len(to_load) / manifest['num_matrices'])
        print(f"Reduction: {reduction_pct:.1f}% fewer matrices to load ({len(to_load)}/{manifest['num_matrices']})", flush=True)

    # Load each matrix
    matrices = {}
    for i, mat_info in enumerate(to_load, 1):
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
            print(f"  Loaded {i}/{len(to_load)} matrices", flush=True)

    load_time = time.time() - start_time
    print(f"Loaded {len(matrices):,} matrices in {load_time:.1f}s", flush=True)
    print(f"Memory: {get_memory_mb():.0f} MB", flush=True)
    print()

    return matrices


def build_matrix_list(matrices):
    """Build extended matrix list with forward and reverse directions."""
    all_matrices = []
    matrix_metadata = {}  # (src_type, pred, tgt_type, direction) -> matrix

    # Get symmetric predicates from biolink model
    symmetric_predicates = get_symmetric_predicates()

    for (src_type, pred, tgt_type), matrix in matrices.items():
        # Forward
        all_matrices.append((src_type, pred, tgt_type, matrix, 'F'))
        matrix_metadata[(src_type, pred, tgt_type, 'F')] = matrix

        # Reverse (if not symmetric)
        is_symmetric = pred in symmetric_predicates
        if not is_symmetric:
            all_matrices.append((tgt_type, pred, src_type, matrix.T, 'R'))
            matrix_metadata[(tgt_type, pred, src_type, 'R')] = matrix.T

    return all_matrices, matrix_metadata


def analyze_nhop_overlap(matrices, output_file, n_hops=3, matrix1_index=None, matrix1_spec=None,
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
        matrix1_index: Optional index for path tracking (legacy, use matrix1_spec instead)
        matrix1_spec: Optional (src_type, pred, direction, tgt_type) tuple to process single Matrix1
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

    if matrix1_spec is not None:
        print(f"Processing ONLY Matrix1: {matrix1_spec[0]}|{matrix1_spec[1]}|{matrix1_spec[2]}|{matrix1_spec[3]}", flush=True)
    elif matrix1_index is not None:
        print(f"Processing ONLY Matrix1 index: {matrix1_index}", flush=True)


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
        f.write("predictor_metapath\tpredictor_count\tpredicted_metapath\tpredicted_count\toverlap\ttotal_possible\n")

        rows_written = 0
        paths_computed = 0

        # Determine which Matrix1 indices to process
        if matrix1_spec is not None:
            # Find the matrix matching the spec
            src_spec, pred_spec, dir_spec, tgt_spec = matrix1_spec
            found_idx = None
            for idx, (src, pred, tgt, mat, direction) in enumerate(all_matrices):
                if src == src_spec and pred == pred_spec and tgt == tgt_spec and direction == dir_spec:
                    found_idx = idx
                    break

            if found_idx is None:
                raise ValueError(f"Matrix1 spec {matrix1_spec} not found in loaded matrices")

            # Use matrix1_index for tracking if provided, otherwise use found_idx
            tracking_idx = matrix1_index if matrix1_index is not None else found_idx
            matrix1_list = [(tracking_idx, all_matrices[found_idx])]
            print(f"Found Matrix1 at index {found_idx} in loaded matrices (tracking as job {tracking_idx})", flush=True)
        elif matrix1_index is not None:
            # Legacy: using direct index (only works if all matrices are loaded)
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

                # For n_hops == 1: duplicate elimination (alphabetical) happens here
                # because there is no recursive step. For n_hops > 1 the check already
                # happened before the final multiplication (see recursive case below).
                if n_hops == 1:
                    if not should_process_path(1, first_matrix_nvals, accumulated_matrix.nvals,
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
                for (onehop_src, onehop_pred, onehop_tgt, onehop_dir), onehop_matrix in matrix_metadata.items():
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

                    # Skip zero-overlap rows (optimization: reduces output size significantly)
                    if overlap_count == 0:
                        continue

                    # Write row
                    f.write(f"{nhop_metapath}\t{nhop_count}\t{onehop_metapath}\t{onehop_count}\t{overlap_count}\t{total_possible}\n")
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

                # For the final hop: duplicate elimination before multiplying.
                # Compare the last hop matrix nvals against the first matrix nvals
                # to decide which direction is canonical. This avoids doing the
                # expensive matrix multiplication for the non-canonical direction.
                if depth == n_hops - 1:
                    if not should_process_path(n_hops, first_matrix_nvals, matrix.nvals,
                                              node_types[0], tgt_type):
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

    print(f"\nDone! Wrote {rows_written:,} explicit rows to {output_file}", flush=True)
    print(f"Total {n_hops}-hop paths computed: {paths_computed:,}", flush=True)

    # NOTE: Hierarchical aggregation is now done ONLY during the grouping step,
    # using precomputed aggregated counts from prepare_grouping.py.
    # This avoids double-aggregation and ensures correct count computation.
    # The per-job files contain EXPLICIT results only.

    total_analysis_time = time.time() - start_time
    print(f"\n[TIMING] Total analysis function time: {total_analysis_time:.1f}s ({total_analysis_time/60:.1f}min)", flush=True)
    print(f"[TIMING] Breakdown:", flush=True)
    print(f"  Matrix list build:  {list_build_time:7.1f}s ({list_build_time/total_analysis_time*100:5.1f}%)", flush=True)
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
    parser.add_argument('--matrices-dir', required=True,
                        help='Directory with pre-built matrices')
    parser.add_argument('--output', required=True, help='Output TSV file path')
    parser.add_argument('--n-hops', type=int, default=3,
                        help='Number of hops for metapath analysis (default: 3)')
    parser.add_argument('--matrix1-index', type=int, default=None,
                        help='Process only this Matrix1 index (for parallelization)')
    parser.add_argument('--src-type', type=str, default=None,
                        help='Source node type for matrix1 (required with --matrix1-index)')
    parser.add_argument('--pred', type=str, default=None,
                        help='Predicate for matrix1 (required with --matrix1-index)')
    parser.add_argument('--direction', type=str, default=None,
                        help='Direction for matrix1: F, R, or A (required with --matrix1-index)')
    parser.add_argument('--tgt-type', type=str, default=None,
                        help='Target node type for matrix1 (required with --matrix1-index)')
    parser.add_argument('--memory-gb', type=int, default=180,
                        help='Current SLURM memory tier in GB (for path tracking, default: 180)')
    parser.add_argument('--disable-path-tracking', action='store_true',
                        help='Disable per-path tracking (for testing or benchmarking)')

    args = parser.parse_args()

    # Validate matrix1 spec arguments
    if args.matrix1_index is not None:
        if not all([args.src_type, args.pred, args.direction, args.tgt_type]):
            parser.error("--matrix1-index requires --src-type, --pred, --direction, and --tgt-type")

    print(f"\n{'=' * 80}", flush=True)
    print(f"STARTING {args.n_hops}-HOP ANALYSIS", flush=True)
    print(f"{'=' * 80}", flush=True)
    overall_start = time.time()

    # Determine which matrices are actually needed
    print(f"\n[TIMING] Using pre-built matrices from {args.matrices_dir}...", flush=True)
    matrices_path = Path(args.matrices_dir)
    manifest_path = matrices_path / 'manifest.json'

    # Build matrix1_spec from CLI arguments if processing a specific matrix
    matrix1_spec = None
    if args.matrix1_index is not None:
        matrix1_spec = (args.src_type, args.pred, args.direction, args.tgt_type)
        print(f"Matrix1 spec: {matrix1_spec}", flush=True)

    print(f"\n[TIMING] Determining needed matrices...", flush=True)
    filter_start = time.time()
    needed_triples = determine_needed_matrices(manifest_path, args.n_hops, matrix1_spec)
    filter_time = time.time() - filter_start
    print(f"[TIMING] Matrix filtering completed in {filter_time:.1f}s", flush=True)

    # Load only needed matrices
    load_start = time.time()
    matrices = load_prebuilt_matrices(args.matrices_dir, needed_triples=needed_triples)
    load_time = time.time() - load_start

    print(f"[TIMING] Matrix loading completed in {load_time:.1f}s ({load_time/60:.1f}min)", flush=True)
    print(f"[TIMING] Memory after loading: {get_memory_mb():.0f} MB", flush=True)

    # Analyze overlap
    print(f"\n[TIMING] Starting overlap analysis...", flush=True)
    analysis_start = time.time()
    enable_path_tracking = not args.disable_path_tracking
    if enable_path_tracking and args.matrix1_index is not None:
        print(f"[TRACKING] Path tracking ENABLED (memory tier: {args.memory_gb}GB)", flush=True)

    analyze_nhop_overlap(
        matrices, args.output, n_hops=args.n_hops,
        matrix1_index=args.matrix1_index,
        matrix1_spec=matrix1_spec,
        current_memory_gb=args.memory_gb,
        enable_path_tracking=enable_path_tracking
    )
    analysis_time = time.time() - analysis_start
    print(f"\n[TIMING] Overlap analysis completed in {analysis_time:.1f}s ({analysis_time/60:.1f}min)", flush=True)

    overall_time = time.time() - overall_start
    print(f"\n{'=' * 80}", flush=True)
    print(f"TIMING SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"Matrix loading:    {load_time:8.1f}s ({load_time/overall_time*100:5.1f}%)", flush=True)
    print(f"Overlap analysis:  {analysis_time:8.1f}s ({analysis_time/overall_time*100:5.1f}%)", flush=True)
    print(f"{'-' * 80}", flush=True)
    print(f"Total:             {overall_time:8.1f}s ({overall_time/60:.1f}min)", flush=True)
    print(f"{'=' * 80}", flush=True)

    print(f"\nFinal memory: {get_memory_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
