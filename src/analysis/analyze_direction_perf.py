#!/usr/bin/env python3
"""
Analyze whether computing 3-hop paths in forward vs reverse direction
would have different performance characteristics.

For path A -> B -> C -> D (3 edges, 4 nodes):
- Forward: (MatrixAB @ MatrixBC) @ MatrixCD
  - Intermediate: MatrixAB @ MatrixBC (connects A to C)
- Reverse: (MatrixDC^T @ MatrixCB^T) @ MatrixBA^T
  - Intermediate: MatrixDC^T @ MatrixCB^T (connects D to B)

Key insight: The intermediate matrix size/sparsity determines performance.
If (AB @ BC) is much larger than (DC^T @ CB^T), reverse would be faster.

This script:
1. Loads all matrices (forward + reverse)
2. For each valid 3-hop combination, computes BOTH intermediate results
3. Compares size and estimates which direction would be faster
4. Outputs statistics on how often reverse is better
"""

import argparse
import json
import time
import os
import gc
import random
import psutil
from collections import defaultdict
import graphblas as gb
from library import get_most_specific_type

SYMMETRIC_PREDICATES = {
    'biolink:interacts_with',
    'biolink:coexists_with',
    'biolink:correlated_with',
    'biolink:associated_with',
    'biolink:related_to',
    'biolink:similar_to',
    'biolink:homologous_to',
    'biolink:orthologous_to',
    'biolink:paralogous_to',
    'biolink:xenologous_to',
}


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

    edge_triples = defaultdict(list)
    node_to_idx = defaultdict(dict)

    with open(edges_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1_000_000 == 0:
                print(f"  Processed {line_num:,} edges", flush=True)

            edge = json.loads(line)
            predicate = edge.get('predicate', '')

            if predicate == 'biolink:subclass_of':
                continue

            subject = edge['subject']
            obj = edge['object']
            src_type = node_types.get(subject)
            tgt_type = node_types.get(obj)

            if not src_type or not tgt_type:
                continue

            pred = predicate.replace('biolink:', '')
            triple = (src_type, pred, tgt_type)
            edge_triples[triple].append((subject, obj))

            if subject not in node_to_idx[src_type]:
                idx = len(node_to_idx[src_type])
                node_to_idx[src_type][subject] = idx

            if obj not in node_to_idx[tgt_type]:
                idx = len(node_to_idx[tgt_type])
                node_to_idx[tgt_type][obj] = idx

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


def get_matrix_memory_mb(matrix):
    """
    Estimate memory usage of a GraphBLAS sparse matrix in MB.

    For BOOL dtype sparse matrix: ~17 bytes per entry
    """
    return (matrix.nvals * 17) / (1024 * 1024)


def analyze_direction_performance(matrices, max_samples=1000):
    """
    Compare ACTUAL runtime and memory usage for forward vs reverse direction.

    For each sampled 3-hop path:
    - Forward: times (M1 @ M2) and (result @ M3), measures memory at each step
    - Reverse: times (M3^T @ M2^T) and (result @ M1^T), measures memory at each step
    - Records real performance data, not estimates
    """
    print(f"\n{'=' * 80}", flush=True)
    print("ANALYZING FORWARD VS REVERSE PERFORMANCE (REAL TIMING)", flush=True)
    print(f"{'=' * 80}", flush=True)

    # Build extended matrix list with inverses
    all_matrices = []
    for (src_type, pred, tgt_type), matrix in matrices.items():
        # Forward
        all_matrices.append((src_type, pred, tgt_type, matrix, 'F'))

        # Reverse (if not symmetric)
        is_symmetric = f'biolink:{pred}' in SYMMETRIC_PREDICATES
        if not is_symmetric:
            all_matrices.append((tgt_type, pred, src_type, matrix.T, 'R'))

    print(f"Total matrices (with inverses): {len(all_matrices):,}", flush=True)

    # Group by source type
    by_source_type = defaultdict(list)
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        by_source_type[src_type].append((src_type, pred, tgt_type, matrix, direction))

    # Pre-enumerate all valid 3-hop combinations (by type compatibility only)
    print(f"\nEnumerating all valid 3-hop combinations (by type)...", flush=True)
    valid_combinations = []

    for src_type1, pred1, tgt_type1, matrix1, dir1 in all_matrices:
        if tgt_type1 not in by_source_type:
            continue

        for src_type2, pred2, tgt_type2, matrix2, dir2 in by_source_type[tgt_type1]:
            if tgt_type2 not in by_source_type:
                continue

            for src_type3, pred3, tgt_type3, matrix3, dir3 in by_source_type[tgt_type2]:
                # Only check type compatibility, not dimensions (that's checked during processing)
                valid_combinations.append((
                    (src_type1, pred1, tgt_type1, matrix1, dir1),
                    (src_type2, pred2, tgt_type2, matrix2, dir2),
                    (src_type3, pred3, tgt_type3, matrix3, dir3)
                ))

    print(f"Found {len(valid_combinations):,} valid 3-hop combinations (by type)", flush=True)

    # Count amortization opportunities for both directions
    print(f"Counting amortization opportunities...", flush=True)
    m1m2_to_m3_count = defaultdict(int)  # Forward: how many M3 per (M1, M2)
    m2m3_to_m1_count = defaultdict(int)  # Reverse: how many M1 per (M2, M3)

    for (src_type1, pred1, tgt_type1, matrix1, dir1), \
        (src_type2, pred2, tgt_type2, matrix2, dir2), \
        (src_type3, pred3, tgt_type3, matrix3, dir3) in valid_combinations:
        m1m2_key = (id(matrix1), id(matrix2))
        m2m3_key = (id(matrix2), id(matrix3))
        m1m2_to_m3_count[m1m2_key] += 1
        m2m3_to_m1_count[m2m3_key] += 1

    print(f"Found {len(m1m2_to_m3_count):,} unique (M1, M2) pairs", flush=True)
    print(f"Found {len(m2m3_to_m1_count):,} unique (M2, M3) pairs", flush=True)

    # Randomly sample from valid combinations
    num_samples = min(max_samples, len(valid_combinations))
    print(f"Randomly sampling {num_samples:,} combinations...", flush=True)
    sampled_combinations = random.sample(valid_combinations, num_samples)

    # Sample valid 3-hop paths and compare
    samples_checked = 0
    forward_faster = 0
    reverse_faster = 0
    equal = 0

    # Track timing statistics
    forward_time_list = []
    reverse_time_list = []
    forward_step1_time_list = []
    forward_step2_time_list = []
    reverse_step1_time_list = []
    reverse_step2_time_list = []

    # Track memory statistics
    forward_mem_list = []
    reverse_mem_list = []

    with open('direction_analysis.tsv', 'w') as f:
        f.write("forward_metapath\t"
                "m1_nrows\tm1_ncols\tm1_nvals\t"
                "m2_nrows\tm2_ncols\tm2_nvals\t"
                "m3_nrows\tm3_ncols\tm3_nvals\t"
                "forward_amortize_count\treverse_amortize_count\t"
                "forward_step1_time\tforward_step1_edges\tforward_step1_mem_mb\t"
                "forward_step2_time\tforward_result_edges\tforward_step2_mem_mb\t"
                "forward_total_time\tforward_amortized_time\t"
                "reverse_metapath\t"
                "reverse_step1_time\treverse_step1_edges\treverse_step1_mem_mb\t"
                "reverse_step2_time\treverse_result_edges\treverse_step2_mem_mb\t"
                "reverse_total_time\treverse_amortized_time\t"
                "better_direction\ttime_ratio\tamortized_better\tamortized_ratio\n")

        for (src_type1, pred1, tgt_type1, matrix1, dir1), \
            (src_type2, pred2, tgt_type2, matrix2, dir2), \
            (src_type3, pred3, tgt_type3, matrix3, dir3) in sampled_combinations:

            # ===================================================================
            # FORWARD DIRECTION: (M1 @ M2) @ M3
            # ===================================================================

            mem_before_forward = get_memory_mb()

            # Step 1: M1 @ M2
            start_time = time.time()
            forward_intermediate = matrix1.mxm(matrix2, gb.semiring.any_pair).new()
            forward_step1_time = time.time() - start_time
            forward_step1_edges = forward_intermediate.nvals
            forward_step1_mem = get_matrix_memory_mb(forward_intermediate)
            mem_after_step1 = get_memory_mb()

            # Step 2: intermediate @ M3 (do this even if intermediate is empty - it still takes time!)
            start_time = time.time()
            forward_result = forward_intermediate.mxm(matrix3, gb.semiring.any_pair).new()
            forward_step2_time = time.time() - start_time
            forward_result_edges = forward_result.nvals
            forward_step2_mem = get_matrix_memory_mb(forward_result)
            mem_after_forward = get_memory_mb()

            forward_total_time = forward_step1_time + forward_step2_time
            forward_peak_mem = mem_after_forward - mem_before_forward

            # ===================================================================
            # REVERSE DIRECTION: (M3^T @ M2^T) @ M1^T
            # ===================================================================

            mem_before_reverse = get_memory_mb()

            # Step 1: M3^T @ M2^T
            start_time = time.time()
            reverse_intermediate = matrix3.T.mxm(matrix2.T, gb.semiring.any_pair).new()
            reverse_step1_time = time.time() - start_time
            reverse_step1_edges = reverse_intermediate.nvals
            reverse_step1_mem = get_matrix_memory_mb(reverse_intermediate)
            mem_after_step1_rev = get_memory_mb()

            # Step 2: intermediate @ M1^T (do this even if intermediate is empty - it still takes time!)
            start_time = time.time()
            reverse_result = reverse_intermediate.mxm(matrix1.T, gb.semiring.any_pair).new()
            reverse_step2_time = time.time() - start_time
            reverse_result_edges = reverse_result.nvals
            reverse_step2_mem = get_matrix_memory_mb(reverse_result)
            mem_after_reverse = get_memory_mb()

            reverse_total_time = reverse_step1_time + reverse_step2_time
            reverse_peak_mem = mem_after_reverse - mem_before_reverse

            # ===================================================================
            # COMPARISON
            # ===================================================================

            # Track timing distributions
            forward_time_list.append(forward_total_time)
            reverse_time_list.append(reverse_total_time)
            forward_step1_time_list.append(forward_step1_time)
            forward_step2_time_list.append(forward_step2_time)
            reverse_step1_time_list.append(reverse_step1_time)
            reverse_step2_time_list.append(reverse_step2_time)

            # Track memory distributions
            forward_mem_list.append(forward_peak_mem)
            reverse_mem_list.append(reverse_peak_mem)

            # Calculate time ratio (forward / reverse)
            if reverse_total_time > 0:
                time_ratio = forward_total_time / reverse_total_time
            else:
                time_ratio = float('inf') if forward_total_time > 0 else 1.0

            # Determine which is better (faster)
            if forward_total_time < reverse_total_time:
                better = "forward"
                forward_faster += 1
            elif reverse_total_time < forward_total_time:
                better = "reverse"
                reverse_faster += 1
            else:
                better = "equal"
                equal += 1

            # Calculate amortization counts
            m1m2_key = (id(matrix1), id(matrix2))
            m2m3_key = (id(matrix2), id(matrix3))
            forward_amortize = m1m2_to_m3_count[m1m2_key]
            reverse_amortize = m2m3_to_m1_count[m2m3_key]

            # Calculate amortized times
            forward_amortized_time = (forward_step1_time / forward_amortize) + forward_step2_time
            reverse_amortized_time = (reverse_step1_time / reverse_amortize) + reverse_step2_time

            # Determine better direction with amortization
            if forward_amortized_time < reverse_amortized_time:
                amortized_better = "forward"
            elif reverse_amortized_time < forward_amortized_time:
                amortized_better = "reverse"
            else:
                amortized_better = "equal"

            amortized_ratio = forward_amortized_time / reverse_amortized_time if reverse_amortized_time > 0 else float('inf')

            # Format metapaths
            forward_path = f"{src_type1}|{pred1}|{dir1}|{tgt_type1}|{pred2}|{dir2}|{tgt_type2}|{pred3}|{dir3}|{tgt_type3}"
            reverse_path = f"{tgt_type3}|{pred3}|{'F' if dir3=='R' else 'R'}|{tgt_type2}|{pred2}|{'F' if dir2=='R' else 'R'}|{tgt_type1}|{pred1}|{'F' if dir1=='R' else 'R'}|{src_type1}"

            # Write results
            f.write(f"{forward_path}\t"
                    f"{matrix1.nrows}\t{matrix1.ncols}\t{matrix1.nvals}\t"
                    f"{matrix2.nrows}\t{matrix2.ncols}\t{matrix2.nvals}\t"
                    f"{matrix3.nrows}\t{matrix3.ncols}\t{matrix3.nvals}\t"
                    f"{forward_amortize}\t{reverse_amortize}\t"
                    f"{forward_step1_time:.6f}\t{forward_step1_edges}\t{forward_step1_mem:.3f}\t"
                    f"{forward_step2_time:.6f}\t{forward_result_edges}\t{forward_step2_mem:.3f}\t"
                    f"{forward_total_time:.6f}\t{forward_amortized_time:.6f}\t"
                    f"{reverse_path}\t"
                    f"{reverse_step1_time:.6f}\t{reverse_step1_edges}\t{reverse_step1_mem:.3f}\t"
                    f"{reverse_step2_time:.6f}\t{reverse_result_edges}\t{reverse_step2_mem:.3f}\t"
                    f"{reverse_total_time:.6f}\t{reverse_amortized_time:.6f}\t"
                    f"{better}\t{time_ratio:.3f}\t{amortized_better}\t{amortized_ratio:.3f}\n")

            samples_checked += 1

            # Force garbage collection to free C memory from GraphBLAS matrices
            if samples_checked % 5 == 0:
                gc.collect()

            if samples_checked % 10 == 0:
                print(f"Checked {samples_checked:,} paths | "
                      f"Forward faster: {forward_faster} | "
                      f"Reverse faster: {reverse_faster} | "
                      f"Equal: {equal} | "
                      f"Avg speedup: {sum(reverse_time_list)/sum(forward_time_list) if forward_time_list else 1:.2f}x | "
                      f"Mem: {get_memory_mb():.0f}MB",
                      flush=True)
                f.flush()  # Flush every 10 samples instead of 100

    # Calculate statistics
    import statistics

    forward_time_list.sort()
    reverse_time_list.sort()
    forward_mem_list.sort()
    reverse_mem_list.sort()

    def percentile(data, p):
        """Calculate percentile p (0-100) from sorted data."""
        if not data:
            return 0
        k = (len(data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1 if (f + 1) < len(data) else f
        if f == c:
            return data[f]
        return data[f] * (c - k) + data[c] * (k - f)

    print(f"\n{'=' * 80}", flush=True)
    print("RESULTS: ACTUAL PERFORMANCE COMPARISON", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"Total paths analyzed: {samples_checked:,}")
    print(f"Forward faster: {forward_faster} ({forward_faster/samples_checked*100:.1f}%)")
    print(f"Reverse faster: {reverse_faster} ({reverse_faster/samples_checked*100:.1f}%)")
    print(f"Equal: {equal} ({equal/samples_checked*100:.1f}%)")

    if forward_time_list and reverse_time_list:
        total_forward_time = sum(forward_time_list)
        total_reverse_time = sum(reverse_time_list)
        print(f"\nTotal time - Forward: {total_forward_time:.2f}s | Reverse: {total_reverse_time:.2f}s")
        print(f"Average speedup (reverse vs forward): {total_forward_time/total_reverse_time:.2f}x")

    print(f"\n{'=' * 80}", flush=True)
    print("TIMING STATISTICS (seconds per complete 3-hop path)", flush=True)
    print(f"{'=' * 80}", flush=True)

    print(f"\nFORWARD DIRECTION:")
    print(f"  Min:     {min(forward_time_list):.6f} s")
    print(f"  p25:     {percentile(forward_time_list, 25):.6f} s")
    print(f"  Median:  {percentile(forward_time_list, 50):.6f} s")
    print(f"  Mean:    {statistics.mean(forward_time_list):.6f} s")
    print(f"  p75:     {percentile(forward_time_list, 75):.6f} s")
    print(f"  p90:     {percentile(forward_time_list, 90):.6f} s")
    print(f"  p95:     {percentile(forward_time_list, 95):.6f} s")
    print(f"  p99:     {percentile(forward_time_list, 99):.6f} s")
    print(f"  Max:     {max(forward_time_list):.6f} s")

    print(f"\nREVERSE DIRECTION:")
    print(f"  Min:     {min(reverse_time_list):.6f} s")
    print(f"  p25:     {percentile(reverse_time_list, 25):.6f} s")
    print(f"  Median:  {percentile(reverse_time_list, 50):.6f} s")
    print(f"  Mean:    {statistics.mean(reverse_time_list):.6f} s")
    print(f"  p75:     {percentile(reverse_time_list, 75):.6f} s")
    print(f"  p90:     {percentile(reverse_time_list, 90):.6f} s")
    print(f"  p95:     {percentile(reverse_time_list, 95):.6f} s")
    print(f"  p99:     {percentile(reverse_time_list, 99):.6f} s")
    print(f"  Max:     {max(reverse_time_list):.6f} s")

    print(f"\n{'=' * 80}", flush=True)
    print("MEMORY STATISTICS (peak MB change during computation)", flush=True)
    print(f"{'=' * 80}", flush=True)

    print(f"\nFORWARD DIRECTION:")
    print(f"  Min:     {min(forward_mem_list):.2f} MB")
    print(f"  p25:     {percentile(forward_mem_list, 25):.2f} MB")
    print(f"  Median:  {percentile(forward_mem_list, 50):.2f} MB")
    print(f"  Mean:    {statistics.mean(forward_mem_list):.2f} MB")
    print(f"  p75:     {percentile(forward_mem_list, 75):.2f} MB")
    print(f"  p90:     {percentile(forward_mem_list, 90):.2f} MB")
    print(f"  p95:     {percentile(forward_mem_list, 95):.2f} MB")
    print(f"  p99:     {percentile(forward_mem_list, 99):.2f} MB")
    print(f"  Max:     {max(forward_mem_list):.2f} MB")

    print(f"\nREVERSE DIRECTION:")
    print(f"  Min:     {min(reverse_mem_list):.2f} MB")
    print(f"  p25:     {percentile(reverse_mem_list, 25):.2f} MB")
    print(f"  Median:  {percentile(reverse_mem_list, 50):.2f} MB")
    print(f"  Mean:    {statistics.mean(reverse_mem_list):.2f} MB")
    print(f"  p75:     {percentile(reverse_mem_list, 75):.2f} MB")
    print(f"  p90:     {percentile(reverse_mem_list, 90):.2f} MB")
    print(f"  p95:     {percentile(reverse_mem_list, 95):.2f} MB")
    print(f"  p99:     {percentile(reverse_mem_list, 99):.2f} MB")
    print(f"  Max:     {max(reverse_mem_list):.2f} MB")

    print(f"\n{'=' * 80}", flush=True)
    print("PARALLELIZATION POTENTIAL (by memory budget)", flush=True)
    print(f"{'=' * 80}", flush=True)

    def count_fits_in_memory(data, memory_budget_mb):
        """Count how many paths fit within memory budget."""
        return sum(1 for x in data if x <= memory_budget_mb)

    memory_budgets = [100, 500, 1000, 2000, 4000]
    print(f"\n{'Budget (MB)':<15} {'Forward Fits':<15} {'Reverse Fits':<15} {'Better':<10}")
    print(f"{'-'*55}")

    for budget in memory_budgets:
        forward_fits = count_fits_in_memory(forward_mem_list, budget)
        reverse_fits = count_fits_in_memory(reverse_mem_list, budget)
        forward_pct = forward_fits / samples_checked * 100
        reverse_pct = reverse_fits / samples_checked * 100

        better = "Forward" if forward_fits > reverse_fits else "Reverse" if reverse_fits > forward_fits else "Equal"

        print(f"{budget:<15} {forward_fits:>6} ({forward_pct:>5.1f}%)  "
              f"{reverse_fits:>6} ({reverse_pct:>5.1f}%)  {better:<10}")

    print(f"\n{'=' * 80}", flush=True)
    print("KEY INSIGHTS", flush=True)
    print(f"{'=' * 80}", flush=True)

    if forward_faster > reverse_faster:
        winner = "FORWARD"
        win_pct = forward_faster / samples_checked * 100
        avg_speedup = sum(reverse_time_list) / sum(forward_time_list)
    elif reverse_faster > forward_faster:
        winner = "REVERSE"
        win_pct = reverse_faster / samples_checked * 100
        avg_speedup = sum(forward_time_list) / sum(reverse_time_list)
    else:
        winner = "TIE"
        win_pct = 50.0
        avg_speedup = 1.0

    print(f"\n1. WINNER: {winner} direction is faster {win_pct:.1f}% of the time")
    print(f"   Average speedup: {avg_speedup:.2f}x faster")

    print(f"\n2. MEMORY: With 1GB/worker memory budget:")
    f1000 = count_fits_in_memory(forward_mem_list, 1000)
    r1000 = count_fits_in_memory(reverse_mem_list, 1000)
    print(f"   Forward can parallelize {f1000}/{samples_checked} paths ({f1000/samples_checked*100:.1f}%)")
    print(f"   Reverse can parallelize {r1000}/{samples_checked} paths ({r1000/samples_checked*100:.1f}%)")

    print(f"\n3. RECOMMENDATION:")
    if winner == "FORWARD":
        print(f"   Compute 3-hop paths in FORWARD direction only (cut computation ~2x)")
    elif winner == "REVERSE":
        print(f"   Compute 3-hop paths in REVERSE direction only (cut computation ~2x)")
    else:
        print(f"   Implement adaptive direction selection based on intermediate size")

    print(f"\n{'=' * 80}", flush=True)
    print(f"Wrote detailed results to direction_analysis.tsv")
    print(f"{'=' * 80}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze whether forward or reverse direction is faster for 3-hop paths'
    )
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Maximum number of 3-hop paths to sample')

    args = parser.parse_args()

    # Load data and build matrices
    node_types = load_node_types(args.nodes)
    matrices = build_matrices(args.edges, node_types)

    # Analyze direction performance
    analyze_direction_performance(matrices, args.max_samples)


if __name__ == "__main__":
    main()
