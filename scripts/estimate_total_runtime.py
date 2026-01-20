#!/usr/bin/env python3
"""
Estimate total runtime by computing size distribution of all Matrix3 iterations.

Strategy:
1. For each valid Matrix1/Matrix2 combo, compute A@B (fast)
2. Count how many Matrix3 iterations will use each A@B
3. Bucket by result size
4. Use benchmark timing to estimate total runtime

Output: Distribution of iteration sizes and estimated total time

Usage:
    uv run python estimate_total_runtime.py \
        --edges ../SimplePredictions/input_graphs/robokop_base_nonredundant/edges.jsonl \
        --nodes ../SimplePredictions/input_graphs/robokop_base_nonredundant/nodes.jsonl \
        --benchmark matrix_timing_benchmark.tsv
"""

import argparse
import json
import time
from collections import defaultdict, Counter
import graphblas as gb
from metapath_counts import get_most_specific_type


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
    """Build sparse matrices."""
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


def load_benchmark_timings(benchmark_file):
    """Load timing data from benchmark results file.

    Expected format from run_benchmark_samples.py:
    bucket | metapath | ab_edges | abc_edges | num_comparisons | ab_time | abc_time | comparison_time | total_time
    """
    if benchmark_file is None:
        return None

    print(f"\nLoading benchmark timings from {benchmark_file}...", flush=True)

    from collections import defaultdict
    bucket_timings = defaultdict(list)

    with open(benchmark_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 9:
                bucket = parts[0]
                total_time = float(parts[8])
                bucket_timings[bucket].append(total_time)

    # Calculate average per bucket
    bucket_averages = {}
    for bucket, times in bucket_timings.items():
        avg = sum(times) / len(times)
        bucket_averages[bucket] = avg
        print(f"  {bucket:<10} {len(times):>4} samples, avg time: {avg:.4f}s", flush=True)

    print(f"Loaded {sum(len(t) for t in bucket_timings.values())} total benchmark samples", flush=True)
    return bucket_averages


def estimate_runtime(matrices, benchmark_file):
    """Estimate total runtime by computing A@B distribution."""
    print(f"\n{'=' * 80}", flush=True)
    print("ESTIMATING TOTAL RUNTIME", flush=True)
    print(f"{'=' * 80}", flush=True)

    # Load benchmark data
    bucket_averages = load_benchmark_timings(benchmark_file)

    # Build extended matrix list
    all_matrices = []
    for (src_type, pred, tgt_type), matrix in matrices.items():
        all_matrices.append((src_type, pred, tgt_type, matrix, 'F'))
        is_symmetric = f'biolink:{pred}' in SYMMETRIC_PREDICATES
        if not is_symmetric:
            all_matrices.append((tgt_type, pred, src_type, matrix.T, 'R'))

    # Group by source type
    by_source_type = defaultdict(list)
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        by_source_type[src_type].append((src_type, pred, tgt_type, matrix, direction))

    # Compute all A@B matrices and count Matrix3 iterations per bucket
    print("\nComputing A@B matrices and categorizing by size...", flush=True)
    print(f"Total Matrix1 to process: {len(all_matrices)}", flush=True)

    size_buckets = Counter()  # Count iterations per size bucket
    total_iterations = 0

    start_time = time.time()

    for idx1, (src_type1, pred1, tgt_type1, matrix1, dir1) in enumerate(all_matrices):
        if idx1 == 0:
            print(f"Processing first Matrix1: {src_type1}|{pred1}|{tgt_type1}", flush=True)
        if idx1 % 100 == 0 and idx1 > 0:
            elapsed = time.time() - start_time
            rate = idx1 / elapsed
            eta_sec = (len(all_matrices) - idx1) / rate
            print(f"  Matrix1: {idx1}/{len(all_matrices)} | Elapsed: {elapsed:.0f}s | ETA: {eta_sec:.0f}s", flush=True)

        if tgt_type1 not in by_source_type:
            continue

        for src_type2, pred2, tgt_type2, matrix2, dir2 in by_source_type[tgt_type1]:
            if matrix1.ncols != matrix2.nrows:
                continue

            # Compute A@B (don't cache - each pair only used once)
            result_AB = matrix1.mxm(matrix2, gb.semiring.any_pair).new()
            ab_edges = result_AB.nvals

            if ab_edges == 0:
                continue

            if tgt_type2 not in by_source_type:
                continue

            # Count how many Matrix3 iterations will use this A@B
            num_matrix3 = len(by_source_type[tgt_type2])

            # Categorize by A@B size
            if ab_edges < 1_000:
                bucket = 'tiny'
            elif ab_edges < 10_000:
                bucket = 'small'
            elif ab_edges < 100_000:
                bucket = 'medium'
            elif ab_edges < 1_000_000:
                bucket = 'large'
            elif ab_edges < 10_000_000:
                bucket = 'xlarge'
            elif ab_edges < 100_000_000:
                bucket = 'xxlarge'
            else:
                bucket = 'huge'

            size_buckets[bucket] += num_matrix3
            total_iterations += num_matrix3

    print(f"\nTotal Matrix3 iterations: {total_iterations:,}", flush=True)
    print(f"\nDistribution by A@B size:", flush=True)

    if bucket_averages:
        print(f"{'Bucket':<15} {'Iterations':<15} {'Percent':<10} {'Avg Time':<15} {'Est Total':<15}")
    else:
        print(f"{'Bucket':<15} {'Iterations':<15} {'Percent':<10}")
    print("-" * 80, flush=True)

    total_estimated_time = 0
    for bucket in ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'huge']:
        count = size_buckets.get(bucket, 0)
        if count == 0:
            continue

        percent = 100 * count / total_iterations

        if bucket_averages and bucket in bucket_averages:
            avg_time = bucket_averages[bucket]
            est_total = count * avg_time
            total_estimated_time += est_total
            print(f"{bucket:<15} {count:<15,} {percent:<10.2f} {avg_time:<15.4f} {est_total:<15,.0f}s", flush=True)
        else:
            print(f"{bucket:<15} {count:<15,} {percent:<10.2f}", flush=True)

    print(f"\n{'=' * 80}", flush=True)
    if bucket_averages:
        print(f"TOTAL ESTIMATED TIME: {total_estimated_time:,.0f} seconds", flush=True)
        print(f"  = {total_estimated_time/60:,.0f} minutes", flush=True)
        print(f"  = {total_estimated_time/3600:,.1f} hours", flush=True)
        print(f"  = {total_estimated_time/86400:,.1f} days", flush=True)
    else:
        print(f"No benchmark data provided.", flush=True)
        print(f"To estimate runtime:", flush=True)
        print(f"  1. Run: generate_benchmark_samples.py to create stratified samples")
        print(f"  2. Run: run_benchmark_samples.py to collect real timing data")
        print(f"  3. Re-run this script with --benchmark <results.tsv>", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Estimate total runtime for 3-hop analysis'
    )
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--benchmark', required=False, help='Benchmark results TSV file (optional)')

    args = parser.parse_args()

    node_types = load_node_types(args.nodes)
    matrices = build_matrices(args.edges, node_types)
    estimate_runtime(matrices, args.benchmark)


if __name__ == "__main__":
    main()
