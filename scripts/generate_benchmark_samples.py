#!/usr/bin/env python3
"""
Generate stratified sample of Matrix3 iterations for benchmarking.

Strategy:
1. For each valid Matrix1/Matrix2/Matrix3 combo, compute A@B size
2. Categorize into buckets
3. Sample from each bucket proportionally (with minimums)
4. Output sample list with metadata

Output format (TSV):
  bucket | src_type1 | pred1 | tgt_type1 | dir1 | src_type2 | pred2 | tgt_type2 | dir2 | src_type3 | pred3 | tgt_type3 | dir3 | ab_edges

Usage:
    uv run python scripts/metapaths/generate_benchmark_samples.py \
        --edges ../SimplePredictions/input_graphs/robokop_base_nonredundant/edges.jsonl \
        --nodes ../SimplePredictions/input_graphs/robokop_base_nonredundant/nodes.jsonl \
        --output benchmark_samples.tsv \
        --total-samples 1000
"""

import argparse
import json
import random
from collections import defaultdict
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


def categorize_bucket(ab_edges):
    """Categorize by A@B size."""
    if ab_edges < 1_000:
        return 'tiny'
    elif ab_edges < 10_000:
        return 'small'
    elif ab_edges < 100_000:
        return 'medium'
    elif ab_edges < 1_000_000:
        return 'large'
    elif ab_edges < 10_000_000:
        return 'xlarge'
    elif ab_edges < 100_000_000:
        return 'xxlarge'
    else:
        return 'huge'


def generate_samples(matrices, total_samples, output_file):
    """Generate stratified sample of iterations."""
    print(f"\n{'=' * 80}", flush=True)
    print("GENERATING STRATIFIED SAMPLES", flush=True)
    print(f"{'=' * 80}", flush=True)

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

    # Collect all iterations by bucket
    print(f"\nCategorizing all iterations...", flush=True)
    bucket_iterations = defaultdict(list)
    total_iterations = 0

    for idx1, (src_type1, pred1, tgt_type1, matrix1, dir1) in enumerate(all_matrices):
        if idx1 % 100 == 0 and idx1 > 0:
            print(f"  Processed {idx1}/{len(all_matrices)} Matrix1 combinations", flush=True)

        if tgt_type1 not in by_source_type:
            continue

        for src_type2, pred2, tgt_type2, matrix2, dir2 in by_source_type[tgt_type1]:
            if matrix1.ncols != matrix2.nrows:
                continue

            # Compute A@B
            result_AB = matrix1.mxm(matrix2, gb.semiring.any_pair).new()
            ab_edges = result_AB.nvals

            if ab_edges == 0:
                continue

            if tgt_type2 not in by_source_type:
                continue

            # Store metadata for each Matrix3 iteration
            for src_type3, pred3, tgt_type3, matrix3, dir3 in by_source_type[tgt_type2]:
                if result_AB.ncols != matrix3.nrows:
                    continue

                bucket = categorize_bucket(ab_edges)
                iteration_data = {
                    'src_type1': src_type1, 'pred1': pred1, 'tgt_type1': tgt_type1, 'dir1': dir1,
                    'src_type2': src_type2, 'pred2': pred2, 'tgt_type2': tgt_type2, 'dir2': dir2,
                    'src_type3': src_type3, 'pred3': pred3, 'tgt_type3': tgt_type3, 'dir3': dir3,
                    'ab_edges': ab_edges
                }
                bucket_iterations[bucket].append(iteration_data)
                total_iterations += 1

    print(f"\nTotal iterations: {total_iterations:,}", flush=True)
    print(f"\nDistribution:", flush=True)
    for bucket in ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'huge']:
        count = len(bucket_iterations[bucket])
        if count > 0:
            percent = 100 * count / total_iterations
            print(f"  {bucket:<10} {count:>12,} ({percent:>5.2f}%)", flush=True)

    # Stratified sampling with minimums
    print(f"\nSampling {total_samples} iterations (stratified)...", flush=True)

    # Minimum samples per bucket (if bucket exists)
    min_samples = {
        'tiny': 400,
        'small': 200,
        'medium': 150,
        'large': 100,
        'xlarge': 75,
        'xxlarge': 50,
        'huge': 25,
    }

    samples = []
    remaining_budget = total_samples

    # First: allocate minimums
    for bucket in ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'huge']:
        available = len(bucket_iterations[bucket])
        if available == 0:
            continue

        allocated = min(min_samples[bucket], available, remaining_budget)
        samples.extend([(bucket, item) for item in random.sample(bucket_iterations[bucket], allocated)])
        remaining_budget -= allocated

    # Second: distribute remaining budget proportionally
    if remaining_budget > 0:
        for bucket in ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'huge']:
            available = len(bucket_iterations[bucket])
            if available == 0:
                continue

            proportion = available / total_iterations
            extra = int(remaining_budget * proportion)
            if extra > 0:
                # Don't re-sample already selected ones
                already_sampled = len([s for s in samples if s[0] == bucket])
                can_add = min(extra, available - already_sampled)
                if can_add > 0:
                    samples.extend([(bucket, item) for item in random.sample(bucket_iterations[bucket], can_add)])

    print(f"\nSampled {len(samples)} iterations:", flush=True)
    sample_counts = defaultdict(int)
    for bucket, _ in samples:
        sample_counts[bucket] += 1
    for bucket in ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'huge']:
        count = sample_counts[bucket]
        if count > 0:
            print(f"  {bucket:<10} {count:>4} samples", flush=True)

    # Write output
    print(f"\nWriting samples to {output_file}...", flush=True)
    with open(output_file, 'w') as f:
        f.write("bucket\tsrc_type1\tpred1\ttgt_type1\tdir1\tsrc_type2\tpred2\ttgt_type2\tdir2\tsrc_type3\tpred3\ttgt_type3\tdir3\tab_edges\n")
        for bucket, item in samples:
            f.write(f"{bucket}\t{item['src_type1']}\t{item['pred1']}\t{item['tgt_type1']}\t{item['dir1']}\t"
                   f"{item['src_type2']}\t{item['pred2']}\t{item['tgt_type2']}\t{item['dir2']}\t"
                   f"{item['src_type3']}\t{item['pred3']}\t{item['tgt_type3']}\t{item['dir3']}\t"
                   f"{item['ab_edges']}\n")

    print(f"Done! Wrote {len(samples)} samples to {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate stratified sample for benchmarking'
    )
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--output', required=True, help='Output TSV file path')
    parser.add_argument('--total-samples', type=int, default=1000, help='Total samples to generate (default: 1000)')

    args = parser.parse_args()

    node_types = load_node_types(args.nodes)
    matrices = build_matrices(args.edges, node_types)
    generate_samples(matrices, args.total_samples, args.output)


if __name__ == "__main__":
    main()
