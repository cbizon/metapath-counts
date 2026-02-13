#!/usr/bin/env python3
"""
Run benchmarks on sampled Matrix3 iterations.

Takes sample list from generate_benchmark_samples.py and runs actual timing.

Output format (TSV):
  bucket | metapath | ab_edges | abc_edges | num_comparisons | ab_time | abc_time | comparison_time | total_time

Usage:
    uv run python scripts/run_benchmark_samples.py \
        --edges ../SimplePredictions/input_graphs/robokop_base_nonredundant/edges.jsonl \
        --nodes ../SimplePredictions/input_graphs/robokop_base_nonredundant/nodes.jsonl \
        --samples benchmark_samples.tsv \
        --output benchmark_results.tsv
"""

import argparse
import json
import time
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


def format_metapath(node_types, predicates, directions):
    """Format metapath as parsable string."""
    parts = []
    for i, node_type in enumerate(node_types):
        parts.append(node_type)
        if i < len(predicates):
            parts.append(predicates[i])
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


def load_samples(samples_file):
    """Load sample list."""
    print(f"\nLoading samples from {samples_file}...", flush=True)
    samples = []

    with open(samples_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 14:
                samples.append({
                    'bucket': parts[0],
                    'src_type1': parts[1], 'pred1': parts[2], 'tgt_type1': parts[3], 'dir1': parts[4],
                    'src_type2': parts[5], 'pred2': parts[6], 'tgt_type2': parts[7], 'dir2': parts[8],
                    'src_type3': parts[9], 'pred3': parts[10], 'tgt_type3': parts[11], 'dir3': parts[12],
                    'ab_edges': int(parts[13])
                })

    print(f"Loaded {len(samples)} samples", flush=True)
    return samples


def run_benchmarks(matrices, samples, output_file):
    """Run benchmarks on sampled iterations."""
    print(f"\n{'=' * 80}", flush=True)
    print("RUNNING BENCHMARKS", flush=True)
    print(f"{'=' * 80}", flush=True)

    # Build extended matrix list with inverses
    all_matrices = []
    matrix_metadata = {}

    for (src_type, pred, tgt_type), matrix in matrices.items():
        # Forward
        all_matrices.append((src_type, pred, tgt_type, matrix, 'F'))
        matrix_metadata[(src_type, pred, tgt_type, 'F')] = (matrix, 'F')

        # Reverse (if not symmetric)
        is_symmetric = f'biolink:{pred}' in SYMMETRIC_PREDICATES
        if not is_symmetric:
            all_matrices.append((tgt_type, pred, src_type, matrix.T, 'R'))
            matrix_metadata[(tgt_type, pred, src_type, 'R')] = (matrix.T, 'R')

    # Create matrix lookup
    matrix_lookup = {}
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        matrix_lookup[(src_type, pred, tgt_type, direction)] = matrix

    # Run benchmarks
    with open(output_file, 'w') as f:
        f.write("bucket\tmetapath\tab_edges\tabc_edges\tnum_comparisons\tab_time\tabc_time\tcomparison_time\ttotal_time\n")

        for idx, sample in enumerate(samples):
            if idx % 10 == 0:
                print(f"  Benchmark {idx}/{len(samples)}", flush=True)

            # Get matrices
            m1_key = (sample['src_type1'], sample['pred1'], sample['tgt_type1'], sample['dir1'])
            m2_key = (sample['src_type2'], sample['pred2'], sample['tgt_type2'], sample['dir2'])
            m3_key = (sample['src_type3'], sample['pred3'], sample['tgt_type3'], sample['dir3'])

            if m1_key not in matrix_lookup or m2_key not in matrix_lookup or m3_key not in matrix_lookup:
                print(f"  WARNING: Matrix not found for sample {idx}, skipping", flush=True)
                continue

            matrix1 = matrix_lookup[m1_key]
            matrix2 = matrix_lookup[m2_key]
            matrix3 = matrix_lookup[m3_key]

            # Check dimensions
            if matrix1.ncols != matrix2.nrows:
                print(f"  WARNING: Dimension mismatch M1/M2 for sample {idx}, skipping", flush=True)
                continue

            # Time A@B
            ab_start = time.time()
            result_AB = matrix1.mxm(matrix2, gb.semiring.any_pair).new()
            ab_time = time.time() - ab_start
            ab_edges = result_AB.nvals

            if ab_edges == 0:
                continue

            if result_AB.ncols != matrix3.nrows:
                print(f"  WARNING: Dimension mismatch AB/M3 for sample {idx}, skipping", flush=True)
                continue

            # Time (A@B)@C
            abc_start = time.time()
            result_ABC = result_AB.mxm(matrix3, gb.semiring.any_pair).new()
            abc_time = time.time() - abc_start
            abc_edges = result_ABC.nvals

            if abc_edges == 0:
                continue

            # Time comparisons with 1-hop matrices
            comparison_start = time.time()
            num_comparisons = 0
            for (onehop_src, onehop_pred, onehop_tgt, onehop_dir), onehop_matrix in matrix_metadata.items():
                # Check dimension compatibility
                if onehop_src != sample['src_type1'] or onehop_tgt != sample['tgt_type3']:
                    continue

                if result_ABC.nrows != onehop_matrix[0].nrows or result_ABC.ncols != onehop_matrix[0].ncols:
                    continue

                # Calculate overlap
                overlap_matrix = result_ABC.ewise_mult(onehop_matrix[0], gb.binary.pair).new()
                num_comparisons += 1

            comparison_time = time.time() - comparison_start
            total_time = ab_time + abc_time + comparison_time

            # Format metapath
            metapath = format_metapath(
                [sample['src_type1'], sample['tgt_type1'], sample['tgt_type2'], sample['tgt_type3']],
                [sample['pred1'], sample['pred2'], sample['pred3']],
                [sample['dir1'], sample['dir2'], sample['dir3']]
            )

            # Write result
            f.write(f"{sample['bucket']}\t{metapath}\t{ab_edges}\t{abc_edges}\t{num_comparisons}\t"
                   f"{ab_time:.6f}\t{abc_time:.6f}\t{comparison_time:.6f}\t{total_time:.6f}\n")

            if (idx + 1) % 100 == 0:
                f.flush()

    print(f"\nDone! Wrote benchmarks to {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks on sampled iterations'
    )
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--samples', required=True, help='Sample TSV file from generate_benchmark_samples.py')
    parser.add_argument('--output', required=True, help='Output TSV file path')

    args = parser.parse_args()

    node_types = load_node_types(args.nodes)
    matrices = build_matrices(args.edges, node_types)
    samples = load_samples(args.samples)
    run_benchmarks(matrices, samples, args.output)


if __name__ == "__main__":
    main()
