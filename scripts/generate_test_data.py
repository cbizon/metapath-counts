#!/usr/bin/env python3
"""Generate small test dataset from production KGX files.

Samples nodes with diverse type hierarchies and their connecting edges
to create a small, representative test graph.

Usage:
    uv run python scripts/generate_test_data.py \\
        --edges /path/to/edges.jsonl \\
        --nodes /path/to/nodes.jsonl \\
        --output-dir tests/fixtures \\
        --num-nodes 1000 \\
        --num-edges 5000
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def analyze_type_distribution(nodes_file: str) -> Dict[int, List[str]]:
    """Analyze nodes by number of types they have.

    Returns:
        Dict mapping type_count -> list of node IDs
    """
    print(f"Analyzing type distribution in {nodes_file}...", flush=True)

    nodes_by_type_count = defaultdict(list)
    total_nodes = 0

    with open(nodes_file, 'r') as f:
        for line in f:
            node = json.loads(line)
            categories = node.get('category', [])
            type_count = len(categories)
            nodes_by_type_count[type_count].append(node)
            total_nodes += 1

            if total_nodes % 1_000_000 == 0:
                print(f"  Processed {total_nodes:,} nodes", flush=True)

    print(f"\nType distribution:", flush=True)
    for count in sorted(nodes_by_type_count.keys()):
        num_nodes = len(nodes_by_type_count[count])
        print(f"  {count} types: {num_nodes:,} nodes ({100*num_nodes/total_nodes:.1f}%)", flush=True)

    return nodes_by_type_count


def sample_diverse_nodes(
    nodes_by_type_count: Dict[int, List[dict]],
    num_nodes: int,
    excluded_types: Set[str]
) -> Tuple[List[dict], Dict[str, int]]:
    """Sample nodes with diverse type hierarchies.

    Ensures:
    - Nodes with 1, 2, 3+ types are represented
    - Excluded types are included for testing filtering
    - Even distribution across type counts

    Returns:
        Tuple of (selected_nodes, type_frequency_counts)
    """
    print(f"\nSampling {num_nodes} nodes with diverse hierarchies...", flush=True)

    selected_nodes = []
    type_counter = Counter()

    # Calculate how many nodes to sample from each type count bucket
    type_counts = sorted(nodes_by_type_count.keys())
    nodes_per_bucket = num_nodes // len(type_counts)

    for type_count in type_counts:
        available = nodes_by_type_count[type_count]
        sample_size = min(nodes_per_bucket, len(available))
        sampled = random.sample(available, sample_size)
        selected_nodes.extend(sampled)

        # Count type frequencies
        for node in sampled:
            for cat in node.get('category', []):
                type_name = cat.replace('biolink:', '')
                type_counter[type_name] += 1

        print(f"  Sampled {sample_size} nodes with {type_count} types", flush=True)

    # Add more nodes to reach target if needed
    remaining = num_nodes - len(selected_nodes)
    if remaining > 0:
        # Sample from all remaining nodes
        all_remaining = []
        for nodes_list in nodes_by_type_count.values():
            all_remaining.extend(nodes_list)

        # Remove already selected
        selected_ids = {n['id'] for n in selected_nodes}
        available = [n for n in all_remaining if n['id'] not in selected_ids]

        if len(available) >= remaining:
            additional = random.sample(available, remaining)
            selected_nodes.extend(additional)
            for node in additional:
                for cat in node.get('category', []):
                    type_name = cat.replace('biolink:', '')
                    type_counter[type_name] += 1

    print(f"\nSelected {len(selected_nodes)} nodes total", flush=True)

    # Check for excluded types
    excluded_found = {t for t in excluded_types if type_counter[t] > 0}
    print(f"\nExcluded types found in sample: {len(excluded_found)}/{len(excluded_types)}", flush=True)
    for t in sorted(excluded_found):
        print(f"  {t}: {type_counter[t]} nodes", flush=True)

    return selected_nodes, dict(type_counter)


def sample_connecting_edges(
    edges_file: str,
    selected_node_ids: Set[str],
    num_edges: int
) -> List[dict]:
    """Sample edges that connect the selected nodes.

    Returns:
        List of edge dictionaries
    """
    print(f"\nSampling edges connecting selected nodes...", flush=True)

    connecting_edges = []
    total_edges = 0

    with open(edges_file, 'r') as f:
        for line in f:
            edge = json.loads(line)
            subject = edge.get('subject')
            obj = edge.get('object')

            # Keep edges where both nodes are in our sample
            if subject in selected_node_ids and obj in selected_node_ids:
                connecting_edges.append(edge)

            total_edges += 1
            if total_edges % 1_000_000 == 0:
                print(f"  Processed {total_edges:,} edges, found {len(connecting_edges):,} connecting", flush=True)

    print(f"\nFound {len(connecting_edges):,} edges connecting selected nodes", flush=True)

    # If we have more than requested, sample down
    if len(connecting_edges) > num_edges:
        connecting_edges = random.sample(connecting_edges, num_edges)
        print(f"Sampled down to {len(connecting_edges):,} edges", flush=True)
    elif len(connecting_edges) < num_edges:
        print(f"Warning: Only found {len(connecting_edges):,} connecting edges (requested {num_edges:,})", flush=True)

    # Count edge predicates
    predicate_counts = Counter(e.get('predicate', '') for e in connecting_edges)
    print(f"\nPredicate distribution:", flush=True)
    for pred, count in predicate_counts.most_common(10):
        print(f"  {pred}: {count}", flush=True)

    return connecting_edges


def write_output(nodes: List[dict], edges: List[dict], output_dir: Path):
    """Write nodes and edges to output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_file = output_dir / "test_nodes.jsonl"
    edges_file = output_dir / "test_edges.jsonl"

    print(f"\nWriting output files...", flush=True)
    print(f"  Nodes: {nodes_file}", flush=True)
    print(f"  Edges: {edges_file}", flush=True)

    with open(nodes_file, 'w') as f:
        for node in nodes:
            f.write(json.dumps(node) + '\n')

    with open(edges_file, 'w') as f:
        for edge in edges:
            f.write(json.dumps(edge) + '\n')

    print(f"\n✓ Generated test dataset:", flush=True)
    print(f"  {len(nodes):,} nodes -> {nodes_file}", flush=True)
    print(f"  {len(edges):,} edges -> {edges_file}", flush=True)


def write_statistics(type_frequencies: Dict[str, int], output_dir: Path):
    """Write type distribution statistics."""
    stats_file = output_dir / "test_data_stats.json"

    stats = {
        "type_frequencies": type_frequencies,
        "total_types": len(type_frequencies),
        "most_common_types": sorted(
            type_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Statistics written to {stats_file}", flush=True)
    print(f"\nTop 10 most common types:", flush=True)
    for type_name, count in stats["most_common_types"][:10]:
        print(f"  {type_name}: {count}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test dataset from production KGX files"
    )
    parser.add_argument(
        '--edges',
        required=True,
        help='Path to production edges.jsonl file'
    )
    parser.add_argument(
        '--nodes',
        required=True,
        help='Path to production nodes.jsonl file'
    )
    parser.add_argument(
        '--output-dir',
        default='tests/fixtures',
        help='Output directory for test files (default: tests/fixtures)'
    )
    parser.add_argument(
        '--num-nodes',
        type=int,
        default=1000,
        help='Target number of nodes to sample (default: 1000)'
    )
    parser.add_argument(
        '--num-edges',
        type=int,
        default=5000,
        help='Target number of edges to sample (default: 5000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Excluded types from user decision
    excluded_types = {
        'ThingWithTaxon',
        'SubjectOfInvestigation',
        'PhysicalEssenceOrOccurrent',
        'PhysicalEssence',
        'OntologyClass',
        'Occurrent',
        'InformationContentEntity',
        'Attribute'
    }

    print("="*60)
    print("TEST DATA GENERATOR")
    print("="*60)
    print(f"Input nodes: {args.nodes}")
    print(f"Input edges: {args.edges}")
    print(f"Output dir: {args.output_dir}")
    print(f"Target: {args.num_nodes} nodes, {args.num_edges} edges")
    print(f"Random seed: {args.seed}")
    print()

    # Step 1: Analyze type distribution
    nodes_by_type_count = analyze_type_distribution(args.nodes)

    # Step 2: Sample diverse nodes
    selected_nodes, type_frequencies = sample_diverse_nodes(
        nodes_by_type_count,
        args.num_nodes,
        excluded_types
    )

    # Step 3: Sample connecting edges
    selected_node_ids = {node['id'] for node in selected_nodes}
    selected_edges = sample_connecting_edges(
        args.edges,
        selected_node_ids,
        args.num_edges
    )

    # Step 4: Write output
    output_dir = Path(args.output_dir)
    write_output(selected_nodes, selected_edges, output_dir)
    write_statistics(type_frequencies, output_dir)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
