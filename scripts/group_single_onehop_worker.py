#!/usr/bin/env python3
"""
Worker script to group results for all 1-hop metapaths between a type pair.

Streams through specified result files, finds all 1-hop metapaths between
the given type pair, aggregates counts for each, and computes performance metrics.

Uses precomputed aggregated path counts (from prepare_grouping.py) to get
global counts for aggregated paths like Entity|related_to|F|Entity.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import zstandard

# Import hierarchy and aggregation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from metapath_counts.hierarchy import get_type_ancestors, get_predicate_ancestors
from metapath_counts.type_assignment import is_pseudo_type, parse_pseudo_type
from metapath_counts.aggregation import expand_metapath_to_variants, calculate_metrics


# Global caches for precomputed data
_aggregated_counts_cache = None
_aggregated_nhop_counts_cache = None
_type_node_counts_cache = None


def load_aggregated_counts(counts_path):
    """Load precomputed aggregated path counts.

    Args:
        counts_path: Path to aggregated_path_counts.json

    Returns:
        Dict mapping path string to global count
    """
    global _aggregated_counts_cache

    if _aggregated_counts_cache is not None:
        return _aggregated_counts_cache

    print(f"Loading precomputed aggregated counts from {counts_path}...")
    with open(counts_path, 'r') as f:
        data = json.load(f)

    _aggregated_counts_cache = data.get("counts", {})
    print(f"  Loaded {len(_aggregated_counts_cache)} aggregated path counts")

    return _aggregated_counts_cache


def load_aggregated_nhop_counts(counts_path):
    """Load precomputed aggregated N-hop path counts.

    Args:
        counts_path: Path to aggregated_nhop_counts.json

    Returns:
        Dict mapping N-hop path variant to global count
    """
    global _aggregated_nhop_counts_cache

    if _aggregated_nhop_counts_cache is not None:
        return _aggregated_nhop_counts_cache

    print(f"Loading precomputed aggregated N-hop counts from {counts_path}...")
    with open(counts_path, 'r') as f:
        data = json.load(f)

    _aggregated_nhop_counts_cache = data.get("counts", {})
    print(f"  Loaded {len(_aggregated_nhop_counts_cache)} aggregated N-hop path counts")

    return _aggregated_nhop_counts_cache


def load_type_node_counts(counts_path):
    """Load precomputed type node counts.

    Args:
        counts_path: Path to type_node_counts.json

    Returns:
        Dict mapping type name to node count
    """
    global _type_node_counts_cache

    if _type_node_counts_cache is not None:
        return _type_node_counts_cache

    print(f"Loading precomputed type node counts from {counts_path}...")
    with open(counts_path, 'r') as f:
        _type_node_counts_cache = json.load(f)

    print(f"  Loaded {len(_type_node_counts_cache)} type node counts")

    return _type_node_counts_cache


def compute_total_possible(type1, type2, type_node_counts):
    """Compute total possible pairs for a type pair.

    Args:
        type1: First type name
        type2: Second type name
        type_node_counts: Dict mapping type name to node count

    Returns:
        Total possible pairs (|type1 nodes| * |type2 nodes|)
    """
    count1 = type_node_counts.get(type1, 0)
    count2 = type_node_counts.get(type2, 0)
    return count1 * count2


def check_type_match(onehop_path, type1, type2):
    """Check if a 1-hop metapath aggregates to a type pair.

    This checks if the 1-hop path's types are descendants of the target type pair.
    For example, SmallMolecule|treats|F|Disease matches (ChemicalEntity, DiseaseOrPhenotypicFeature)
    because SmallMolecule is a descendant of ChemicalEntity and Disease is a descendant of
    DiseaseOrPhenotypicFeature.

    Args:
        onehop_path: Pipe-separated 1-hop metapath like "Gene|affects|F|Disease"
        type1: First type (may be hierarchical like ChemicalEntity)
        type2: Second type (may be hierarchical like DiseaseOrPhenotypicFeature)

    Returns:
        True if the metapath aggregates to (type1, type2) in either direction
    """
    parts = onehop_path.split('|')
    if len(parts) != 4:
        return False

    src_type, pred, direction, tgt_type = parts

    # Get ancestors of each type (includes self)
    src_ancestors = get_type_ancestors(src_type)
    tgt_ancestors = get_type_ancestors(tgt_type)

    # Check if type pair is reachable via hierarchy
    # Forward: src_type -> type1, tgt_type -> type2
    match_forward = (type1 in src_ancestors and type2 in tgt_ancestors)
    # Reverse: src_type -> type2, tgt_type -> type1
    match_reverse = (type2 in src_ancestors and type1 in tgt_ancestors)

    return match_forward or match_reverse


def contains_pseudo_type(metapath):
    """Check if a metapath contains any pseudo-type nodes.

    Pseudo-types contain '+' (e.g., Gene+Protein).
    We filter these out of final output since their counts are
    already included in the constituent type aggregations.
    """
    parts = metapath.split('|')
    # Check node positions: 0, 3, 6, ...
    for i in range(0, len(parts), 3):
        if '+' in parts[i]:
            return True
    return False


def should_exclude_metapath(metapath, excluded_types, excluded_predicates):
    """Check if a metapath should be excluded based on types or predicates."""
    if not excluded_types and not excluded_predicates:
        return False

    parts = metapath.split('|')
    # Extract nodes and predicates from metapath
    # Format: Node|pred|dir|Node|pred|dir|...
    nodes = []
    predicates = []
    for i, part in enumerate(parts):
        if i % 3 == 0:  # Node positions: 0, 3, 6, ...
            nodes.append(part)
        elif i % 3 == 1:  # Predicate positions: 1, 4, 7, ...
            predicates.append(part)

    if excluded_types:
        for node in nodes:
            if node in excluded_types:
                return True

    if excluded_predicates:
        for pred in predicates:
            if pred in excluded_predicates:
                return True

    return False


def group_type_pair(type1, type2, file_list, output_dir, n_hops, aggregate=True,
                    aggregated_counts=None, aggregated_nhop_counts=None, type_node_counts=None,
                    min_count=0, min_precision=0.0, excluded_types=None, excluded_predicates=None):
    """Group all N-hop results for 1-hop metapaths between a type pair.

    Args:
        type1: First type (e.g., "Gene")
        type2: Second type (e.g., "Disease")
        file_list: List of result file paths to scan
        output_dir: Output directory for grouped results
        n_hops: Number of hops
        aggregate: Whether to do hierarchical aggregation (default: True)
        aggregated_counts: Dict mapping aggregated 1-hop path -> global count (for target counts)
        aggregated_nhop_counts: Dict mapping aggregated N-hop path -> global count (for predictor counts)
        type_node_counts: Dict mapping type name -> node count (for total_possible calculation)
        min_count: Minimum N-hop count to include (default: 0)
        min_precision: Minimum precision to include (default: 0.0)
        excluded_types: Set of node types to exclude
        excluded_predicates: Set of predicates to exclude
    """
    excluded_types = excluded_types or set()
    excluded_predicates = excluded_predicates or set()

    print(f"Grouping for type pair: ({type1}, {type2})")
    print(f"Aggregation: {'enabled' if aggregate else 'disabled'}")
    if min_count > 0:
        print(f"Min count filter: {min_count}")
    if min_precision > 0:
        print(f"Min precision filter: {min_precision}")
    if excluded_types:
        print(f"Excluded types: {sorted(excluded_types)}")
    if excluded_predicates:
        print(f"Excluded predicates: {sorted(excluded_predicates)}")
    if aggregated_counts:
        print(f"Using precomputed 1-hop counts: {len(aggregated_counts)} paths")
    if aggregated_nhop_counts:
        print(f"Using precomputed N-hop counts: {len(aggregated_nhop_counts)} paths")
    if type_node_counts:
        print(f"Using precomputed type node counts: {len(type_node_counts)} types")

    # Compute total_possible for this type pair from node counts
    total_possible_for_pair = compute_total_possible(type1, type2, type_node_counts) if type_node_counts else 0
    print(f"Total possible for ({type1}, {type2}): {total_possible_for_pair:,}")

    print(f"Scanning {len(file_list)} result files (optimized subset)...")

    # Data structure: onehop_path -> {nhop_path -> [overlap, nhop_count]}
    # Note: total_possible is computed per type pair, not summed from rows
    # We track BOTH overlap and nhop_count because nhop_count must be summed during aggregation
    onehop_to_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    files_processed = 0
    rows_found = 0

    # Stream through all files to find matching 1-hops
    for file_path in file_list:
        files_processed += 1

        with open(file_path, 'r') as f:
            # Skip header
            f.readline()

            # Find matching rows
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 6:
                    continue

                nhop_path = parts[0]
                nhop_count = int(parts[1])  # Read from results - must be aggregated, not looked up
                onehop_path = parts[2]
                # onehop_count = int(parts[3])  # Don't use - will look up from aggregated_counts
                overlap = int(parts[4])
                # total_possible from file is ignored - we compute it from type node counts

                # Check if this 1-hop involves our type pair (hierarchically)
                if check_type_match(onehop_path, type1, type2):
                    rows_found += 1

                    # Expand 1-hop path to all hierarchical variants
                    onehop_variants = expand_metapath_to_variants(onehop_path)

                    # Store overlap AND nhop_count for each variant that matches the type pair
                    for onehop_variant in onehop_variants:
                        # Only keep variants that match this job's type pair
                        variant_parts = onehop_variant.split('|')
                        if len(variant_parts) == 4:
                            v_src, v_pred, v_dir, v_tgt = variant_parts
                            # Check exact match (not hierarchical - variants are already expanded)
                            if (v_src == type1 and v_tgt == type2) or (v_src == type2 and v_tgt == type1):
                                data = onehop_to_data[onehop_variant][nhop_path]
                                data[0] += overlap
                                data[1] += nhop_count

        if files_processed % 500 == 0:
            print(f"  Processed {files_processed}/{len(file_list)} files, found {rows_found} matching rows")

    print(f"\n✓ Found {rows_found} total rows")
    print(f"  Unique 1-hop metapaths: {len(onehop_to_data)}")

    if len(onehop_to_data) == 0:
        print("\nWARNING: No matching 1-hop metapaths found. Skipping output.")
        return

    # Process each 1-hop metapath
    for onehop_path, nhop_data in onehop_to_data.items():
        print(f"\nProcessing 1-hop: {onehop_path}")
        print(f"  {len(nhop_data)} unique N-hop paths")

        # Get onehop_count from precomputed lookup
        if aggregated_counts and onehop_path in aggregated_counts:
            onehop_count_global = aggregated_counts[onehop_path]
        else:
            # Fallback: this shouldn't happen if prepare_grouping.py was run
            print(f"  WARNING: No precomputed count for {onehop_path}, using 0")
            onehop_count_global = 0

        # Apply hierarchical aggregation if enabled
        # Data structure: nhop_variant -> overlap
        # NOTE: nhop_count is looked up from aggregated_nhop_counts, NOT accumulated here
        # This ensures we get the correct global count for each variant
        aggregated_overlaps = {}
        if aggregate:
            print(f"  Applying hierarchical aggregation...")

            # For each nhop_path, expand and aggregate overlap only
            final_aggregated = defaultdict(int)

            for nhop_path, (overlap, _nhop_count) in nhop_data.items():
                # Expand nhop_path to all hierarchical variants
                nhop_variants = expand_metapath_to_variants(nhop_path)

                # Add overlap to all variants
                # nhop_count will be looked up from aggregated_nhop_counts when writing
                for variant in nhop_variants:
                    final_aggregated[variant] += overlap

            aggregated_overlaps = dict(final_aggregated)
            print(f"    N-hop paths expanded to {len(aggregated_overlaps)} aggregated paths")
        else:
            # No aggregation - extract just the overlap
            aggregated_overlaps = {k: v[0] for k, v in nhop_data.items()}

        # Check if 1-hop should be excluded
        if should_exclude_metapath(onehop_path, excluded_types, excluded_predicates):
            print(f"  Skipping (excluded type/predicate in 1-hop)")
            continue

        # Skip 1-hop paths containing pseudo-types (their counts are in constituent types)
        if contains_pseudo_type(onehop_path):
            print(f"  Skipping (pseudo-type in 1-hop)")
            continue

        # Write output file for this 1-hop (zstd compressed)
        safe_filename = onehop_path.replace('|', '_').replace(':', '_').replace(' ', '_')
        output_file = f"{output_dir}/{safe_filename}.tsv.zst"

        # Filter and count rows
        rows_written = 0
        rows_filtered_excluded = 0
        rows_filtered_pseudo = 0
        rows_filtered_count = 0
        rows_filtered_precision = 0
        rows_missing_nhop_count = 0

        with zstandard.open(output_file, 'wt') as out:
            # Header
            out.write("predictor_metapath\tpredictor_count\toverlap\ttotal_possible\t")
            out.write("precision\trecall\tf1\tmcc\tspecificity\tnpv\n")

            # Sort by overlap descending
            sorted_items = sorted(aggregated_overlaps.items(), key=lambda x: x[1], reverse=True)

            for nhop_path, overlap in sorted_items:
                # Filter by excluded types/predicates
                if should_exclude_metapath(nhop_path, excluded_types, excluded_predicates):
                    rows_filtered_excluded += 1
                    continue

                # Filter out pseudo-types (their counts are in constituent types)
                if contains_pseudo_type(nhop_path):
                    rows_filtered_pseudo += 1
                    continue

                # Look up nhop_count from precomputed aggregated N-hop counts
                # This gives the correct global count for this variant
                if aggregated_nhop_counts and nhop_path in aggregated_nhop_counts:
                    nhop_count = aggregated_nhop_counts[nhop_path]
                else:
                    # Path not in precomputed counts - skip
                    rows_missing_nhop_count += 1
                    continue

                # Filter by minimum count
                if nhop_count < min_count:
                    rows_filtered_count += 1
                    continue

                # Calculate metrics using type-pair total_possible
                metrics = calculate_metrics(nhop_count, onehop_count_global, overlap, total_possible_for_pair)

                # Filter by minimum precision
                if metrics['precision'] < min_precision:
                    rows_filtered_precision += 1
                    continue

                # Write row
                out.write(f"{nhop_path}\t{nhop_count}\t{overlap}\t{total_possible_for_pair}\t")
                out.write(f"{metrics['precision']:.6f}\t{metrics['recall']:.6f}\t")
                out.write(f"{metrics['f1']:.6f}\t{metrics['mcc']:.6f}\t")
                out.write(f"{metrics['specificity']:.6f}\t{metrics['npv']:.6f}\n")
                rows_written += 1

        print(f"  Written {rows_written} rows to {output_file}")
        if rows_filtered_excluded > 0:
            print(f"    Filtered (excluded): {rows_filtered_excluded}")
        if rows_filtered_pseudo > 0:
            print(f"    Filtered (pseudo-type): {rows_filtered_pseudo}")
        if rows_filtered_count > 0:
            print(f"    Filtered (count < {min_count}): {rows_filtered_count}")
        if rows_filtered_precision > 0:
            print(f"    Filtered (precision < {min_precision}): {rows_filtered_precision}")
        if rows_missing_nhop_count > 0:
            print(f"    WARNING: Skipped {rows_missing_nhop_count} paths missing from aggregated_nhop_counts")

    print(f"\n✓ All 1-hop metapaths processed for type pair ({type1}, {type2})")


def main():
    parser = argparse.ArgumentParser(
        description='Group results for all 1-hop metapaths between a type pair'
    )
    parser.add_argument(
        '--type1',
        type=str,
        required=True,
        help='First type (e.g., "Gene")'
    )
    parser.add_argument(
        '--type2',
        type=str,
        required=True,
        help='Second type (e.g., "Disease")'
    )
    parser.add_argument(
        '--file-list',
        type=str,
        required=True,
        help='Path to file containing list of result files to scan'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for grouped results'
    )
    parser.add_argument(
        '--n-hops',
        type=int,
        required=True,
        help='Number of hops'
    )
    parser.add_argument(
        '--explicit-only',
        action='store_true',
        help='Disable hierarchical aggregation (explicit results only)'
    )
    parser.add_argument(
        '--aggregated-counts',
        type=str,
        required=True,
        help='Path to aggregated_path_counts.json (1-hop counts from prepare_grouping.py)'
    )
    parser.add_argument(
        '--aggregated-nhop-counts',
        type=str,
        required=True,
        help='Path to aggregated_nhop_counts.json (N-hop counts from prepare_grouping.py)'
    )
    parser.add_argument(
        '--type-node-counts',
        type=str,
        required=True,
        help='Path to type_node_counts.json (from prepare_grouping.py)'
    )
    parser.add_argument(
        '--min-count',
        type=int,
        default=0,
        help='Minimum N-hop count to include in output (default: 0)'
    )
    parser.add_argument(
        '--min-precision',
        type=float,
        default=0.0,
        help='Minimum precision to include in output (default: 0)'
    )
    parser.add_argument(
        '--exclude-types',
        type=str,
        default='Entity,ThingWithTaxon',
        help='Comma-separated list of node types to exclude (default: Entity,ThingWithTaxon)'
    )
    parser.add_argument(
        '--exclude-predicates',
        type=str,
        default='related_to_at_instance_level,related_to_at_concept_level',
        help='Comma-separated list of predicates to exclude (default: related_to_at_instance_level,related_to_at_concept_level)'
    )

    args = parser.parse_args()

    # Build exclusion sets
    excluded_types = set(t.strip() for t in args.exclude_types.split(',') if t.strip())
    excluded_predicates = set(p.strip() for p in args.exclude_predicates.split(',') if p.strip())

    # Read file list from file
    with open(args.file_list, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    # Load precomputed data
    aggregated_counts = load_aggregated_counts(args.aggregated_counts)
    aggregated_nhop_counts = load_aggregated_nhop_counts(args.aggregated_nhop_counts)
    type_node_counts = load_type_node_counts(args.type_node_counts)

    group_type_pair(
        type1=args.type1,
        type2=args.type2,
        file_list=file_list,
        output_dir=args.output_dir,
        n_hops=args.n_hops,
        aggregate=not args.explicit_only,
        aggregated_counts=aggregated_counts,
        aggregated_nhop_counts=aggregated_nhop_counts,
        type_node_counts=type_node_counts,
        min_count=args.min_count,
        min_precision=args.min_precision,
        excluded_types=excluded_types,
        excluded_predicates=excluded_predicates
    )


if __name__ == "__main__":
    main()
