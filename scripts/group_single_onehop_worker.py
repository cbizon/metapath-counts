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

# Import hierarchy and aggregation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from metapath_counts.hierarchy import get_type_ancestors, get_predicate_ancestors
from metapath_counts.type_assignment import is_pseudo_type, parse_pseudo_type


# Global caches for precomputed data
_aggregated_counts_cache = None
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


def calculate_metrics(overlap, onehop_count, nhop_count, total_possible):
    """Calculate performance metrics for a prediction.

    Args:
        overlap: Number of overlapping edges
        onehop_count: Total edges in 1-hop path
        onehop_count: Total edges in N-hop path
        total_possible: Total possible pairs (nodes_start * nodes_end)

    Returns:
        dict with precision, recall, f1, mcc, specificity, npv
    """
    # True positives: overlap
    tp = overlap

    # False positives: predicted by N-hop but not in 1-hop
    fp = nhop_count - overlap

    # False negatives: in 1-hop but not predicted by N-hop
    fn = onehop_count - overlap

    # True negatives: not in either
    # Note: tn can be negative when aggregated counts exceed total_possible
    tn = total_possible - (tp + fp + fn)

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (Sensitivity): TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Specificity: TN / (TN + FP) - undefined if tn < 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 and tn >= 0 else 0.0

    # NPV (Negative Predictive Value): TN / (TN + FN) - undefined if tn < 0
    npv = tn / (tn + fn) if (tn + fn) > 0 and tn >= 0 else 0.0

    # Matthews Correlation Coefficient (MCC)
    # MCC is undefined when tn < 0 (denominator would involve sqrt of negative)
    numerator = (tp * tn) - (fp * fn)
    denom_product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_product > 0:
        denominator = denom_product ** 0.5
        mcc = numerator / denominator
    else:
        mcc = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'specificity': specificity,
        'npv': npv
    }


def expand_metapath_with_hierarchy(metapath):
    """Expand a metapath to all hierarchical variants.

    Args:
        metapath: Pipe-separated metapath like "Gene|affects|F|Disease"

    Returns:
        Set of expanded metapaths including hierarchical ancestors
    """
    parts = metapath.split('|')

    # For N-hop: parts = [type1, pred1, dir1, type2, pred2, dir2, ..., typeN]
    # For 1-hop: parts = [type1, pred, dir, type2]

    # Extract types and predicates
    types = []
    predicates = []
    directions = []

    i = 0
    while i < len(parts):
        # Type
        types.append(parts[i])
        i += 1

        if i >= len(parts):
            break

        # Predicate
        predicates.append(parts[i])
        i += 1

        if i >= len(parts):
            break

        # Direction
        directions.append(parts[i])
        i += 1

    # Expand types (handle pseudo-types)
    expanded_types_list = []
    for typ in types:
        if is_pseudo_type(typ):
            # Pseudo-type: expand to constituents and their ancestors
            constituents = parse_pseudo_type(typ)
            all_variants = set()
            for constituent in constituents:
                all_variants.update(get_type_ancestors(constituent))
            expanded_types_list.append(sorted(all_variants))
        else:
            # Regular type: get ancestors
            expanded_types_list.append(sorted(get_type_ancestors(typ)))

    # Expand predicates (normalize to remove biolink: prefix)
    expanded_preds_list = []
    for pred in predicates:
        # Normalize: strip biolink: prefix if present
        clean_pred = pred.replace('biolink:', '')
        # Get ancestors (already returned without biolink: prefix)
        ancestors = get_predicate_ancestors(pred)
        # Include self + ancestors
        expanded_preds_list.append(sorted([clean_pred] + list(ancestors)))

    # Generate all combinations
    def cartesian_product(lists):
        if not lists:
            return [[]]
        result = []
        for item in lists[0]:
            for rest in cartesian_product(lists[1:]):
                result.append([item] + rest)
        return result

    # Build expanded metapaths
    expanded = set()

    # Cartesian product of all type variants
    type_combos = cartesian_product(expanded_types_list)

    # Cartesian product of all predicate variants
    pred_combos = cartesian_product(expanded_preds_list)

    # Combine types and predicates
    for type_combo in type_combos:
        for pred_combo in pred_combos:
            # Interleave types, predicates, directions
            parts_expanded = []
            for i, typ in enumerate(type_combo):
                parts_expanded.append(typ)
                if i < len(pred_combo):
                    parts_expanded.append(pred_combo[i])
                    parts_expanded.append(directions[i])

            expanded.add('|'.join(parts_expanded))

    return expanded


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


def group_type_pair(type1, type2, file_list, output_dir, n_hops, aggregate=True,
                    aggregated_counts=None, type_node_counts=None):
    """Group all N-hop results for 1-hop metapaths between a type pair.

    Args:
        type1: First type (e.g., "Gene")
        type2: Second type (e.g., "Disease")
        file_list: List of result file paths to scan
        output_dir: Output directory for grouped results
        n_hops: Number of hops
        aggregate: Whether to do hierarchical aggregation (default: True)
        aggregated_counts: Dict mapping aggregated path -> global count (from prepare_grouping.py)
        type_node_counts: Dict mapping type name -> node count (for total_possible calculation)
    """
    print(f"Grouping for type pair: ({type1}, {type2})")
    print(f"Aggregation: {'enabled' if aggregate else 'disabled'}")
    if aggregated_counts:
        print(f"Using precomputed counts: {len(aggregated_counts)} paths")
    if type_node_counts:
        print(f"Using precomputed type node counts: {len(type_node_counts)} types")

    # Compute total_possible for this type pair from node counts
    total_possible_for_pair = compute_total_possible(type1, type2, type_node_counts) if type_node_counts else 0
    print(f"Total possible for ({type1}, {type2}): {total_possible_for_pair:,}")

    print(f"Scanning {len(file_list)} result files (optimized subset)...")

    # Data structure: onehop_path -> {nhop_path -> overlap}
    # Note: total_possible is computed per type pair, not summed from rows
    onehop_to_data = defaultdict(lambda: defaultdict(int))

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
                # nhop_count = int(parts[1])  # Don't use - will look up from aggregated_counts
                onehop_path = parts[2]
                # onehop_count = int(parts[3])  # Don't use - will look up from aggregated_counts
                overlap = int(parts[4])
                # total_possible from file is ignored - we compute it from type node counts

                # Check if this 1-hop involves our type pair (hierarchically)
                if check_type_match(onehop_path, type1, type2):
                    rows_found += 1

                    # Expand 1-hop path to all hierarchical variants
                    onehop_variants = expand_metapath_with_hierarchy(onehop_path)

                    # Store overlap for each variant that matches the type pair
                    for onehop_variant in onehop_variants:
                        # Only keep variants that match this job's type pair
                        variant_parts = onehop_variant.split('|')
                        if len(variant_parts) == 4:
                            v_src, v_pred, v_dir, v_tgt = variant_parts
                            # Check exact match (not hierarchical - variants are already expanded)
                            if (v_src == type1 and v_tgt == type2) or (v_src == type2 and v_tgt == type1):
                                onehop_to_data[onehop_variant][nhop_path] += overlap

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
        aggregated = {}
        if aggregate:
            print(f"  Applying hierarchical aggregation...")

            # For each nhop_path, expand and aggregate overlap
            final_aggregated = defaultdict(int)

            for nhop_path, overlap in nhop_data.items():
                # Expand nhop_path to all hierarchical variants
                nhop_variants = expand_metapath_with_hierarchy(nhop_path)

                # Add overlap to all variants
                for variant in nhop_variants:
                    final_aggregated[variant] += overlap

            aggregated = dict(final_aggregated)
            print(f"    N-hop paths expanded to {len(aggregated)} aggregated paths")
        else:
            # No aggregation - just copy data
            aggregated = dict(nhop_data)

        # Write output file for this 1-hop
        safe_filename = onehop_path.replace('|', '_').replace(':', '_').replace(' ', '_')
        output_file = f"{output_dir}/{safe_filename}.tsv"

        print(f"  Writing {len(aggregated)} rows to {output_file}...")

        with open(output_file, 'w') as out:
            # Header
            out.write(f"{n_hops}hop_metapath\t{n_hops}hop_count\toverlap\ttotal_possible\t")
            out.write("precision\trecall\tf1\tmcc\tspecificity\tnpv\n")

            # Sort by overlap descending
            sorted_items = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

            for nhop_path, overlap in sorted_items:
                # Look up nhop_count from precomputed aggregated counts
                if aggregated_counts and nhop_path in aggregated_counts:
                    nhop_count = aggregated_counts[nhop_path]
                else:
                    # Fallback: shouldn't happen
                    print(f"  WARNING: No precomputed count for {nhop_path}")
                    nhop_count = 0

                # Calculate metrics using type-pair total_possible
                metrics = calculate_metrics(overlap, onehop_count_global, nhop_count, total_possible_for_pair)

                # Write row
                out.write(f"{nhop_path}\t{nhop_count}\t{overlap}\t{total_possible_for_pair}\t")
                out.write(f"{metrics['precision']:.6f}\t{metrics['recall']:.6f}\t")
                out.write(f"{metrics['f1']:.6f}\t{metrics['mcc']:.6f}\t")
                out.write(f"{metrics['specificity']:.6f}\t{metrics['npv']:.6f}\n")

        print(f"  ✓ Output written")

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
        help='Path to aggregated_path_counts.json (from prepare_grouping.py)'
    )
    parser.add_argument(
        '--type-node-counts',
        type=str,
        required=True,
        help='Path to type_node_counts.json (from prepare_grouping.py)'
    )

    args = parser.parse_args()

    # Read file list from file
    with open(args.file_list, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    # Load precomputed data
    aggregated_counts = load_aggregated_counts(args.aggregated_counts)
    type_node_counts = load_type_node_counts(args.type_node_counts)

    group_type_pair(
        type1=args.type1,
        type2=args.type2,
        file_list=file_list,
        output_dir=args.output_dir,
        n_hops=args.n_hops,
        aggregate=not args.explicit_only,
        aggregated_counts=aggregated_counts,
        type_node_counts=type_node_counts
    )


if __name__ == "__main__":
    main()
