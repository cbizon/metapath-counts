#!/usr/bin/env python3
"""
Group N-hop metapath results by their corresponding 1-hop metapath.

This script:
1. Reads aggregated results (from merged SLURM job outputs)
2. Normalizes 1-hop metapaths to canonical direction (alphabetically)
3. When reversing a 1-hop, also reverses the corresponding N-hop metapath
4. Groups results by normalized 1-hop metapath
5. Calculates prediction metrics (Precision, Recall, F1, MCC, etc.)
6. Writes each group to a separate output file

Note: Hierarchical aggregation (pseudo-type expansion, ancestor aggregation)
is now performed during SLURM jobs, so input is already aggregated.

Example normalization:
- 1-hop: Disease|has_adverse_event|R|Drug -> Drug|has_adverse_event|F|Disease
- N-hop: A|p1|F|B|p2|R|C|p3|F|D -> D|p3|R|C|p2|F|B|p1|R|A (reversed)
"""

import os
import re
import time
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, List, Iterator
import math
import itertools
from metapath_counts import (
    is_pseudo_type,
    parse_pseudo_type,
    get_type_ancestors,
    get_predicate_ancestors,
    parse_metapath,
    build_metapath,
    get_type_variants,
    get_predicate_variants,
    generate_metapath_variants,
    calculate_metrics
)


def normalize_1hop(metapath: str) -> Tuple[str, bool]:
    """
    Normalize a 1-hop metapath to forward direction.

    Returns:
        (normalized_metapath, was_reversed)
    """
    nodes, predicates, directions = parse_metapath(metapath)

    if directions[0] == 'F':
        return metapath, False
    elif directions[0] == 'R':
        # Reverse: Type1|pred|R|Type2 -> Type2|pred|F|Type1
        normalized = build_metapath([nodes[1], nodes[0]], predicates, ['F'])
        return normalized, True
    elif directions[0] == 'A':
        # ANY direction - keep as is
        return metapath, False
    else:
        raise ValueError(f"Unknown direction: {directions[0]} in {metapath}")


def reverse_metapath(metapath: str) -> str:
    """
    Reverse an N-hop metapath.

    Reverses the path direction by:
    1. Reversing the order of nodes
    2. Reversing the order of predicates
    3. Reversing the order of directions and flipping F<->R (A stays A)

    Examples:
        1-hop: "A|p1|F|B" -> "B|p1|R|A"
        2-hop: "A|p1|F|B|p2|R|C" -> "C|p2|F|B|p1|R|A"
        3-hop: "A|p1|F|B|p2|R|C|p3|F|D" -> "D|p3|R|C|p2|F|B|p1|R|A"

    Args:
        metapath: The metapath string to reverse

    Returns:
        The reversed metapath string
    """
    nodes, predicates, directions = parse_metapath(metapath)

    # Reverse nodes
    nodes_rev = nodes[::-1]
    # Reverse predicates
    predicates_rev = predicates[::-1]
    # Reverse directions and flip F<->R (A stays A)
    directions_rev = []
    for d in directions[::-1]:
        if d == 'F':
            directions_rev.append('R')
        elif d == 'R':
            directions_rev.append('F')
        elif d == 'A':
            directions_rev.append('A')
        else:
            raise ValueError(f"Unknown direction: {d}")

    return build_metapath(nodes_rev, predicates_rev, directions_rev)


# Keep old name for backwards compatibility
def reverse_3hop(metapath: str) -> str:
    """Alias for reverse_metapath (backwards compatibility)."""
    return reverse_metapath(metapath)


def get_endpoint_types(metapath: str) -> Tuple[str, str]:
    """
    Extract start and end node types from a metapath.

    Example: "Drug|has_adverse_event|F|Disease" -> ("Drug", "Disease")
    Example: "Drug|...|F|Disease|...|A|Gene|...|F|Disease" -> ("Drug", "Disease")
    """
    nodes, predicates, directions = parse_metapath(metapath)
    return nodes[0], nodes[-1]


def get_canonical_direction(type1: str, type2: str) -> Tuple[str, str]:
    """
    Determine canonical direction for a type pair (alphabetically).

    Returns: (start_type, end_type) in canonical order

    Example: ("Disease", "Drug") -> ("Disease", "Drug")
    Example: ("Drug", "Disease") -> ("Disease", "Drug")
    """
    if type1 <= type2:
        return type1, type2
    else:
        return type2, type1


def canonicalize_row(nhop: str, onehop: str) -> Tuple[str, str]:
    """
    Canonicalize a row by ensuring both metapaths follow canonical type pair direction.

    If the row's direction doesn't match canonical, reverse BOTH metapaths.

    Args:
        nhop: The N-hop metapath (predictor)
        onehop: The 1-hop metapath (target)

    Returns:
        Tuple of (canonicalized_nhop, canonicalized_onehop)
    """
    # Get endpoint types
    nhop_start, nhop_end = get_endpoint_types(nhop)
    onehop_start, onehop_end = get_endpoint_types(onehop)

    # They should match
    assert nhop_start == onehop_start and nhop_end == onehop_end, \
        f"Metapath endpoints don't match: N-hop ({nhop_start}->{nhop_end}) vs 1-hop ({onehop_start}->{onehop_end})"

    # Determine canonical direction
    canonical_start, canonical_end = get_canonical_direction(nhop_start, nhop_end)

    # If current direction matches canonical, keep as-is
    if nhop_start == canonical_start:
        return nhop, onehop

    # Otherwise, reverse both
    return reverse_metapath(nhop), reverse_metapath(onehop)


def safe_filename(metapath: str) -> str:
    """Convert metapath to safe filename."""
    # Replace | with _ and handle special characters
    name = metapath.replace('|', '_')
    # Remove or replace any other problematic characters
    name = re.sub(r'[^\w\-]', '_', name)
    return name + '.tsv'


class FileHandleManager:
    """
    Manages file handles with LRU eviction to avoid 'too many open files' error.

    Keeps at most max_open files open at once. When the limit is reached,
    closes the least recently used file.
    """

    def __init__(self, output_dir: Path, header: str, max_open: int = 500):
        self.output_dir = output_dir
        self.header = header
        self.max_open = max_open
        self.handles = OrderedDict()  # metapath -> file handle
        self.initialized = set()  # metapaths that have been initialized (header written)

    def get_handle(self, metapath: str):
        """Get file handle for a metapath, opening if needed."""
        # If already open, move to end (most recently used)
        if metapath in self.handles:
            self.handles.move_to_end(metapath)
            return self.handles[metapath]

        # Need to open a new file
        # First check if we need to close an old one
        if len(self.handles) >= self.max_open:
            # Close the least recently used file (first item)
            lru_metapath, lru_handle = self.handles.popitem(last=False)
            lru_handle.close()

        # Open the file
        filename = safe_filename(metapath)
        filepath = self.output_dir / filename

        # Check if this is first time opening (need to write header)
        if metapath not in self.initialized:
            handle = open(filepath, 'w')
            handle.write(self.header + '\n')
            self.initialized.add(metapath)
        else:
            # Reopening - append mode
            handle = open(filepath, 'a')

        # Add to our cache
        self.handles[metapath] = handle
        return handle

    def close_all(self):
        """Close all open file handles."""
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def aggregate_results(explicit_results: List[Tuple]) -> Dict[Tuple[str, str], Tuple[int, int, int, int]]:
    """
    Aggregate explicit results to include all implied hierarchical paths.

    This is where "de-re-aggregation" happens:
    1. Expand pseudo-types to constituent types
    2. Aggregate to ancestor types
    3. Aggregate to ancestor predicates

    Args:
        explicit_results: List of (nhop_path, nhop_count, 1hop_path, 1hop_count, overlap, total_possible)

    Returns:
        Dict mapping (nhop_path, 1hop_path) -> (nhop_count, 1hop_count, overlap, total_possible)
        where counts are summed across all explicit paths that imply this pair
    """
    # Step 1: Extract all unique metapaths
    print("  [2a/2] Extracting unique metapaths...")
    unique_nhop_paths = set()
    unique_1hop_paths = set()
    for nhop_path, _, onehop_path, _, _, _ in explicit_results:
        unique_nhop_paths.add(nhop_path)
        unique_1hop_paths.add(onehop_path)

    print(f"    Found {len(unique_nhop_paths):,} unique N-hop paths")
    print(f"    Found {len(unique_1hop_paths):,} unique 1-hop paths")

    # Step 2: Pre-compute variants for all unique N-hop paths
    print("  [2b/2] Pre-computing N-hop path variants...")
    nhop_variants_cache = {}
    for i, path in enumerate(unique_nhop_paths, 1):
        if i % 100 == 0:
            print(f"    Processed {i:,}/{len(unique_nhop_paths):,} N-hop paths")
        nhop_variants_cache[path] = list(generate_metapath_variants(path))
    print(f"    Completed {len(unique_nhop_paths):,} N-hop paths")

    # Step 3: Pre-compute variants for all unique 1-hop paths
    print("  [2c/2] Pre-computing 1-hop path variants...")
    onehop_variants_cache = {}
    for i, path in enumerate(unique_1hop_paths, 1):
        if i % 100 == 0:
            print(f"    Processed {i:,}/{len(unique_1hop_paths):,} 1-hop paths")
        onehop_variants_cache[path] = list(generate_metapath_variants(path))
    print(f"    Completed {len(unique_1hop_paths):,} 1-hop paths")

    # Step 4: Compute aggregated counts for each nhop variant (independent of onehop)
    # This fixes the bug where nhop_count was multiplied by number of onehop variants
    print(f"  [2d/2] Computing independent variant counts...")
    unique_nhop_counts = {}
    unique_onehop_counts = {}
    for nhop_path, nhop_count, onehop_path, onehop_count, _, _ in explicit_results:
        if nhop_path not in unique_nhop_counts:
            unique_nhop_counts[nhop_path] = nhop_count
        if onehop_path not in unique_onehop_counts:
            unique_onehop_counts[onehop_path] = onehop_count

    # Aggregate nhop counts to variants
    nhop_variant_counts = defaultdict(int)
    for nhop_path, nhop_count in unique_nhop_counts.items():
        for nhop_variant in nhop_variants_cache[nhop_path]:
            nhop_variant_counts[nhop_variant] += nhop_count

    # Aggregate onehop counts to variants
    onehop_variant_counts = defaultdict(int)
    for onehop_path, onehop_count in unique_onehop_counts.items():
        for onehop_variant in onehop_variants_cache[onehop_path]:
            onehop_variant_counts[onehop_variant] += onehop_count

    print(f"    {len(nhop_variant_counts):,} N-hop variants, {len(onehop_variant_counts):,} 1-hop variants")

    # Step 5: Aggregate overlap and total_possible for each (nhop_variant, onehop_variant) pair
    print(f"  [2e/2] Aggregating pair data...")
    print(f"    Total explicit results to process: {len(explicit_results):,}")
    pair_data = defaultdict(lambda: [0, 0])  # [overlap, total_possible]

    start_time = time.time()
    for i, (nhop_path, nhop_count, onehop_path, onehop_count, overlap, total_possible) in enumerate(explicit_results, 1):
        # Show progress: first few, then every 1000, then every 10000
        if i == 1 or i == 10 or i == 100 or i == 1000 or (i < 10000 and i % 1000 == 0) or i % 10000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(explicit_results) - i) / rate if rate > 0 else 0
            eta_mins = remaining / 60
            print(f"    Aggregated {i:,}/{len(explicit_results):,} ({100*i/len(explicit_results):.1f}%) | "
                  f"Rate: {rate:.0f} rows/sec | ETA: {eta_mins:.1f} min")

        # Use cached variants - only accumulate overlap and total_possible
        for nhop_variant in nhop_variants_cache[nhop_path]:
            for onehop_variant in onehop_variants_cache[onehop_path]:
                key = (nhop_variant, onehop_variant)
                agg = pair_data[key]
                agg[0] += overlap
                agg[1] += total_possible

    elapsed_total = time.time() - start_time
    print(f"    Completed {len(explicit_results):,} explicit results in {elapsed_total/60:.1f} minutes")

    # Step 6: Combine into final aggregated results
    aggregated = {}
    for (nhop_variant, onehop_variant), (overlap, total_possible) in pair_data.items():
        aggregated[(nhop_variant, onehop_variant)] = (
            nhop_variant_counts[nhop_variant],
            onehop_variant_counts[onehop_variant],
            overlap,
            total_possible
        )

    print(f"    Generated {len(aggregated):,} aggregated results")

    return aggregated


def main():
    import sys
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Group N-hop metapaths by normalized 1-hop metapath with metric calculations'
    )
    parser.add_argument('--n-hops', type=int, default=3,
                        help='Number of hops analyzed (default: 3)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input directory or file (default: results_{n_hops}hop/all_{n_hops}hop_overlaps.tsv)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for grouped files (default: grouped_by_results_{n_hops}hop)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with test_results directory')
    parser.add_argument('--max-open-files', type=int, default=500,
                        help='Maximum number of file handles to keep open at once (default: 500)')
    parser.add_argument('--aggregate', action='store_true',
                        help='Perform hierarchical aggregation (expand pseudo-types and aggregate to ancestors). NOTE: Aggregation is now done during SLURM jobs by default, so this is rarely needed.')
    parser.add_argument('--explicit-only', action='store_true',
                        help='DEPRECATED: Aggregation is now off by default')
    args = parser.parse_args()

    # Determine aggregation mode
    # NEW DEFAULT: aggregation is OFF (done during SLURM jobs)
    do_aggregation = args.aggregate

    # Input and output directories
    if args.test:
        results_dir = Path('test_results')
        output_dir = Path('test_output')
    else:
        # Use n_hops to determine defaults
        if args.input:
            input_path = Path(args.input)
            # If it's a directory, look for result files
            if input_path.is_dir():
                results_dir = input_path
            else:
                # It's a single merged file
                results_dir = input_path.parent
        else:
            results_dir = Path(f'results_{args.n_hops}hop')

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f'grouped_by_results_{args.n_hops}hop')

    output_dir.mkdir(exist_ok=True)

    # Get all result files or single merged file
    if args.input and Path(args.input).is_file():
        # Single merged file
        result_files = [Path(args.input)]
    else:
        # Directory of individual result files
        result_files = sorted(results_dir.glob('results_matrix1_*.tsv'))

    print(f"Found {len(result_files)} result file(s) to process")

    # Enhanced header with calculated metrics
    # Original: Nhop_metapath, Nhop_count, 1hop_metapath, 1hop_count, overlap, total_possible
    enhanced_header = '\t'.join([
        f'{args.n_hops}hop_metapath',
        f'{args.n_hops}hop_count',
        '1hop_metapath',
        '1hop_count',
        'overlap',
        'total_possible',
        'TP',
        'FP',
        'FN',
        'TN',
        'Total',
        'Precision',
        'Recall',
        'Specificity',
        'NPV',
        'Accuracy',
        'Balanced_Accuracy',
        'F1',
        'MCC',
        'TPR',
        'FPR',
        'FNR',
        'PLR',
        'NLR'
    ])

    # Initialize file handle manager
    print(f"Max open file handles: {args.max_open_files}")
    print(f"Hierarchical aggregation: {'ENABLED (rare, results should already be aggregated)' if do_aggregation else 'DISABLED (default, aggregation done in SLURM jobs)'}")
    file_manager = FileHandleManager(output_dir, enhanced_header, max_open=args.max_open_files)

    # Statistics
    total_lines = 0
    files_processed = 0
    unique_1hop_metapaths = set()

    # Step 1: Read all explicit results (needed for aggregation)
    if do_aggregation:
        print("\n[1/2] Reading explicit results for aggregation...")
        explicit_results = []

        for input_file in result_files:
            files_processed += 1
            if files_processed % 100 == 0:
                print(f"  Reading file {files_processed}/{len(result_files)}: {input_file.name}")

            with open(input_file, 'r') as f:
                # Skip header
                f.readline()

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    total_lines += 1
                    if total_lines % 1000000 == 0:
                        print(f"    Read {total_lines:,} lines")

                    # Parse the line
                    parts = line.split('\t')
                    if len(parts) != 6:
                        print(f"Warning: skipping malformed line with {len(parts)} columns")
                        continue

                    nhop_path = parts[0]
                    nhop_count = int(parts[1])
                    onehop_path = parts[2]
                    onehop_count = int(parts[3])
                    overlap = int(parts[4])
                    total_possible = int(parts[5])

                    explicit_results.append((nhop_path, nhop_count, onehop_path, onehop_count, overlap, total_possible))

        print(f"  Read {len(explicit_results):,} explicit result rows")

        # Step 2: Perform aggregation
        print("\n[2/2] Performing hierarchical aggregation...")
        aggregated_results = aggregate_results(explicit_results)

        # Process aggregated results
        print("\n[3/3] Writing grouped results...")
        files_processed = 0
        for (nhop_path, onehop_path), (nhop_count, onehop_count, overlap, total_possible) in aggregated_results.items():
            files_processed += 1
            if files_processed % 100000 == 0:
                print(f"  Processed {files_processed:,}/{len(aggregated_results):,} aggregated rows")

            # Canonicalize row
            nhop_path, onehop_path = canonicalize_row(nhop_path, onehop_path)

            # Calculate metrics
            metrics = calculate_metrics(nhop_count, onehop_count, overlap, total_possible, full_metrics=True)

            # Track unique 1-hop metapaths
            unique_1hop_metapaths.add(onehop_path)

            # Get file handle
            handle = file_manager.get_handle(onehop_path)

            # Write output line
            output_line = '\t'.join([
                nhop_path,
                str(nhop_count),
                onehop_path,
                str(onehop_count),
                str(overlap),
                str(total_possible),
                str(metrics['TP']),
                str(metrics['FP']),
                str(metrics['FN']),
                str(metrics['TN']),
                str(metrics['Total']),
                f"{metrics['Precision']:.6f}",
                f"{metrics['Recall']:.6f}",
                f"{metrics['Specificity']:.6f}",
                f"{metrics['NPV']:.6f}",
                f"{metrics['Accuracy']:.6f}",
                f"{metrics['Balanced_Accuracy']:.6f}",
                f"{metrics['F1']:.6f}",
                f"{metrics['MCC']:.6f}",
                f"{metrics['TPR']:.6f}",
                f"{metrics['FPR']:.6f}",
                f"{metrics['FNR']:.6f}",
                'inf' if math.isinf(metrics['PLR']) else f"{metrics['PLR']:.6f}",
                'inf' if math.isinf(metrics['NLR']) else f"{metrics['NLR']:.6f}"
            ])
            handle.write(output_line + '\n')

        # Close file handles
        file_manager.close_all()

    else:
        # Process explicit results directly (no aggregation)
        print("\nProcessing explicit results (no aggregation)...")

        try:
            for input_file in result_files:
                files_processed += 1
                if files_processed % 100 == 0:
                    print(f"Processing file {files_processed}/{len(result_files)}: {input_file.name}")

                with open(input_file, 'r') as f:
                    # Skip original header
                    f.readline()

                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        total_lines += 1
                        if total_lines % 1000000 == 0:
                            print(f"  Processed {total_lines:,} lines, {len(unique_1hop_metapaths)} unique 1-hop metapaths, {len(file_manager.handles)} files open")

                        # Parse the line
                        parts = line.split('\t')
                        if len(parts) != 6:
                            print(f"Warning: skipping malformed line with {len(parts)} columns")
                            continue

                        threehop_metapath = parts[0]
                        threehop_count = int(parts[1])
                        onehop_metapath = parts[2]
                        onehop_count = int(parts[3])
                        overlap = int(parts[4])
                        total_possible = int(parts[5])

                        # Canonicalize row: reverse BOTH metapaths if needed to match canonical type pair direction
                        threehop_metapath, onehop_metapath = canonicalize_row(threehop_metapath, onehop_metapath)

                        # Calculate metrics
                        metrics = calculate_metrics(threehop_count, onehop_count, overlap, total_possible, full_metrics=True)

                        # Track unique 1-hop metapaths
                        unique_1hop_metapaths.add(onehop_metapath)

                        # Get file handle (will open/reopen as needed)
                        handle = file_manager.get_handle(onehop_metapath)

                        # Write the enhanced output line
                        output_line = '\t'.join([
                            threehop_metapath,
                            str(threehop_count),
                            onehop_metapath,
                            str(onehop_count),
                            str(overlap),
                            str(total_possible),
                            str(metrics['TP']),
                            str(metrics['FP']),
                            str(metrics['FN']),
                            str(metrics['TN']),
                            str(metrics['Total']),
                            f"{metrics['Precision']:.6f}",
                            f"{metrics['Recall']:.6f}",
                            f"{metrics['Specificity']:.6f}",
                            f"{metrics['NPV']:.6f}",
                            f"{metrics['Accuracy']:.6f}",
                            f"{metrics['Balanced_Accuracy']:.6f}",
                            f"{metrics['F1']:.6f}",
                            f"{metrics['MCC']:.6f}",
                            f"{metrics['TPR']:.6f}",
                            f"{metrics['FPR']:.6f}",
                            f"{metrics['FNR']:.6f}",
                            'inf' if math.isinf(metrics['PLR']) else f"{metrics['PLR']:.6f}",
                            'inf' if math.isinf(metrics['NLR']) else f"{metrics['NLR']:.6f}"
                        ])
                        handle.write(output_line + '\n')

            print(f"\nProcessing complete!")
            print(f"  Total lines processed: {total_lines:,}")
            print(f"  Unique 1-hop metapaths: {len(unique_1hop_metapaths)}")
            print(f"  Currently open file handles: {len(file_manager.handles)}")
            print(f"  Output files written to: {output_dir}")

        finally:
            # Close all open file handles
            file_manager.close_all()

    # Final summary
    print(f"\nProcessing complete!")
    print(f"  Mode: {'Aggregated' if do_aggregation else 'Explicit only'}")
    print(f"  Unique 1-hop metapaths: {len(unique_1hop_metapaths)}")
    print(f"  Output files written to: {output_dir}")


if __name__ == '__main__':
    main()
