#!/usr/bin/env python3
"""
Group N-hop metapath results by their corresponding 1-hop metapath.

This script:
1. Reads all results_matrix1_*.tsv files
2. Normalizes 1-hop metapaths to forward (F) direction
3. When reversing a 1-hop, also reverses the corresponding N-hop metapath
4. Groups results by normalized 1-hop metapath
5. Writes each group to a separate output file

Example normalization:
- 1-hop: Disease|has_adverse_event|R|Drug -> Drug|has_adverse_event|F|Disease
- N-hop: A|p1|F|B|p2|R|C|p3|F|D -> D|p3|R|C|p2|F|B|p1|R|A (reversed)
"""

import os
import re
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple
import math


def parse_metapath(metapath: str) -> Tuple[list, list, list]:
    """
    Parse a metapath into node types, predicates, and directions.

    Metapath format: Type1|pred1|dir1|Type2|pred2|dir2|...|TypeN
    - N-hop path has N+1 node types, N predicates, N directions
    - Number of parts = 1 + 3*N (for N hops)

    Examples:
        1-hop: "A|pred1|F|B" -> nodes=['A', 'B'], predicates=['pred1'], directions=['F']
        2-hop: "A|p1|F|B|p2|R|C" -> nodes=['A', 'B', 'C'], predicates=['p1', 'p2'], directions=['F', 'R']
        3-hop: "A|p1|F|B|p2|R|C|p3|F|D" -> nodes=['A', 'B', 'C', 'D'], ...

    Returns:
        Tuple of (nodes, predicates, directions)
    """
    parts = metapath.split('|')
    num_parts = len(parts)

    # Formula: num_parts = 1 + 3*n_hops
    # So: n_hops = (num_parts - 1) / 3
    if (num_parts - 1) % 3 != 0 or num_parts < 4:
        raise ValueError(f"Invalid metapath format: {metapath} (parts: {num_parts}, expected 1+3*N)")

    n_hops = (num_parts - 1) // 3

    # Extract nodes: positions 0, 3, 6, 9, ... (every 3rd starting at 0)
    nodes = [parts[i * 3] for i in range(n_hops + 1)]

    # Extract predicates: positions 1, 4, 7, 10, ... (every 3rd starting at 1)
    predicates = [parts[i * 3 + 1] for i in range(n_hops)]

    # Extract directions: positions 2, 5, 8, 11, ... (every 3rd starting at 2)
    directions = [parts[i * 3 + 2] for i in range(n_hops)]

    return nodes, predicates, directions


def build_metapath(nodes: list, predicates: list, directions: list) -> str:
    """Build metapath string from components."""
    result = []
    for i in range(len(predicates)):
        result.append(nodes[i])
        result.append(predicates[i])
        result.append(directions[i])
    result.append(nodes[-1])
    return '|'.join(result)


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


def calculate_metrics(threehop_count: int, onehop_count: int, overlap: int, total_possible: int) -> dict:
    """
    Calculate prediction metrics from N-hop and 1-hop metapath overlap.

    Args:
        threehop_count: Number of N-hop paths (parameter name kept for backwards compat)
        onehop_count: Number of 1-hop paths
        overlap: Number of node pairs with both N-hop and 1-hop paths
        total_possible: Total possible node pairs

    Confusion matrix:
    - TP (True Positives): overlap
    - FP (False Positives): nhop_count - overlap
    - FN (False Negatives): 1hop_count - overlap
    - TN (True Negatives): total_possible - nhop_count - 1hop_count + overlap

    Returns dict with all calculated metrics.
    """
    # Confusion matrix (threehop_count is actually nhop_count)
    TP = overlap
    FP = threehop_count - overlap
    FN = onehop_count - overlap
    TN = total_possible - threehop_count - onehop_count + overlap
    Total = TP + FP + FN + TN

    # Basic metrics with division by zero protection
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0.0  # Negative Predictive Value
    Accuracy = (TP + TN) / Total if Total > 0 else 0.0

    # Derived metrics
    TPR = Recall  # True Positive Rate = Recall
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # False Positive Rate
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # False Negative Rate

    # Balanced Accuracy
    Balanced_Accuracy = (TPR + Specificity) / 2.0

    # F1 Score
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

    # Matthews Correlation Coefficient
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = ((TP * TN) - (FP * FN)) / denominator if denominator > 0 else 0.0

    # Likelihood Ratios
    PLR = TPR / FPR if FPR > 0 else float('inf')  # Positive Likelihood Ratio
    NLR = FNR / Specificity if Specificity > 0 else float('inf')  # Negative Likelihood Ratio

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'Total': Total,
        'Precision': Precision,
        'Recall': Recall,
        'Specificity': Specificity,
        'NPV': NPV,
        'Accuracy': Accuracy,
        'Balanced_Accuracy': Balanced_Accuracy,
        'F1': F1,
        'MCC': MCC,
        'TPR': TPR,
        'FPR': FPR,
        'FNR': FNR,
        'PLR': PLR,
        'NLR': NLR
    }


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
    args = parser.parse_args()

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
    file_manager = FileHandleManager(output_dir, enhanced_header, max_open=args.max_open_files)

    # Statistics
    total_lines = 0
    files_processed = 0
    unique_1hop_metapaths = set()

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
                    metrics = calculate_metrics(threehop_count, onehop_count, overlap, total_possible)

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


if __name__ == '__main__':
    main()
