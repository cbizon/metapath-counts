#!/usr/bin/env python3
"""
Group 3-hop metapath results by their corresponding 1-hop metapath.

This script:
1. Reads all results_matrix1_*.tsv files
2. Normalizes 1-hop metapaths to forward (F) direction
3. When reversing a 1-hop, also reverses the corresponding 3-hop metapath
4. Groups results by normalized 1-hop metapath
5. Writes each group to a separate output file

Example normalization:
- 1-hop: Disease|has_adverse_event|R|Drug -> Drug|has_adverse_event|F|Disease
- 3-hop: A|p1|F|B|p2|R|C|p3|F|D -> D|p3|R|C|p2|F|B|p1|R|A (reversed)
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

    Example: "A|pred1|F|B|pred2|R|C" ->
        nodes=['A', 'B', 'C'], predicates=['pred1', 'pred2'], directions=['F', 'R']
    """
    parts = metapath.split('|')

    # For 1-hop: Type1|predicate|direction|Type2 (4 parts)
    # For 3-hop: Type1|pred1|dir1|Type2|pred2|dir2|Type3|pred3|dir3|Type4 (10 parts)

    if len(parts) == 4:
        # 1-hop: parts[0]|parts[1]|parts[2]|parts[3]
        return [parts[0], parts[3]], [parts[1]], [parts[2]]
    elif len(parts) == 10:
        # 3-hop: parts[0]|parts[1]|parts[2]|parts[3]|parts[4]|parts[5]|parts[6]|parts[7]|parts[8]|parts[9]
        # Node1|pred1|dir1|Node2|pred2|dir2|Node3|pred3|dir3|Node4
        nodes = [parts[0], parts[3], parts[6], parts[9]]
        predicates = [parts[1], parts[4], parts[7]]
        directions = [parts[2], parts[5], parts[8]]
        return nodes, predicates, directions
    else:
        raise ValueError(f"Unexpected metapath format: {metapath} (parts: {len(parts)})")


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


def reverse_3hop(metapath: str) -> str:
    """
    Reverse a 3-hop metapath.

    Example: A|p1|F|B|p2|R|C|p3|F|D -> D|p3|R|C|p2|F|B|p1|R|A
    """
    nodes, predicates, directions = parse_metapath(metapath)

    # Reverse nodes
    nodes_rev = nodes[::-1]
    # Reverse predicates
    predicates_rev = predicates[::-1]
    # Reverse directions and flip F<->R
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


def canonicalize_row(threehop: str, onehop: str) -> Tuple[str, str]:
    """
    Canonicalize a row by ensuring both metapaths follow canonical type pair direction.

    If the row's direction doesn't match canonical, reverse BOTH metapaths.

    Returns: (canonicalized_threehop, canonicalized_onehop)
    """
    # Get endpoint types
    threehop_start, threehop_end = get_endpoint_types(threehop)
    onehop_start, onehop_end = get_endpoint_types(onehop)

    # They should match
    assert threehop_start == onehop_start and threehop_end == onehop_end, \
        f"Metapath endpoints don't match: 3-hop ({threehop_start}->{threehop_end}) vs 1-hop ({onehop_start}->{onehop_end})"

    # Determine canonical direction
    canonical_start, canonical_end = get_canonical_direction(threehop_start, threehop_end)

    # If current direction matches canonical, keep as-is
    if threehop_start == canonical_start:
        return threehop, onehop

    # Otherwise, reverse both
    return reverse_3hop(threehop), reverse_3hop(onehop)


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
    Calculate prediction metrics from 3-hop and 1-hop metapath overlap.

    Following the notebook's approach:
    - TP (True Positives): overlap
    - FP (False Positives): 3hop_count - overlap
    - FN (False Negatives): 1hop_count - overlap
    - TN (True Negatives): total_possible - 3hop_count - 1hop_count + overlap

    Returns dict with all calculated metrics.
    """
    # Confusion matrix
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
        description='Group 3-hop metapaths by normalized 1-hop metapath with metric calculations'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with test_results directory')
    parser.add_argument('--max-open-files', type=int, default=500,
                        help='Maximum number of file handles to keep open at once (default: 500)')
    args = parser.parse_args()

    # Input and output directories
    if args.test:
        results_dir = Path('scripts/metapaths/test_results')
        output_dir = Path('scripts/metapaths/test_output')
    else:
        results_dir = Path('scripts/metapaths/results')
        output_dir = Path('scripts/metapaths/grouped_by_1hop')
    output_dir.mkdir(exist_ok=True)

    # Get all result files
    result_files = sorted(results_dir.glob('results_matrix1_*.tsv'))
    print(f"Found {len(result_files)} result files to process")

    # Enhanced header with calculated metrics
    # Original: 3hop_metapath, 3hop_count, 1hop_metapath, 1hop_count, overlap, total_possible
    enhanced_header = '\t'.join([
        '3hop_metapath',
        '3hop_count',
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
