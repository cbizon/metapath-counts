#!/usr/bin/env python3
"""
Merge individual matrix1 result files into a single output file.

This script:
1. Finds all results_matrix1_*.tsv files
2. Verifies all expected indices are present
3. Merges them by SUMMING counts for duplicate (nhop_path, onehop_path) keys
4. Reports statistics

Note: Now that results are aggregated within each job, the same
(nhop_path, onehop_path) pair can appear in multiple files and
must be summed (not concatenated).

Usage:
    # 3-hop (default)
    uv run python scripts/merge_results.py

    # Custom N-hop
    uv run python scripts/merge_results.py --n-hops 2
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


def extract_index_from_filename(filename):
    """Extract matrix1 index from filename like 'results_matrix1_042.tsv'."""
    match = re.search(r'results_matrix1_(\d+)\.tsv', filename)
    if match:
        return int(match.group(1))
    return None


def merge_results(results_dir=None, output_file=None, n_hops=3):
    """Merge all result files into a single output file.

    Args:
        results_dir: Directory containing result files (default: results_{n_hops}hop)
        output_file: Output file path (default: results_dir/all_{n_hops}hop_overlaps.tsv)
        n_hops: Number of hops analyzed (default: 3)
    """
    if results_dir is None:
        results_dir = f"results_{n_hops}hop"

    print("=" * 80)
    print(f"MERGING {n_hops}-HOP ANALYSIS RESULTS")
    print("=" * 80)

    # Load manifest to get expected count
    manifest_path = os.path.join(results_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Count only actual job entries (skip _metadata)
    expected_count = len(manifest) - 1 if "_metadata" in manifest else len(manifest)
    completed_jobs = [k for k, v in manifest.items() if k != "_metadata" and v["status"] == "completed"]
    completed_count = len(completed_jobs)

    print(f"\nExpected jobs: {expected_count}")
    print(f"Completed jobs (from manifest): {completed_count}")

    if completed_count < expected_count:
        print(f"\nWARNING: Only {completed_count}/{expected_count} jobs completed!")
        print("Some results may be missing. Continuing with available results...")

    # Find all result files
    pattern = os.path.join(results_dir, "results_matrix1_*.tsv")
    result_files = sorted(glob.glob(pattern))

    print(f"\nFound {len(result_files)} result files")

    if len(result_files) == 0:
        raise FileNotFoundError(f"No result files found matching: {pattern}")

    # Extract indices and verify
    found_indices = set()
    file_by_index = {}

    for file_path in result_files:
        idx = extract_index_from_filename(os.path.basename(file_path))
        if idx is not None:
            found_indices.add(idx)
            file_by_index[idx] = file_path

    expected_indices = set(range(expected_count))
    missing_indices = expected_indices - found_indices
    extra_indices = found_indices - expected_indices

    if missing_indices:
        print(f"\nWARNING: Missing results for {len(missing_indices)} indices:")
        missing_list = sorted(missing_indices)
        if len(missing_list) <= 20:
            print(f"  {missing_list}")
        else:
            print(f"  First 20: {missing_list[:20]}")

    if extra_indices:
        print(f"\nWARNING: Found {len(extra_indices)} unexpected result files:")
        print(f"  {sorted(extra_indices)}")

    # Determine output file path
    if output_file is None:
        output_file = os.path.join(results_dir, f"all_{n_hops}hop_overlaps.tsv")

    print(f"\nMerging {len(file_by_index)} files into: {output_file}")
    print(f"Note: Summing counts for duplicate (nhop_path, onehop_path) keys...")

    # Merge files by summing duplicate keys
    # aggregated: (nhop_path, onehop_path) -> [nhop_count, onehop_count, overlap, total_possible]
    aggregated = defaultdict(lambda: [0, 0, 0, 0])
    total_rows_read = 0
    files_processed = 0
    header = None

    # Read all files and aggregate
    for idx in sorted(file_by_index.keys()):
        file_path = file_by_index[idx]
        files_processed += 1

        with open(file_path, 'r') as f:
            # Read header from first file
            file_header = f.readline().strip()
            if header is None:
                header = file_header

            # Read data rows
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 6:
                    print(f"Warning: skipping malformed line in {file_path}: {len(parts)} columns")
                    continue

                nhop_path = parts[0]
                nhop_count = int(parts[1])
                onehop_path = parts[2]
                onehop_count = int(parts[3])
                overlap = int(parts[4])
                total_possible = int(parts[5])

                # Accumulate counts
                key = (nhop_path, onehop_path)
                agg = aggregated[key]
                agg[0] += nhop_count
                agg[1] += onehop_count
                agg[2] += overlap
                agg[3] += total_possible

                total_rows_read += 1

        if (files_processed % 50 == 0) or (files_processed == len(file_by_index)):
            print(f"  Processed {files_processed}/{len(file_by_index)} files ({total_rows_read:,} rows read, {len(aggregated):,} unique keys)")

    # Write merged results
    print(f"\nWriting {len(aggregated):,} unique rows to {output_file}...")
    with open(output_file, 'w') as out:
        out.write(header + '\n')

        rows_written = 0
        for (nhop_path, onehop_path), (nhop_count, onehop_count, overlap, total_possible) in aggregated.items():
            out.write(f"{nhop_path}\t{nhop_count}\t{onehop_path}\t{onehop_count}\t{overlap}\t{total_possible}\n")
            rows_written += 1

            if rows_written % 100000 == 0:
                print(f"  Written {rows_written:,}/{len(aggregated):,} rows")

    total_rows = len(aggregated)

    print(f"\n{'=' * 80}")
    print("MERGE COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nStatistics:")
    print(f"  Files merged: {files_processed}")
    print(f"  Rows read: {total_rows_read:,}")
    print(f"  Unique (nhop, 1hop) pairs: {total_rows:,}")
    print(f"  Compression ratio: {total_rows_read/total_rows if total_rows > 0 else 1:.2f}x")
    print(f"  Output file: {output_file}")

    if missing_indices:
        print(f"\n⚠ WARNING: {len(missing_indices)} result files were missing")
        print(f"  Check manifest.json for failed jobs")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Merge individual matrix1 result files into a single output'
    )
    parser.add_argument('--n-hops', type=int, default=3,
                        help='Number of hops analyzed (default: 3)')
    parser.add_argument('--results-dir', default=None,
                        help='Directory containing result files (default: results_{n_hops}hop)')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: results_dir/all_{n_hops}hop_overlaps.tsv)')

    args = parser.parse_args()

    merge_results(results_dir=args.results_dir, output_file=args.output, n_hops=args.n_hops)


if __name__ == "__main__":
    main()
