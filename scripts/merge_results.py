#!/usr/bin/env python3
"""
Merge individual matrix1 result files into a single output file.

This script:
1. Finds all results_matrix1_*.tsv files
2. Verifies all expected indices are present
3. Merges them into a single TSV with one header
4. Reports statistics

Usage:
    uv run python scripts/metapaths/merge_results.py [--output OUTPUT_FILE]
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


def merge_results(results_dir="scripts/metapaths/results", output_file=None):
    """Merge all result files into a single output file."""
    print("=" * 80)
    print("MERGING 3-HOP ANALYSIS RESULTS")
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
        output_file = os.path.join(results_dir, "all_3hop_overlaps.tsv")

    print(f"\nMerging {len(file_by_index)} files into: {output_file}")

    # Merge files
    total_rows = 0
    files_processed = 0

    with open(output_file, 'w') as out:
        # Write header from first file
        first_file = file_by_index[min(file_by_index.keys())]
        with open(first_file, 'r') as f:
            header = f.readline()
            out.write(header)

        # Append data from all files (in index order)
        for idx in sorted(file_by_index.keys()):
            file_path = file_by_index[idx]
            files_processed += 1

            with open(file_path, 'r') as f:
                # Skip header
                f.readline()

                # Copy remaining lines
                lines_from_file = 0
                for line in f:
                    out.write(line)
                    lines_from_file += 1
                    total_rows += 1

            if (files_processed % 50 == 0) or (files_processed == len(file_by_index)):
                print(f"  Processed {files_processed}/{len(file_by_index)} files ({total_rows:,} rows so far)")

    print(f"\n{'=' * 80}")
    print("MERGE COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nStatistics:")
    print(f"  Files merged: {files_processed}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Output file: {output_file}")

    if missing_indices:
        print(f"\n⚠ WARNING: {len(missing_indices)} result files were missing")
        print(f"  Check manifest.json for failed jobs")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Merge individual matrix1 result files into a single output'
    )
    parser.add_argument('--results-dir', default='scripts/metapaths/results',
                        help='Directory containing result files (default: scripts/metapaths/results)')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: results_dir/all_3hop_overlaps.tsv)')

    args = parser.parse_args()

    merge_results(results_dir=args.results_dir, output_file=args.output)


if __name__ == "__main__":
    main()
