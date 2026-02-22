#!/usr/bin/env python3
"""
Compare two grouped results directories for equality.

Checks:
1) File set equality (tsv files)
2) Header line equality
3) Row count equality
4) Content equality ignoring order (sorted line comparison)
"""

import argparse
import hashlib
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare grouped results directories")
    parser.add_argument("--left", required=True, help="Reference grouped results dir")
    parser.add_argument("--right", required=True, help="Rerun grouped results dir")
    parser.add_argument("--sample-diff", type=int, default=0,
                        help="Show up to N differing rows per file (order-insensitive)")
    return parser.parse_args()


def list_tsv_files(root):
    root_path = Path(root)
    return sorted([p for p in root_path.rglob("*.tsv") if p.is_file()])


def relpath(path, root):
    return str(Path(path).relative_to(root))


def read_lines(path):
    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]
    return lines


def hash_lines(lines):
    h = hashlib.blake2b()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compare_sorted_lines(left_lines, right_lines, sample_diff):
    left_sorted = sorted(left_lines)
    right_sorted = sorted(right_lines)
    if left_sorted == right_sorted:
        return True, []

    diffs = []
    if sample_diff > 0:
        i = j = 0
        while i < len(left_sorted) and j < len(right_sorted) and len(diffs) < sample_diff:
            if left_sorted[i] == right_sorted[j]:
                i += 1
                j += 1
                continue
            if left_sorted[i] < right_sorted[j]:
                diffs.append(f"- {left_sorted[i]}")
                i += 1
            else:
                diffs.append(f"+ {right_sorted[j]}")
                j += 1
        while i < len(left_sorted) and len(diffs) < sample_diff:
            diffs.append(f"- {left_sorted[i]}")
            i += 1
        while j < len(right_sorted) and len(diffs) < sample_diff:
            diffs.append(f"+ {right_sorted[j]}")
            j += 1

    return False, diffs


def main():
    args = parse_args()
    left_root = Path(args.left)
    right_root = Path(args.right)

    if not left_root.exists():
        raise FileNotFoundError(f"Left dir not found: {left_root}")
    if not right_root.exists():
        raise FileNotFoundError(f"Right dir not found: {right_root}")

    left_files = list_tsv_files(left_root)
    right_files = list_tsv_files(right_root)

    left_rel = {relpath(p, left_root) for p in left_files}
    right_rel = {relpath(p, right_root) for p in right_files}

    missing = sorted(left_rel - right_rel)
    extra = sorted(right_rel - left_rel)

    if missing:
        print(f"Missing in right: {len(missing)} files")
        for p in missing[:10]:
            print(f"  - {p}")
        if len(missing) > 10:
            print("  ...")
    if extra:
        print(f"Extra in right: {len(extra)} files")
        for p in extra[:10]:
            print(f"  + {p}")
        if len(extra) > 10:
            print("  ...")

    common = sorted(left_rel & right_rel)
    mismatched = 0

    for rel in common:
        left_path = left_root / rel
        right_path = right_root / rel

        left_lines = read_lines(left_path)
        right_lines = read_lines(right_path)

        if not left_lines or not right_lines:
            if left_lines != right_lines:
                print(f"Mismatch (empty file): {rel}")
                mismatched += 1
            continue

        left_header = left_lines[0]
        right_header = right_lines[0]
        if left_header != right_header:
            print(f"Header mismatch: {rel}")
            print(f"  left:  {left_header}")
            print(f"  right: {right_header}")
            mismatched += 1
            continue

        left_body = left_lines[1:]
        right_body = right_lines[1:]

        if len(left_body) != len(right_body):
            print(f"Row count mismatch: {rel} (left {len(left_body)}, right {len(right_body)})")
            mismatched += 1
            continue

        # Order-insensitive comparison
        same, diffs = compare_sorted_lines(left_body, right_body, args.sample_diff)
        if not same:
            print(f"Content mismatch: {rel} (rows {len(left_body)})")
            if diffs:
                for d in diffs:
                    print(f"  {d}")
            else:
                # Fall back to hash info for quick diagnosis
                left_hash = hash_lines(sorted(left_body))
                right_hash = hash_lines(sorted(right_body))
                print(f"  left hash:  {left_hash}")
                print(f"  right hash: {right_hash}")
            mismatched += 1

    if missing or extra or mismatched:
        print("\nComparison failed.")
        print(f"  Missing: {len(missing)}")
        print(f"  Extra: {len(extra)}")
        print(f"  Mismatched: {mismatched}")
        raise SystemExit(1)

    print("Comparison OK: all files match.")


if __name__ == "__main__":
    main()
