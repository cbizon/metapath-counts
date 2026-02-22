#!/usr/bin/env python3
"""
Pass A map worker: scan a shard of result files and record first-seen counts
for each explicit path (predictor + predicted).
"""

import argparse
import math
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare grouping Pass A map worker")
    parser.add_argument("--files-list", required=True, help="Path to sorted file list")
    parser.add_argument("--shard-index", type=int, required=True, help="Shard index")
    parser.add_argument("--shard-count", type=int, required=True, help="Total shard count")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard files")
    return parser.parse_args()


def load_files_list(path):
    with open(path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def compute_range(total_files, shard_index, shard_count):
    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index out of range")
    chunk = math.ceil(total_files / shard_count)
    start = shard_index * chunk
    end = min(start + chunk, total_files)
    return start, end


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = load_files_list(args.files_list)
    total_files = len(files)
    start, end = compute_range(total_files, args.shard_index, args.shard_count)

    first_seen = {}
    for file_index in range(start, end):
        file_path = files[file_index]
        with open(file_path, "r") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                nhop_path = parts[0]
                nhop_count = int(parts[1])
                onehop_path = parts[2]
                onehop_count = int(parts[3])

                if nhop_path not in first_seen:
                    first_seen[nhop_path] = (file_index, nhop_count)
                if onehop_path not in first_seen:
                    first_seen[onehop_path] = (file_index, onehop_count)

    output_path = os.path.join(args.output_dir, f"first_seen_{args.shard_index:05d}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(first_seen, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
