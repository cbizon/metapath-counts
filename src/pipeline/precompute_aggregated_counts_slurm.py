#!/usr/bin/env python3
"""
Submit SLURM jobs to precompute explicit path counts for grouping.

Pass A: Scan result files to find first-seen explicit counts per path.
Reduce A: Merge explicit counts and shard them by ancestor endpoint type pair.
"""

import argparse
import glob
import math
import os
import subprocess


WORKER_PASSA = "src/pipeline/workers/run_prepare_grouping_passA.sh"
WORKER_REDUCEA = "src/pipeline/workers/run_prepare_grouping_reduceA.sh"
def parse_args():
    parser = argparse.ArgumentParser(description="Precompute aggregated counts via SLURM")
    parser.add_argument("--n-hops", type=int, required=True, help="Number of hops (e.g. 3)")
    parser.add_argument("--results-dir", type=str, default=None, help="Results dir (default: results_{n}hop)")
    parser.add_argument("--tmp-dir", type=str, default=None, help="Temp dir (default: results_{n}hop/_tmp_prepare_grouping)")
    parser.add_argument("--partition", type=str, default="lowpri", help="SLURM partition (default: lowpri)")
    parser.add_argument("--passa-files-per-shard", type=int, default=40, help="Files per Pass A shard")
    parser.add_argument("--passa-max-concurrent", type=int, default=100, help="Max concurrent Pass A array tasks")
    parser.add_argument("--mem-passa", type=int, default=64, help="Memory (GB) for Pass A tasks")
    parser.add_argument("--mem-reducea", type=int, default=128, help="Memory (GB) for Reduce A")
    return parser.parse_args()


def submit_sbatch(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    output = result.stdout.strip()
    if "Submitted batch job" not in output:
        raise RuntimeError(f"Unexpected sbatch output: {output}")
    return output.split()[-1]


def write_files_list(results_dir, tmp_dir):
    pattern = os.path.join(results_dir, "results_matrix1_*.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result files found at {pattern}")

    os.makedirs(tmp_dir, exist_ok=True)
    files_list_path = os.path.join(tmp_dir, "files.txt")
    with open(files_list_path, "w") as f:
        f.write("\n".join(files))
        f.write("\n")
    return files_list_path, len(files)


def main():
    args = parse_args()
    results_dir = args.results_dir or f"results_{args.n_hops}hop"
    tmp_dir = args.tmp_dir or os.path.join(results_dir, "_tmp_prepare_grouping")

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    files_list_path, total_files = write_files_list(results_dir, tmp_dir)
    passa_shards = max(1, math.ceil(total_files / args.passa_files_per_shard))

    logs_dir = f"logs_grouping_{args.n_hops}hop"
    os.makedirs(logs_dir, exist_ok=True)

    # Pass A array
    array_spec_a = f"0-{passa_shards - 1}%{args.passa_max_concurrent}"
    cmd_a = [
        "sbatch",
        f"--partition={args.partition}",
        f"--mem={args.mem_passa}G",
        f"--array={array_spec_a}",
        "--job-name=prepA_map",
        f"--output={logs_dir}/prepA_%a.out",
        f"--error={logs_dir}/prepA_%a.err",
        WORKER_PASSA,
        files_list_path,
        str(passa_shards),
        tmp_dir,
    ]
    job_a = submit_sbatch(cmd_a)

    # Reduce A
    cmd_ra = [
        "sbatch",
        f"--partition={args.partition}",
        f"--mem={args.mem_reducea}G",
        f"--dependency=afterok:{job_a}",
        "--job-name=prepA_reduce",
        f"--output={logs_dir}/prepA_reduce.out",
        f"--error={logs_dir}/prepA_reduce.err",
        WORKER_REDUCEA,
        tmp_dir,
        results_dir,
        str(args.n_hops),
    ]
    job_ra = submit_sbatch(cmd_ra)

    print("Submitted prepare_grouping explicit-count precompute jobs:")
    print(f"  Pass A array:   {job_a} (shards: {passa_shards})")
    print(f"  Reduce A:       {job_ra}")
    print()
    print("Next step after completion:")
    print(f"  uv run python src/pipeline/prepare_grouping.py --n-hops {args.n_hops}")


if __name__ == "__main__":
    main()
