#!/usr/bin/env python3
"""
Submit SLURM jobs to precompute aggregated N-hop counts in two passes.

Pass A: Scan result files to find first-seen explicit counts per path.
Pass B: Expand explicit counts to hierarchical variants and aggregate.
"""

import argparse
import glob
import math
import os
import subprocess


WORKER_PASSA = "src/pipeline/workers/run_prepare_grouping_passA.sh"
WORKER_REDUCEA = "src/pipeline/workers/run_prepare_grouping_reduceA.sh"
WORKER_PASSB = "src/pipeline/workers/run_prepare_grouping_passB.sh"
WORKER_REDUCEB = "src/pipeline/workers/run_prepare_grouping_reduceB.sh"


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute aggregated counts via SLURM")
    parser.add_argument("--n-hops", type=int, required=True, help="Number of hops (e.g. 3)")
    parser.add_argument("--results-dir", type=str, default=None, help="Results dir (default: results_{n}hop)")
    parser.add_argument("--tmp-dir", type=str, default=None, help="Temp dir (default: results_{n}hop/_tmp_prepare_grouping)")
    parser.add_argument("--partition", type=str, default="lowpri", help="SLURM partition (default: lowpri)")
    parser.add_argument("--passa-files-per-shard", type=int, default=40, help="Files per Pass A shard")
    parser.add_argument("--passa-max-concurrent", type=int, default=100, help="Max concurrent Pass A array tasks")
    parser.add_argument("--passb-shards", type=int, default=512, help="Shard count for Pass B")
    parser.add_argument("--passb-max-concurrent", type=int, default=200, help="Max concurrent Pass B array tasks")
    parser.add_argument("--mem-passa", type=int, default=64, help="Memory (GB) for Pass A tasks")
    parser.add_argument("--mem-reducea", type=int, default=128, help="Memory (GB) for Reduce A")
    parser.add_argument("--mem-passb", type=int, default=64, help="Memory (GB) for Pass B tasks")
    parser.add_argument("--mem-reduceb", type=int, default=250, help="Memory (GB) for Reduce B")
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
        str(args.passb_shards),
        results_dir,
        str(args.n_hops),
    ]
    job_ra = submit_sbatch(cmd_ra)

    # Pass B array
    array_spec_b = f"0-{args.passb_shards - 1}%{args.passb_max_concurrent}"
    cmd_b = [
        "sbatch",
        f"--partition={args.partition}",
        f"--mem={args.mem_passb}G",
        f"--array={array_spec_b}",
        f"--dependency=afterok:{job_ra}",
        "--job-name=prepB_map",
        f"--output={logs_dir}/prepB_%a.out",
        f"--error={logs_dir}/prepB_%a.err",
        WORKER_PASSB,
        tmp_dir,
    ]
    job_b = submit_sbatch(cmd_b)

    # Reduce B
    cmd_rb = [
        "sbatch",
        f"--partition={args.partition}",
        f"--mem={args.mem_reduceb}G",
        f"--dependency=afterok:{job_b}",
        "--job-name=prepB_reduce",
        f"--output={logs_dir}/prepB_reduce.out",
        f"--error={logs_dir}/prepB_reduce.err",
        WORKER_REDUCEB,
        tmp_dir,
        results_dir,
        str(args.n_hops),
    ]
    job_rb = submit_sbatch(cmd_rb)

    print("Submitted prepare_grouping precompute jobs:")
    print(f"  Pass A array:   {job_a} (shards: {passa_shards})")
    print(f"  Reduce A:       {job_ra}")
    print(f"  Pass B array:   {job_b} (shards: {args.passb_shards})")
    print(f"  Reduce B:       {job_rb}")
    print()
    print("Next step after completion:")
    print(f"  uv run python src/pipeline/prepare_grouping.py --n-hops {args.n_hops} --skip-aggregated-precompute")


if __name__ == "__main__":
    main()
