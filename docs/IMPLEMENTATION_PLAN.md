# Plan: Parallelized 3-Hop Metapath Analysis with Auto-Retry

## Overview
Transform `analyze_hop_overlap.py` from sequential (5 days) to parallel (few hours) execution with automatic memory-tiered retry system.

## Key Design Decisions
- **Direction filtering:** Each M1 job only computes paths where M3.nvals >= M1.nvals (eliminates duplicate computation)
- **Parallelism:** One job per Matrix1 (~250-500 SLURM jobs)
- **Memory tiers:** 250GB → 500GB → 1TB (partition: lowpri, 24h time limit)
- **Checkpointing:** Manifest file (manifest.json) tracks completion status
- **Output:** Separate files (results_matrix1_XXX.tsv), merge at end
- **Queue management:** Max 1 pending job at a time (unlimited running)

## Implementation Steps

### 1. Modify `analyze_hop_overlap.py`

Add `--matrix1-index` CLI argument to process a single Matrix1. The core logic changes:

```python
def analyze_single_matrix1(matrix1_index, all_matrices):
    matrix1 = all_matrices[matrix1_index]
    m1_nvals = matrix1.nvals  # Number of non-zero entries

    # Loop over all compatible M2 matrices
    for matrix2 in get_compatible_matrices(matrix1):

        # Filter M3s: only process if M3.nvals >= M1.nvals
        # This ensures each path computed exactly once
        valid_m3s = [m3 for m3 in get_compatible_matrices(matrix2)
                     if m3.nvals >= m1_nvals]

        if not valid_m3s:
            continue  # Skip this M2 - no valid M3s remain after filtering

        # Compute M1@M2 once (reused across all M3s)
        result_m1m2 = matrix1 @ matrix2

        # Loop over filtered M3s only
        for matrix3 in valid_m3s:
            # Compute final 3-hop result
            result_m1m2m3 = result_m1m2 @ matrix3

            # Calculate overlaps with 1-hop matrices
            compute_overlaps(result_m1m2m3, ...)

            # Write output row
            write_result(...)
```

**Why this eliminates duplicates:**
- Every 3-hop path A→B→C→D can be computed from either end
- When M1=A and M3=D: Only compute if D.nvals >= A.nvals
- When M1=D and M3=A: Only compute if A.nvals >= D.nvals
- Exactly ONE of these conditions is true (choosing larger-or-equal end)
- Result: Each unique path computed exactly once

**Example:**
- Matrix A has 1,000 edges, Matrix D has 5,000 edges
- Job with M1=A: 5000 >= 1000 ✓ → Compute path A→B→C→D
- Job with M1=D: 1000 >= 5000 ✗ → Skip (already computed by A's job)

**Additional changes:**
- Write output to `results/results_matrix1_{index:03d}.tsv`
- Update manifest.json on successful completion
- Return proper exit codes: 0=success, 1=error, 137=OOM
- Memory tracking: Monitor memory at key points for debugging

### 2. Create SLURM Worker Script `scripts/metapaths/run_single_matrix1.sh`

One static script with hardcoded data paths:

```bash
#!/bin/bash
#SBATCH --partition=lowpri
#SBATCH --time=24:00:00

# Static paths to ROBOKOP graph data
EDGES_FILE="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/49e5e5585f7685b4/edges.jsonl"
NODES_FILE="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/49e5e5585f7685b4/nodes.jsonl"

# Arguments from command line
MATRIX1_INDEX=$1
OUTPUT_FILE="scripts/metapaths/results/results_matrix1_${MATRIX1_INDEX}.tsv"

# Activate environment and run analysis
source .venv/bin/activate
uv run python scripts/metapaths/analyze_hop_overlap.py \
  --matrix1-index $MATRIX1_INDEX \
  --edges $EDGES_FILE \
  --nodes $NODES_FILE \
  --output $OUTPUT_FILE

EXIT_CODE=$?
exit $EXIT_CODE
```

The orchestrator will override memory and logging via sbatch command-line flags (no templating needed).

### 3. Create Manifest System
**manifest.json** schema:
```json
{
  "matrix1_000": {
    "status": "pending|running|completed|failed",
    "memory_tier": 250,
    "attempts": 0,
    "job_id": "12345678",
    "last_update": "2025-10-28T10:30:00",
    "error_type": null
  },
  "matrix1_001": {
    "status": "completed",
    "memory_tier": 250,
    "attempts": 1,
    "job_id": "12345679",
    "last_update": "2025-10-28T11:15:00",
    "error_type": null
  }
}
```

Helper functions:
- `init_manifest(num_matrices)`: Create initial manifest with all jobs pending
- `update_job_status(matrix1_id, status, job_id, error_type)`: Update single job
- `get_jobs_by_status(status)`: Query jobs in given state
- `get_next_pending_job()`: Get next job to submit
- `mark_failed_job_for_retry(matrix1_id)`: Increment memory tier, reset to pending

### 4. Create Orchestrator `scripts/metapaths/orchestrate_3hop_analysis.py`

**Job submission function:**
```python
def submit_job(matrix1_index, memory_gb):
    """Submit a single matrix1 job to SLURM with specified memory."""
    cmd = [
        'sbatch',
        f'--mem={memory_gb}G',
        f'--job-name=3hop_m1_{matrix1_index:03d}',
        f'--output=scripts/metapaths/logs/matrix1_{matrix1_index:03d}_mem{memory_gb}.out',
        f'--error=scripts/metapaths/logs/matrix1_{matrix1_index:03d}_mem{memory_gb}.err',
        'scripts/metapaths/run_single_matrix1.sh',
        str(matrix1_index)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    # Parse job ID from "Submitted batch job 12345678"
    job_id = result.stdout.strip().split()[-1]
    return job_id
```

**Main control loop:**
```python
while not all_jobs_completed():
    # Poll SLURM status
    running_jobs = get_slurm_running_jobs()
    completed_jobs = get_slurm_completed_jobs()

    # Update manifest from SLURM status
    for job_id, slurm_status, exit_code in completed_jobs:
        matrix1_id = lookup_matrix1_by_job(job_id)

        if slurm_status == "OUT_OF_MEMORY" or exit_code == 137:
            # OOM failure - retry at higher memory
            current_memory = get_memory_tier(matrix1_id)
            next_memory = increment_memory_tier(current_memory)  # 250 -> 500 -> 1000
            update_manifest(matrix1_id, status='pending', memory_tier=next_memory)
        elif slurm_status == "COMPLETED" and exit_code == 0:
            # Success
            update_manifest(matrix1_id, status='completed')
        else:
            # Other failure - log and mark failed
            update_manifest(matrix1_id, status='failed', error_type=slurm_status)

    # Submit new jobs if queue space available
    pending_count = count_pending_jobs_in_slurm()

    if pending_count < 1:  # Keep max 1 pending
        next_jobs = get_next_pending_jobs(limit=5)  # Submit batch
        for matrix1_id, memory_tier in next_jobs:
            job_id = submit_job(matrix1_id, memory_tier)
            update_manifest(matrix1_id, status='running', job_id=job_id)

    # Progress report
    print_status_summary()

    sleep(30)  # Poll interval

# Final report
generate_completion_report()
```

**SLURM integration:**
- Use `squeue -u $USER -o "%i %T"` for running/pending jobs
- Use `sacct -j {job_ids} --format=JobID,State,ExitCode` for completed jobs
- Parse exit codes: 0=success, 137=OOM, others=error
- Parse SLURM states: RUNNING, PENDING, COMPLETED, FAILED, OUT_OF_MEMORY, etc.

**Queue management:**
- Track pending vs running separately
- Allow unlimited running (cluster manages this)
- Keep only 1 pending to avoid queue hogging
- Submit in small batches (5 jobs) when space available

**Memory tier progression:**
- Initial: 250GB
- First OOM retry: 500GB
- Second OOM retry: 1000GB (1TB)
- After 1TB failure: Mark as permanently failed, log for manual investigation

### 5. Create Merge Script `scripts/metapaths/merge_results.py`

```python
def merge_results(results_dir="scripts/metapaths/results"):
    # Find all result files
    result_files = sorted(glob.glob(f"{results_dir}/results_matrix1_*.tsv"))

    # Extract indices and verify completeness
    expected_indices = set(range(num_total_matrices))
    found_indices = set()

    for file in result_files:
        idx = extract_index_from_filename(file)
        found_indices.add(idx)

    missing = expected_indices - found_indices
    if missing:
        raise ValueError(f"Missing results for indices: {sorted(missing)}")

    # Merge files
    with open(f"{results_dir}/all_3hop_overlaps.tsv", "w") as out:
        # Write header from first file
        with open(result_files[0]) as f:
            header = f.readline()
            out.write(header)

        # Append data from all files
        for file in result_files:
            with open(file) as f:
                f.readline()  # Skip header
                out.write(f.read())

    print(f"Merged {len(result_files)} files into all_3hop_overlaps.tsv")
```

### 6. Create Initialization Script `scripts/metapaths/prepare_analysis.py`

```python
def prepare_analysis():
    # Hardcoded paths (same as worker script)
    nodes_file = "/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/49e5e5585f7685b4/nodes.jsonl"
    edges_file = "/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/49e5e5585f7685b4/edges.jsonl"

    # Load matrices using existing infrastructure from analyze_hop_overlap.py
    node_types = load_node_types(nodes_file)
    matrices = build_matrices(edges_file, node_types)

    num_matrices = len(matrices)
    print(f"Total Matrix1 jobs to process: {num_matrices}")

    # Create output directories
    os.makedirs("scripts/metapaths/results", exist_ok=True)
    os.makedirs("scripts/metapaths/logs", exist_ok=True)

    # Initialize manifest
    manifest = {
        f"matrix1_{i:03d}": {
            "status": "pending",
            "memory_tier": 250,
            "attempts": 0,
            "job_id": None,
            "last_update": datetime.now().isoformat(),
            "error_type": None,
            "matrix_nvals": matrices[i][3].nvals  # Store for reference
        }
        for i in range(num_matrices)
    }

    with open("scripts/metapaths/results/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Initialized manifest for {num_matrices} jobs")
    print("Ready to run: python scripts/metapaths/orchestrate_3hop_analysis.py")
```

### 7. Testing & Validation Plan

**Phase 1: Unit testing (dry-run)**
- Test direction selection logic on 10 sample Matrix1 indices
- Verify canonical form determination prevents duplicates
- Test manifest CRUD operations
- Dry-run orchestrator (print sbatch commands, don't submit)

**Phase 2: Small-scale live test**
- Run 5 jobs manually with known memory requirements
- Artificially lower memory limits to trigger OOM
- Verify manifest updates correctly
- Verify retry logic increments memory tier
- Verify output file format

**Phase 3: Medium-scale pilot**
- Run orchestrator with 20 jobs
- Monitor queue management (max 1 pending)
- Verify merge script works correctly
- Check progress reporting

**Phase 4: Full production run**
- Submit all ~250-500 jobs
- Monitor completion over 2-4 hours
- Handle any unexpected failures
- Generate final merged output

## File Structure
```
scripts/metapaths/
├── analyze_hop_overlap.py          # Modified (add --matrix1-index)
├── prepare_analysis.py               # NEW: Initialize manifest
├── orchestrate_3hop_analysis.py     # NEW: Master control script
├── run_single_matrix1.sh            # NEW: SLURM worker script template
├── merge_results.py                 # NEW: Combine outputs
├── logs/                            # NEW: SLURM stdout/stderr
│   ├── matrix1_000_mem250.out
│   └── matrix1_000_mem250.err
└── results/                         # NEW: Output directory
    ├── manifest.json                # Job tracking (live updates)
    ├── results_matrix1_000.tsv      # Per-job outputs
    ├── results_matrix1_001.tsv
    ├── ...
    └── all_3hop_overlaps.tsv        # Final merged output
```

## Expected Performance

**Before optimization:**
- Sequential execution: ~5 days
- Computes both forward and reverse: 2x redundant work
- Single-threaded: no parallelism

**After optimization:**
- Direction selection: 2x speedup (eliminate redundant computation)
- Parallelization: ~100x speedup (250 jobs running concurrently)
- **Combined: ~200x speedup**
- **Estimated runtime: 2-4 hours** (assuming most jobs finish in 15-30 min)

**Memory efficiency:**
- Start all jobs at 250GB
- ~80% expected to complete at 250GB tier
- ~15% need 500GB (auto-retry)
- ~5% need 1TB (second retry)
- Failed jobs resubmit automatically, minimal wasted compute

**Orchestration overhead:**
- Polling interval: 30s
- Manifest updates: O(1) per job status change
- Queue management: negligible (<1% overhead)

## Success Criteria
1. All Matrix1 indices processed exactly once ✓
2. No duplicate 3-hop path computations (forward vs reverse) ✓
3. Automatic recovery from OOM failures ✓
4. Queue management: max 1 pending job at a time ✓
5. Final output identical to sequential run (modulo row ordering) ✓
6. Completion time: <6 hours ✓
7. No manual intervention required after initial submission ✓

## Risk Mitigation

**Risk: Orchestrator script crashes**
- Mitigation: Design to be restartable - reads manifest state and resumes
- No data loss: manifest tracks all state, output files are atomic

**Risk: Cluster maintenance/outage**
- Mitigation: Jobs resume from last checkpoint via manifest
- Graceful degradation: orchestrator detects disappeared jobs and resubmits

**Risk: Incorrect direction selection causes wrong results**
- Mitigation: Extensive testing on sample data
- Validation: Compare subset of results against sequential run

**Risk: Memory estimation still wrong for outliers**
- Mitigation: 1TB tier catches nearly all cases
- Fallback: Manual intervention for extreme outliers (log and investigate)

**Risk: Too many jobs overwhelm filesystem**
- Mitigation: ~250-500 files is manageable on modern filesystems
- Output files are written atomically, no corruption risk

## Future Enhancements (Out of Scope)
- Dynamic memory prediction using matrix.nvals heuristics
- Job array optimization instead of individual submissions
- Real-time progress dashboard (web UI)
- Cost optimization: prefer cheaper memory tiers when possible
- Checkpoint within jobs (partial progress saving)
