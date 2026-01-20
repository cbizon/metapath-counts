# Parallel 3-Hop Metapath Analysis System

This system parallelizes 3-hop metapath analysis across SLURM cluster with automatic memory-tiered retry.

## Quick Start

```bash
# 1. Initialize (if not already done)
uv run python scripts/metapaths/prepare_analysis.py

# 2. Run orchestrator (handles everything automatically)
uv run python scripts/metapaths/orchestrate_3hop_analysis.py

# 3. After completion, merge results
uv run python scripts/metapaths/merge_results.py

# 4. Group results by 1-hop metapath with metrics
uv run python scripts/metapaths/group_by_onehop.py
```

## System Overview

### What This System Does

1. **Splits work** across ~2,879 independent jobs (one per Matrix1)
2. **Prevents node revisiting** via three-level diagonal zeroing (no A→B→A→C paths)
3. **Eliminates duplicates** via M3 filtering (M3.nvals >= M1.nvals rule)
4. **Auto-retries OOM failures** at higher memory tiers: 180GB → 250GB → 500GB → 1TB → 1.5TB
5. **Manages queue** to avoid overwhelming cluster (max 1 pending job)
6. **Tracks progress** via manifest.json
7. **Merges and groups results** for analysis

### Expected Performance

- **Sequential (old)**: ~5 days
- **Parallel (new)**: ~2-4 hours with ~100-200 concurrent jobs
- **Speedup**: ~50-100x

## Step-by-Step Instructions

### Step 1: Preparation (One-Time Setup)

Initialize manifest and directories:

```bash
cd /projects/sequence_analysis/vol3/bizon/sub/pathfilter
uv run python scripts/metapaths/prepare_analysis.py
```

**Output:**
- Creates `scripts/metapaths/results/` directory
- Creates `scripts/metapaths/logs/` directory
- Generates `scripts/metapaths/results/manifest.json` with all 2,879 jobs

**Time:** ~3-4 minutes (loads entire graph to count matrices)

**Note:** Only run this once. If you need to restart, the orchestrator uses the existing manifest.

### Step 2: Run Orchestrator

Submit and monitor all jobs automatically:

```bash
uv run python scripts/metapaths/orchestrate_3hop_analysis.py
```

**What it does:**
- Reads manifest.json
- Submits jobs in batches (maintains max 1 pending)
- Polls SLURM every 30 seconds
- Detects OOM failures and retries at higher memory
- Updates manifest in real-time
- Prints progress summary
- Exits when all jobs complete or fail

**Time:** ~2-4 hours (depends on cluster load and job complexity)

**How to run:**
- Option A: Run in foreground (monitor progress)
  ```bash
  uv run python scripts/metapaths/orchestrate_3hop_analysis.py
  ```

- Option B: Run in background with logs
  ```bash
  nohup uv run python scripts/metapaths/orchestrate_3hop_analysis.py > orchestrator.log 2>&1 &
  tail -f orchestrator.log
  ```

- Option C: Run in screen/tmux session (recommended)
  ```bash
  screen -S metapath_orchestrator
  uv run python scripts/metapaths/orchestrate_3hop_analysis.py
  # Detach: Ctrl+A, then D
  # Reattach: screen -r metapath_orchestrator
  ```

**To stop orchestrator:**
- Ctrl+C (graceful shutdown)
- Jobs will continue running in SLURM
- Restart orchestrator anytime - it resumes from manifest

### Step 3: Merge Results

After all jobs complete, combine outputs:

```bash
uv run python scripts/metapaths/merge_results.py
```

**Output:**
- `scripts/metapaths/results/all_3hop_overlaps.tsv` (final merged file)
- Verifies all expected result files are present
- Reports statistics (total rows, missing files, etc.)

**Time:** ~1-2 minutes

### Step 4: Group Results by 1-Hop Metapath

After merging, group the 3-hop results by their corresponding normalized 1-hop metapaths:

```bash
uv run python scripts/metapaths/group_by_onehop.py
```

**What it does:**
- Reads all result files from `scripts/metapaths/results/`
- Normalizes 1-hop metapaths to forward (F) direction
- Reverses 3-hop paths when needed to match normalization
- Calculates performance metrics (Precision, Recall, F1, MCC, etc.)
- Groups results by normalized 1-hop metapath
- Writes one TSV file per unique 1-hop metapath

**Output:**
- `scripts/metapaths/grouped_by_1hop/*.tsv` (one file per 1-hop metapath)
- Example: `Drug_has_adverse_event_F_Disease.tsv` contains all 3-hop paths that predict that 1-hop

**Options:**
```bash
# Use custom file handle limit (default: 500)
uv run python scripts/metapaths/group_by_onehop.py --max-open-files 1000

# Run in test mode (uses test_results directory)
uv run python scripts/metapaths/group_by_onehop.py --test
```

**Time:** ~10-20 minutes for full dataset (45M lines)

**Output columns:**
- Original: `3hop_metapath`, `3hop_count`, `1hop_metapath`, `1hop_count`, `overlap`, `total_possible`
- Added metrics: `TP`, `FP`, `FN`, `TN`, `Total`, `Precision`, `Recall`, `Specificity`, `NPV`, `Accuracy`, `Balanced_Accuracy`, `F1`, `MCC`, `TPR`, `FPR`, `FNR`, `PLR`, `NLR`

## Complete Rerun from Scratch

If you need to regenerate all results (e.g., after code changes like diagonal zeroing), follow these steps:

### Step 1: Clean Old Results

```bash
# WARNING: This deletes ~22 GB of data
# Optional: backup first if you want to compare old vs new results
mkdir -p scripts/metapaths/old_results_backup
mv scripts/metapaths/results scripts/metapaths/old_results_backup/results_$(date +%Y%m%d)
mv scripts/metapaths/logs scripts/metapaths/old_results_backup/logs_$(date +%Y%m%d)
mv scripts/metapaths/grouped_by_1hop scripts/metapaths/old_results_backup/grouped_$(date +%Y%m%d)

# Clean out old results
rm -rf scripts/metapaths/results/*
rm -rf scripts/metapaths/logs/*
rm -rf scripts/metapaths/grouped_by_1hop/*

# Verify empty
ls scripts/metapaths/results/ | wc -l      # Should show 0
ls scripts/metapaths/logs/ | wc -l         # Should show 0
ls scripts/metapaths/grouped_by_1hop/ | wc -l  # Should show 0
```

### Step 2: Regenerate Manifest

```bash
uv run python scripts/metapaths/prepare_analysis.py
```

This creates a fresh `manifest.json` with all 2,879 jobs set to "pending".

**Time:** ~3-4 minutes (loads entire graph)

### Step 3: Run Orchestrator

```bash
# Recommended: run in screen/tmux
screen -S metapath_orchestrator
uv run python scripts/metapaths/orchestrate_3hop_analysis.py
# Detach: Ctrl+A, then D
```

**Time:** ~2-4 hours

### Step 4: Merge Results

```bash
uv run python scripts/metapaths/merge_results.py
```

### Step 5: Group Results

```bash
uv run python scripts/metapaths/group_by_onehop.py
```

**Total time:** ~3-5 hours (mostly cluster compute time)

## Monitoring Progress

### Check Orchestrator Status

While orchestrator is running, you'll see updates every 30 seconds:

```
================================================================================
STATUS: 1523/2879 completed (52.9%) | Running: 127 | Pending: 1 | Failed: 3
SLURM queue: 127 running jobs
================================================================================
```

### Check SLURM Queue

```bash
# Count running jobs
squeue -u $USER | grep 3hop | wc -l

# View all jobs
squeue -u $USER | grep 3hop

# Check specific job
squeue -j <job_id>
```

### Check Manifest

```bash
# Count completed jobs
jq '[.[] | select(.status == "completed")] | length' scripts/metapaths/results/manifest.json

# Count failed jobs
jq '[.[] | select(.status == "failed")] | length' scripts/metapaths/results/manifest.json

# List failed jobs with errors
jq -r '.[] | select(.status == "failed") | "\(.)|error:\(.error_type)"' \
  scripts/metapaths/results/manifest.json

# Find jobs that needed high memory
jq -r '.[] | select(.memory_tier > 250) | "\(.): \(.memory_tier)GB"' \
  scripts/metapaths/results/manifest.json
```

### Check Individual Job Logs

SLURM output files are in the current directory (e.g., `slurm-32368.out`):

```bash
# List recent log files
ls -lth slurm-*.out | head -20

# View specific job output
tail -f slurm-32368.out

# Search for errors
grep -i error slurm-*.out

# Check memory usage in logs
grep "Mem:" slurm-32368.out
```

### Check Result Files

```bash
# Count generated result files
ls scripts/metapaths/results/results_matrix1_*.tsv | wc -l

# List largest result files
ls -lhS scripts/metapaths/results/results_matrix1_*.tsv | head -10

# Check specific result
head scripts/metapaths/results/results_matrix1_042.tsv
wc -l scripts/metapaths/results/results_matrix1_042.tsv
```

## Troubleshooting

### Problem: Orchestrator not submitting jobs

**Symptoms:** STATUS shows pending jobs but nothing submitted

**Causes:**
- SLURM queue already has 1 pending job (by design)
- Check: `squeue -u $USER -t PENDING | grep 3hop`

**Solution:** Wait for pending job to start running (orchestrator will submit next)

### Problem: Jobs failing with OOM

**Symptoms:** Jobs show FAILED in sacct, logs show "OutOfMemory"

**Check:**
```bash
# Find OOM jobs in manifest
jq -r '.[] | select(.error_type == "OOM_MAX_MEMORY")' \
  scripts/metapaths/results/manifest.json
```

**Solution:** Orchestrator auto-retries at higher memory (250GB → 500GB → 1TB). If job fails at 1TB, it's marked as permanently failed. These are rare and may need manual investigation.

### Problem: Job failed with non-OOM error

**Symptoms:** Job state is FAILED but not due to memory

**Check:**
```bash
# View job error log
cat slurm-<job_id>.out | tail -50

# Check manifest for error type
jq '.matrix1_042' scripts/metapaths/results/manifest.json
```

**Solution:**
- If code error: Fix analyze_3hop_overlap.py and manually rerun
- If cluster issue: Manually resubmit: `sbatch --mem=250G scripts/metapaths/run_single_matrix1.sh 42`

### Problem: Orchestrator crashed

**Symptoms:** Process died, but jobs still running

**Solution:**
```bash
# Restart orchestrator - it resumes from manifest
uv run python scripts/metapaths/orchestrate_3hop_analysis.py
```

The orchestrator is stateless and restartable. It syncs with SLURM and manifest on startup.

### Problem: Missing result files after completion

**Symptoms:** merge_results.py reports missing files

**Check:**
```bash
# Find jobs that completed but have no output file
for i in {0..2878}; do
  matrix_id=$(printf "matrix1_%03d" $i)
  file="scripts/metapaths/results/results_${matrix_id}.tsv"
  status=$(jq -r ".${matrix_id}.status" scripts/metapaths/results/manifest.json)
  if [ "$status" = "completed" ] && [ ! -f "$file" ]; then
    echo "Missing: $matrix_id (marked completed)"
  fi
done
```

**Solution:**
- Check if job actually completed or failed after manifest update
- Manually rerun missing indices
- Investigate job logs for silent failures

## Manual Job Submission (Testing)

Test individual matrices before full run:

```bash
# Submit single job
sbatch --mem=250G scripts/metapaths/run_single_matrix1.sh 42

# Submit with higher memory
sbatch --mem=500G scripts/metapaths/run_single_matrix1.sh 42

# Check output
tail -f slurm-<job_id>.out

# View results
head scripts/metapaths/results/results_matrix1_042.tsv
```

## File Structure

```
scripts/metapaths/
├── README.md                        # This file
├── WORKFLOW.md                      # Workflow overview
├── IMPLEMENTATION_PLAN.md           # Detailed design doc
│
├── prepare_analysis.py              # Step 1: Initialize
├── orchestrate_3hop_analysis.py     # Step 2: Run jobs
├── merge_results.py                 # Step 3: Merge results
├── group_by_onehop.py               # Step 4: Group by 1-hop metapath
├── run_single_matrix1.sh            # SLURM worker script
├── analyze_3hop_overlap.py          # Core analysis (with diagonal zeroing)
│
├── results/                         # Raw output files (6.9 GB)
│   ├── manifest.json                # Job tracking (live updates)
│   ├── results_matrix1_000.tsv      # Per-job results (2,879 files)
│   ├── results_matrix1_001.tsv
│   ├── ...
│   └── all_3hop_overlaps.tsv        # Final merged output (after merge)
│
├── logs/                            # SLURM stdout/stderr (913 MB)
│   ├── matrix1_000_mem250.out
│   ├── matrix1_000_mem250.err
│   └── ...
│
└── grouped_by_1hop/                 # Grouped results (14 GB)
    ├── Drug_has_adverse_event_F_Disease.tsv
    ├── Drug_treats_F_Disease.tsv
    └── ...                          # One file per unique 1-hop metapath
```

**Note:** SLURM job output files (`slurm-*.out`) appear in the directory where orchestrator runs.

## Configuration

### Memory Tiers

Defined in `prepare_analysis.py` (initial) and `orchestrate_3hop_analysis.py` (retries):

```python
# Initial tier (matches most cluster nodes with ~191GB available)
memory_tier: 180  # GB

# Auto-retry progression on OOM
180 GB → 250 GB → 500 GB → 1000 GB (1TB) → 1500 GB (1.5TB)
```

**Rationale:** Starting at 180GB allows jobs to use the majority of cluster nodes (~191GB RAM), maximizing parallelism. Higher tiers are used only when needed via automatic OOM retry. The 1500GB tier uses the largemem partition (~1495GB available) for the most demanding jobs.

### Queue Management

```python
MAX_PENDING_JOBS = 1  # Max jobs in PENDING state
POLL_INTERVAL = 30    # Seconds between SLURM checks
```

### Data Paths

Hardcoded in `run_single_matrix1.sh`:

```bash
EDGES_FILE="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/49e5e5585f7685b4/edges.jsonl"
NODES_FILE="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/49e5e5585f7685b4/nodes.jsonl"
```

## Architecture Details

### Diagonal Zeroing (Node Revisiting Prevention)

The system prevents nodes from appearing multiple times in 3-hop paths through three-level diagonal zeroing:

**Problem:** Without diagonal zeroing, you get invalid paths like:
- Self-loops: `NodeA → NodeA` (direct self-edge)
- Revisiting: `NodeA → NodeB → NodeA → NodeC` (node appears twice)
- Start/end same: `NodeA → NodeB → NodeC → NodeA` (circular path)

**Solution:** Zero out matrix diagonals at three points:

1. **Input matrices (build_matrices):**
   - Removes self-loops from input edges: `A → A`
   - Applied to all square matrices after construction

2. **After Matrix1 @ Matrix2:**
   - Prevents `A → B → A` patterns in intermediate result
   - Blocks paths like `NodeA → NodeB → NodeA → NodeC`

3. **After (Matrix1 @ Matrix2) @ Matrix3:**
   - Final safeguard ensuring start ≠ end
   - Removes any remaining circular paths: `A → B → C → A`

**Implementation:**
```python
if matrix.nrows == matrix.ncols:
    matrix = matrix.select(gb.select.offdiag).new()
```

**Result:** All 3-hop paths guaranteed to have 4 distinct nodes.

### Duplicate Elimination

Each path can be computed from either end (A→B→C→D or D→C→B→A). The system eliminates duplicates by only computing from one direction:

**Rule:** Only process path if `M3.nvals >= M1.nvals`

**Example:**
- Matrix1 has 1,000 edges (M1.nvals = 1000)
- Matrix3 has 5,000 edges (M3.nvals = 5000)
- Path A→B→C→D: When M1=A, M3=D → 5000 >= 1000 ✓ → Compute
- Path D→C→B→A: When M1=D, M3=A → 1000 >= 5000 ✗ → Skip (already computed)

**Result:** Each unique path computed exactly once (~2x speedup)

### Memory Tiering

Jobs start at 250GB. On OOM failure, orchestrator:
1. Detects exit code 137 or SLURM state "OUT_OF_MEMORY"
2. Increments memory tier: 250GB → 500GB → 1TB
3. Resets job status to "pending"
4. Resubmits automatically

If job fails at 1TB, it's marked permanently failed for manual investigation.

### Job States

**pending**: Ready to submit
**running**: Submitted to SLURM
**completed**: Finished successfully
**failed**: Permanent failure

Transitions managed by orchestrator via manifest.json updates.

## Testing Results

Test run with 3 matrices:

| Matrix | Status | Memory | Time | Rows |
|--------|--------|--------|------|------|
| matrix1_000 | OOM (retry at 500GB) | 80GB+ | 4:54 | - |
| matrix1_001 | ✓ Success | 4.3GB | 3:53 | 534 |
| matrix1_002 | ✓ Success | 4GB | 3:36 | 8 |

Most jobs complete in 3-5 minutes with <10GB memory. Small percentage need higher tiers.

## FAQ

**Q: Can I run multiple orchestrators?**
A: No. One orchestrator per analysis run. It manages the entire job pool.

**Q: Can I pause and resume?**
A: Yes. Ctrl+C to stop orchestrator. Jobs continue in SLURM. Restart orchestrator anytime.

**Q: What if cluster maintenance interrupts jobs?**
A: Restart orchestrator after maintenance. It detects incomplete jobs and resubmits.

**Q: Can I manually rerun specific matrices?**
A: Yes. Update manifest status to "pending" and restart orchestrator, or use manual submission.

**Q: How do I rerun everything from scratch?**
A: See the "Complete Rerun from Scratch" section above for detailed instructions. Short version: clean results/logs/grouped directories, run prepare_analysis.py, then orchestrate_3hop_analysis.py, merge_results.py, and group_by_onehop.py.

**Q: Why max 1 pending job?**
A: To be a good cluster citizen and avoid filling the queue. Running jobs have no limit.

## Support

For issues or questions:
1. Check logs (SLURM output files: `slurm-*.out`)
2. Check manifest status: `jq '.matrix1_XXX' scripts/metapaths/results/manifest.json`
3. Review this README troubleshooting section
4. Contact: Chris Bizon
