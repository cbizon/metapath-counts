# Parallel N-Hop Metapath Analysis System

This system parallelizes N-hop metapath analysis across SLURM cluster with automatic memory-tiered retry.

## Quick Start

```bash
# 0. One-time: Pre-build matrices (only needs to be done once per KG)
uv run python scripts/prebuild_matrices.py \
  --edges /path/to/edges.jsonl \
  --nodes /path/to/nodes.jsonl \
  --output matrices

# 1. Initialize (creates manifest and directories)
uv run python scripts/prepare_analysis.py \
  --matrices-dir matrices \
  --n-hops 3

# 2. Run orchestrator (handles everything automatically)
uv run python scripts/orchestrate_hop_analysis.py \
  --n-hops 3

# 3. Precompute aggregated N-hop counts (SLURM)
uv run python src/pipeline/precompute_aggregated_counts_slurm.py --n-hops 3

# 4. Prepare distributed grouping (creates type pair manifest)
uv run python src/pipeline/prepare_grouping.py --n-hops 3 --skip-aggregated-precompute

# 5. Run distributed grouping with filters
uv run python src/pipeline/orchestrate_grouping.py --n-hops 3 \
    --min-count 10 --min-precision 0.001
```

## System Overview

### What This System Does

1. **Splits work** across ~2,879 independent jobs (one per Matrix1)
2. **Eliminates duplicates** via M3 filtering (M3.nvals >= M1.nvals rule)
3. **Auto-retries OOM failures** at higher memory tiers: 180GB → 250GB → 500GB → 1TB → 1.5TB
5. **Manages queue** to avoid overwhelming cluster (max 1 pending job)
6. **Tracks progress** via manifest.json
7. **Merges and groups results** for analysis

### Expected Performance

- **Sequential (old)**: ~5 days
- **Parallel (new)**: ~2-4 hours with ~100-200 concurrent jobs
- **Speedup**: ~50-100x

## Step-by-Step Instructions

### Step 0: Pre-build Matrices (One-Time Setup Per KG)

Pre-build and serialize all matrices from the knowledge graph:

```bash
uv run python scripts/prebuild_matrices.py \
  --edges /path/to/edges.jsonl \
  --nodes /path/to/nodes.jsonl \
  --output matrices
```

**Output:**
- Creates `matrices/` directory with serialized .npz files
- Creates `matrices/manifest.json` with metadata
- Total size: ~5GB (compressed from ~84GB edges file)

**Time:** ~10-15 minutes (depends on graph size)

**Note:** Only needs to be done once per knowledge graph. All analysis runs will use these pre-built matrices.

### Step 1: Initialize Analysis Run

Initialize manifest and directories for specific N-hop analysis:

```bash
uv run python scripts/prepare_analysis.py \
  --matrices-dir matrices \
  --n-hops 3
```

**Output:**
- Creates `results_3hop/` directory
- Creates `logs_3hop/` directory
- Generates `results_3hop/manifest.json` with all jobs

**Time:** <5 seconds (just reads manifest metadata, doesn't load matrices)

**Note:** Run this once per N-hop analysis (1-hop, 2-hop, 3-hop, etc.)

### Step 2: Run Orchestrator

Submit and monitor all jobs automatically:

```bash
uv run python scripts/orchestrate_hop_analysis.py --n-hops 3
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
  uv run python scripts/orchestrate_hop_analysis.py --n-hops 3
  ```

- Option B: Run in background with logs
  ```bash
  nohup uv run python scripts/orchestrate_hop_analysis.py --n-hops 3 > orchestrator.log 2>&1 &
  tail -f orchestrator.log
  ```

- Option C: Run in screen/tmux session (recommended)
  ```bash
  screen -S metapath_orchestrator
  uv run python scripts/orchestrate_hop_analysis.py --n-hops 3
  # Detach: Ctrl+A, then D
  # Reattach: screen -r metapath_orchestrator
  ```

**To stop orchestrator:**
- Ctrl+C (graceful shutdown)
- Jobs will continue running in SLURM
- Restart orchestrator anytime - it resumes from manifest

### Step 3: Precompute Aggregated Counts (SLURM)

After all analysis jobs complete, precompute aggregated N-hop counts on SLURM:

```bash
uv run python src/pipeline/precompute_aggregated_counts_slurm.py --n-hops 3
```

This submits Pass A/B map-reduce jobs and writes:
- `results_3hop/aggregated_nhop_counts.json`

### Step 4: Prepare Grouping

Prepare distributed grouping (loads the SLURM-precomputed counts):

```bash
uv run python src/pipeline/prepare_grouping.py --n-hops 3 --skip-aggregated-precompute
```

**Output:**
- Creates `results_3hop/grouping_manifest.json` with type pair jobs
- Loads `results_3hop/aggregated_nhop_counts.json` for global lookups
- Precomputes type node counts for total_possible calculations

### Step 5: Run Distributed Grouping

Run distributed grouping with filters:

```bash
uv run python src/pipeline/orchestrate_grouping.py --n-hops 3 \
    --min-count 10 --min-precision 0.001
```

**What it does:**
- Distributes grouping across SLURM jobs (one per type pair)
- Normalizes 1-hop metapaths to forward (F) direction
- Reverses N-hop paths when needed to match normalization
- Calculates performance metrics (Precision, Recall, F1, MCC, etc.)
- Applies filters to reduce output size
- Writes one TSV file per unique 1-hop metapath

**Filters (applied by default):**
- `--min-count 10`: Exclude rules with fewer than 10 predictor paths
- `--min-precision 0.001`: Exclude rules with precision < 0.1%
- `--exclude-types Entity,ThingWithTaxon`: Exclude overly general node types
- `--exclude-predicates related_to_at_instance_level,related_to_at_concept_level`: Exclude overly general predicates

**Output:**
- `grouped_by_results_3hop/*.tsv` (one file per 1-hop metapath)
- Example: `Drug_has_adverse_event_F_Disease.tsv` contains all 3-hop paths that predict that 1-hop

**Options:**
```bash
# Run with different filter thresholds
uv run python src/pipeline/orchestrate_grouping.py --n-hops 3 \
    --min-count 100 --min-precision 0.01

# Disable type/predicate exclusions
uv run python src/pipeline/orchestrate_grouping.py --n-hops 3 \
    --exclude-types "" --exclude-predicates ""
```

**Time:** ~1-2 hours (distributed across SLURM)

**Output columns:**
- Original: `predictor_metapath`, `predictor_count`, `predicted_metapath`, `predicted_count`, `overlap`, `total_possible`
- Added metrics: `TP`, `FP`, `FN`, `TN`, `Total`, `Precision`, `Recall`, `Specificity`, `NPV`, `Accuracy`, `Balanced_Accuracy`, `F1`, `MCC`, `TPR`, `FPR`, `FNR`, `PLR`, `NLR`

## Complete Rerun from Scratch

If you need to regenerate all results (e.g., after algorithm changes), follow these steps:

### Step 1: Clean Old Results

```bash
# WARNING: This deletes all results
# Optional: backup first if you want to compare old vs new results
mkdir -p old_results_backup
mv results_3hop old_results_backup/results_$(date +%Y%m%d)
mv logs_3hop old_results_backup/logs_$(date +%Y%m%d)
mv grouped_by_results_3hop old_results_backup/grouped_$(date +%Y%m%d)

# Or just delete everything
rm -rf results_3hop logs_3hop logs_grouping_3hop grouped_by_results_3hop
```

### Step 2: Regenerate Manifest

```bash
uv run python scripts/prepare_analysis.py --matrices-dir matrices --n-hops 3
```

This creates a fresh `results_3hop/manifest.json` with all jobs set to "pending".

**Time:** ~3-4 minutes (loads entire graph)

### Step 3: Run Orchestrator

```bash
# Recommended: run in screen/tmux
screen -S metapath_orchestrator
uv run python scripts/orchestrate_hop_analysis.py --n-hops 3
# Detach: Ctrl+A, then D
```

**Time:** ~2-4 hours

### Step 4: Group Results

```bash
uv run python src/pipeline/precompute_aggregated_counts_slurm.py --n-hops 3
uv run python src/pipeline/prepare_grouping.py --n-hops 3 --skip-aggregated-precompute
uv run python src/pipeline/orchestrate_grouping.py --n-hops 3 \
    --min-count 10 --min-precision 0.001
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
jq '[.[] | select(.status == "completed")] | length' results_3hop/manifest.json

# Count failed jobs
jq '[.[] | select(.status == "failed")] | length' results_3hop/manifest.json

# List failed jobs with errors
jq -r '.[] | select(.status == "failed") | "\(.)|error:\(.error_type)"' \
  results_3hop/manifest.json

# Find jobs that needed high memory
jq -r '.[] | select(.memory_tier > 250) | "\(.): \(.memory_tier)GB"' \
  results_3hop/manifest.json
```

### Check Individual Job Logs

SLURM output files are in logs directories:

```bash
# List recent analysis log files
ls -lth logs_3hop/*.out | head -20

# List recent grouping log files
ls -lth logs_grouping_3hop/*.out | head -20

# Search for errors
grep -i error logs_3hop/*.out
```

### Check Result Files

```bash
# Count generated result files
ls results_3hop/results_matrix1_*.tsv | wc -l

# List largest result files
ls -lhS results_3hop/results_matrix1_*.tsv | head -10

# Check specific result
head results_3hop/results_matrix1_042.tsv
wc -l results_3hop/results_matrix1_042.tsv
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
  results_3hop/manifest.json
```

**Solution:** Orchestrator auto-retries at higher memory (180GB → 250GB → 500GB → 1TB → 1.5TB). If job fails at max tier, it's marked as permanently failed. These are rare and may need manual investigation.

### Problem: Job failed with non-OOM error

**Symptoms:** Job state is FAILED but not due to memory

**Check:**
```bash
# View job error log
cat logs_3hop/matrix1_042_mem*.out | tail -50

# Check manifest for error type
jq '.matrix1_042' results_3hop/manifest.json
```

**Solution:**
- If code error: Fix analyze_hop_overlap.py and manually rerun
- If cluster issue: Manually resubmit: `sbatch --mem=250G scripts/run_single_matrix1.sh 42 3`

### Problem: Orchestrator crashed

**Symptoms:** Process died, but jobs still running

**Solution:**
```bash
# Restart orchestrator - it resumes from manifest
uv run python scripts/orchestrate_hop_analysis.py --n-hops 3
```

The orchestrator is stateless and restartable. It syncs with SLURM and manifest on startup.

### Problem: Missing result files after completion

**Symptoms:** Grouping jobs fail due to missing result files

**Check:**
```bash
# Find jobs that completed but have no output file
for i in {0..2878}; do
  matrix_id=$(printf "matrix1_%03d" $i)
  file="results_3hop/results_${matrix_id}.tsv"
  status=$(jq -r ".${matrix_id}.status" results_3hop/manifest.json)
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
# Submit single job (for 3-hop analysis)
sbatch --mem=250G scripts/run_single_matrix1.sh 42 3

# Submit with higher memory
sbatch --mem=500G scripts/run_single_matrix1.sh 42 3

# Check output
tail -f logs_3hop/matrix1_042_mem250.out

# View results
head results_3hop/results_matrix1_042.tsv
```

## File Structure

```
metapath-counts/
├── docs/
│   └── README.md                    # This file
│
├── src/pipeline/
│   ├── prepare_analysis.py          # Step 1: Initialize analysis
│   ├── orchestrate_analysis.py      # Step 2: Run analysis jobs
│   ├── precompute_aggregated_counts_slurm.py  # Step 3: SLURM precompute
│   ├── prepare_grouping.py          # Step 4: Initialize grouping
│   ├── orchestrate_grouping.py      # Step 5: Run grouping jobs
│   └── workers/                     # SLURM worker scripts
│
├── results_3hop/                    # Analysis results
│   ├── manifest.json                # Job tracking (live updates)
│   ├── results_matrix1_000.tsv      # Per-job results
│   ├── ...
│   ├── aggregated_nhop_counts.json  # Precomputed counts (from SLURM precompute)
│   └── type_node_counts.json        # Node counts per type
│
├── logs_3hop/                       # SLURM logs (analysis)
│
├── logs_grouping_3hop/              # SLURM logs (grouping)
│
└── grouped_by_results_3hop/         # Grouped results
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

### Approximate Metrics (Precision, Recall, etc.)

**IMPORTANT:** The performance metrics (Precision, Recall, F1, MCC, etc.) in grouped results should be treated as **approximate values**, not exact calculations. There are two known sources of approximation:

#### 1. Overlap Aggregation Uses Sum Semantics

When aggregating explicit results to hierarchical types/predicates, overlaps are **summed** rather than computing the true **set union**. This causes overcounting when a node pair has multiple edge types that roll up to the same aggregated target.

**Example:**
- Node pair (Gene_A, Disease_B) has both a `treats` edge AND a `regulates` edge
- Both predicates have `related_to` as an ancestor
- Aggregated overlap counts this pair **twice** instead of once

**Consequence:** For aggregated paths, `overlap` can exceed `predictor_count`, causing `precision > 1.0`.

**Why not fix it?** Computing true set unions requires tracking individual node pairs, which is memory-prohibitive for large graphs.

#### 2. Node Revisiting in Longer Paths

The implementation does NOT filter out paths with repeated nodes. Path counts include all paths regardless of whether nodes are revisited (e.g., `A → B → A → C` is counted). This is intentional because:

1. Matrix-based diagonal zeroing can only prevent revisiting the **start** node, not intermediate nodes
2. Properly filtering all repeated nodes would require path enumeration, which is computationally expensive
3. For rule mining purposes, the statistical signal from repeated-node paths is often still useful

#### Practical Guidance

- **Use metrics for ranking/comparison**, not as absolute values
- **Explicit-only results** (`--explicit-only`) have exact metrics
- **Aggregated results** are approximate but useful for pattern discovery
- **Precision > 1.0** indicates overlap aggregation error; treat with caution

If distinct-node paths or exact aggregated metrics are required, post-processing filters can be applied.

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

## Type Assignment and Hierarchical Aggregation

The system uses **single-type assignment** during matrix building, then **aggregates to hierarchy** during post-processing. This approach avoids matrix explosion while still providing comprehensive hierarchical coverage.

### How It Works

**Step 1: Single-Type Assignment (Matrix Building)**
- Each node assigned to exactly ONE type
- Node with `["SmallMolecule", "ChemicalEntity"]` → assigned to `SmallMolecule` (most specific)
- Node with `["Gene", "SmallMolecule"]` (multi-leaf) → assigned to pseudo-type `Gene+SmallMolecule`
- Fast: No matrix explosion (~4 hours vs multiple days)

**Step 2: Hierarchical Aggregation (Post-Processing)**
- Results aggregated during `group_by_onehop.py` step
- Pseudo-types expanded: `Gene+SmallMolecule` contributes to both `Gene` and `SmallMolecule` paths
- Type hierarchy: Results propagate to ancestors (e.g., `SmallMolecule` → `ChemicalEntity`)
- Predicate hierarchy: Results propagate to ancestor predicates

**Example:**
```
Explicit result: Gene+SmallMolecule|affects|Disease (count: 100)

Aggregated to:
  - Gene|affects|Disease (100)
  - SmallMolecule|affects|Disease (100)
  - BiologicalEntity|affects|Disease (100)  # Gene ancestor
  - ChemicalEntity|affects|Disease (100)    # SmallMolecule ancestor
```

### Benefits

- **No matrix explosion**: Linear scaling with edges (not 15-20x)
- **Fast runtime**: ~4 hours (return to original speed)
- **Comprehensive results**: Same hierarchical coverage via post-processing
- **Lower memory**: Much lower per-job memory requirements

### Disabling Aggregation (Debug Mode)

To see only explicit results without aggregation:

```bash
uv run python scripts/group_by_onehop.py --n-hops 3 --explicit-only
```

### Per-Path OOM Recovery

The system tracks completion at **individual path granularity**, not just job-level:

**How it works:**
1. Each 3-hop path computed is recorded to `completed_paths.txt`
2. Failed paths recorded to `failed_paths.jsonl` with memory tier
3. Jobs can complete partially (some paths succeed, some fail)
4. Retry attempts skip completed paths, only retry failed ones
5. Multi-tier retry: 180GB → 250GB → 500GB → 1TB → 1.5TB

**Example:**
- Job computes 100 paths at 180GB
- 80 paths succeed, 20 fail with OOM
- Job retries at 250GB
- 80 completed paths skipped, only 20 attempted
- 15 of 20 succeed at 250GB, 5 still fail
- Job retries at 500GB for remaining 5

**Result:** Maximum path completion even with memory constraints.

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
A: See the "Complete Rerun from Scratch" section above for detailed instructions. Short version: clean results/logs/grouped directories, run prepare_analysis.py, orchestrate_analysis.py, precompute_aggregated_counts_slurm.py, prepare_grouping.py (with `--skip-aggregated-precompute`), and orchestrate_grouping.py.

**Q: Why max 1 pending job?**
A: To be a good cluster citizen and avoid filling the queue. Running jobs have no limit.

## Support

For issues or questions:
1. Check logs (SLURM output files in `logs_*hop/` directories)
2. Check manifest status: `jq '.matrix1_XXX' results_3hop/manifest.json`
3. Review this README troubleshooting section
4. Contact: Chris Bizon
