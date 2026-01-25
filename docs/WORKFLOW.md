# Parallel 3-Hop Metapath Analysis Workflow

## Quick Start

```bash
# 1. Initialize (creates manifest and directories)
uv run python scripts/prepare_analysis.py \
  --edges /path/to/edges.jsonl \
  --nodes /path/to/nodes.jsonl \
  --config config/type_expansion.yaml  # Optional, uses default if omitted

# 2. Run orchestrator (submit and monitor jobs)
uv run python scripts/orchestrate_hop_analysis.py \
  --config config/type_expansion.yaml  # Optional

# 3. Merge results (after all jobs complete)
uv run python scripts/merge_results.py --n-hops 3

# 4. Group by 1-hop metapath with performance metrics
uv run python scripts/group_by_onehop.py
```

## Configuration

The system uses hierarchical type expansion by default. Configure via `config/type_expansion.yaml`:

```yaml
type_expansion:
  enabled: true
  max_depth: null  # Unlimited hierarchy depth
  exclude_types:   # Abstract types to skip
    - ThingWithTaxon
    - SubjectOfInvestigation
    - PhysicalEssenceOrOccurrent
    - PhysicalEssence
    - OntologyClass
    - Occurrent
    - InformationContentEntity
    - Attribute
  include_most_specific: true  # Always include most specific type
```

**Key features:**
- **Hierarchical types**: Nodes participate as ALL their types, not just most specific
- **Type filtering**: Exclude overly abstract types
- **Configurable**: Pass `--config` to any script to use custom config

## Manual Testing (Small Scale)

Test with a single matrix before full run:

```bash
# Test matrix 0
sbatch --mem=250G scripts/run_single_matrix1.sh 0 /path/to/nodes.jsonl /path/to/edges.jsonl 3 "" config/type_expansion.yaml

# Check output
tail -f logs_3hop/matrix1_000_mem250.out

# View results
head results_3hop/results_matrix1_000.tsv
```

## Architecture

### Scripts Created

1. **prepare_analysis.py** - Initialization
   - Loads matrices to count total jobs
   - Creates results/ and logs/ directories
   - Generates manifest.json

2. **run_single_matrix1.sh** - SLURM worker
   - Processes single Matrix1 index
   - Hardcoded data paths
   - Memory tier set by orchestrator

3. **orchestrate_3hop_analysis.py** - Master controller
   - Monitors manifest and SLURM queue
   - Submits jobs (max 1 pending)
   - Auto-retries OOM failures at higher memory

4. **merge_results.py** - Results combiner
   - Merges all results_matrix1_*.tsv files
   - Verifies completeness
   - Generates final output

### Key Features

**Duplicate Elimination**: Only computes paths where M3.nvals >= M1.nvals
- Ensures each path computed exactly once
- ~2x speedup

**Memory Tiering**: 250GB → 500GB → 1TB
- Start at 250GB
- Auto-retry OOM failures at next tier
- Maximum 3 attempts per job

**Queue Management**: Max 1 pending job
- Avoids queue hogging
- Unlimited running jobs allowed

## File Structure

```
scripts/metapaths/
├── prepare_analysis.py          # Initialization
├── orchestrate_3hop_analysis.py # Master orchestrator
├── run_single_matrix1.sh        # SLURM worker
├── merge_results.py             # Results merger
├── analyze_hop_overlap.py      # Modified core analysis (--matrix1-index)
├── logs/                        # SLURM stdout/stderr
│   └── matrix1_000_mem250.out
└── results/                     # Output files
    ├── manifest.json            # Job tracking
    ├── results_matrix1_000.tsv  # Per-job results
    └── all_3hop_overlaps.tsv    # Final merged output
```

## Monitoring

**Check orchestrator progress:**
```bash
# Watch real-time
tail -f scripts/metapaths/logs/matrix1_*_mem*.out | grep "Matrix2:"

# Check manifest status
jq '[.[] | select(.status == "completed")] | length' scripts/metapaths/results/manifest.json

# SLURM queue
squeue -u $USER | grep 3hop
```

**Find failed jobs:**
```bash
jq -r '.[] | select(.status == "failed") | "\(.)|t\(.error_type)"' \
  scripts/metapaths/results/manifest.json
```

## Troubleshooting

**Problem**: Jobs stuck in pending
- Check: `squeue -u $USER -t PENDING`
- Solution: Orchestrator maintains max 1 pending automatically

**Problem**: Job failed with OOM
- Check: Look for exit code 137 in logs
- Solution: Orchestrator auto-retries at higher memory

**Problem**: Job failed with other error
- Check: Log files in `scripts/metapaths/logs/`
- Check: Manifest error_type field
- Solution: Investigate specific error, may need manual retry

**Problem**: Missing result files after completion
- Check: Manifest for failed jobs
- Check: Log files for errors
- Solution: Manually rerun failed indices or investigate cause

## Performance Estimates

**Sequential (original)**: ~5 days
**Parallel (with ~250 jobs)**: ~2-4 hours

Actual time depends on:
- Cluster load
- Matrix sizes (varied)
- Memory availability
