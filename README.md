# Metapath Counts

N-hop metapath overlap analysis for biolink knowledge graphs using GraphBLAS sparse matrices and SLURM cluster computing.

## Overview

Calculates which N-hop paths (2-hop, 3-hop) predict 1-hop relationships in large knowledge graphs. For every N-hop metapath that shares endpoints with a 1-hop target, it computes exact overlap, precision, recall, F1, and MCC using sparse matrix operations.

The output is metapath statistics for rule mining and link prediction.

## Quick Start

```bash
# Initialize environment
uv venv
uv pip install -e .
```

## Running the Full Pipeline

The easiest way to run everything is `run_analysis.sh`. Edit the `NODES` and `EDGES` paths at the top, then:

```bash
./run_analysis.sh
```

This runs all steps sequentially. Each step is resumable if interrupted.

### Individual Steps

```bash
# 0. Pre-build matrices (once per KG)
uv run python src/pipeline/prebuild_matrices.py \
  --edges /path/to/edges.jsonl --nodes /path/to/nodes.jsonl --output matrices

# 1. Initialize manifest
uv run python src/pipeline/prepare_analysis.py --matrices-dir matrices --n-hops 3

# 2. Run SLURM overlap analysis
uv run python src/pipeline/orchestrate_analysis.py --n-hops 3 --partition lowpri

# 3. Precompute explicit path counts (distributed on SLURM)
uv run python src/pipeline/precompute_aggregated_counts_slurm.py --n-hops 3

# 4. Prepare grouping (type node counts, job manifest)
uv run python src/pipeline/prepare_grouping.py --n-hops 3

# 5. Run distributed grouping
uv run python src/pipeline/orchestrate_grouping.py --n-hops 3 \
  --min-count 10 --min-precision 0.001 --partition lowpri
```

### Re-running Grouping Only

If you need to change filters or exclude lists without re-running the overlap analysis:

```bash
./rerun_grouping.sh 3   # argument is n_hops (default: 2)
```

## Input Format

Requires KGX (Knowledge Graph Exchange) format:
- **nodes.jsonl**: One JSON object per line with `id`, `category`, `name`
- **edges.jsonl**: One JSON object per line with `subject`, `predicate`, `object`

## Output Format

**Metapath notation**: `NodeType|predicate|direction|NodeType|...`
- `F` = forward, `R` = reverse, `A` = any (symmetric)
- Example: `SmallMolecule|affects|F|Gene|affects|F|Disease`

**Grouped results** (`grouped_by_results_3hop/*.tsv.zst`):
- One zstd-compressed file per 1-hop target metapath
- Each row is an explicit N-hop predictor path with exact metrics
- Columns: predictor_metapath, predictor_count, overlap, total_possible, precision, recall, f1, mcc, specificity, npv

**Analysis results** (`results_3hop/results_matrix1_*.tsv`):
- Raw pairwise overlap counts between N-hop and 1-hop paths

## Default Filters

The grouping step applies these filters (configurable via CLI):

- `--min-count 10`: Minimum predictor pair count
- `--min-precision 0.001`: Minimum precision threshold
- `--exclude-types Entity,ThingWithTaxon,PhysicalEssence,PhysicalEssenceOrOccurrent`
- `--exclude-predicates related_to,related_to_at_instance_level,related_to_at_concept_level,associated_with`

## Testing

```bash
uv run pytest
```

## Requirements

- Python >= 3.12
- SLURM cluster environment
- Dependencies: python-graphblas, bmt, psutil, zstandard
