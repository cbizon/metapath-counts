# Metapath Counts

Parallel 3-hop metapath analysis for biolink knowledge graphs using GraphBLAS sparse matrices and SLURM cluster computing.

## Overview

This package calculates metapath overlaps in large knowledge graphs by:
- Computing all possible 3-hop paths via sparse matrix multiplication
- Eliminating duplicate paths via directional filtering
- Running massively parallel jobs on SLURM clusters with automatic memory tiering

The primary output is metapath statistics that can be used for rule mining and pattern discovery in biolink knowledge graphs.

## Features

- **Hierarchical Type Expansion**: Nodes participate as ALL their types in the Biolink hierarchy (not just most specific)
- **Per-Path OOM Recovery**: Granular tracking and retry at individual path level
- **Parallel Processing**: Distributes jobs across SLURM cluster (693 jobs with hierarchical types)
- **Memory Tiering**: Automatic retry at higher memory (180GB → 250GB → 500GB → 1TB → 1.5TB)
- **Smart Retry**: Skip completed paths, only retry failed ones
- **Duplicate Elimination**: Each path computed exactly once (~2x speedup)
- **Performance**: ~50-100x faster than sequential (5 days → 2-4 hours)

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd metapath-counts

# Initialize environment with uv
uv venv
uv pip install -e .
```

### Basic Usage

```bash
# 1. Initialize analysis (creates manifest)
uv run python scripts/prepare_analysis.py \
  --edges /path/to/edges.jsonl \
  --nodes /path/to/nodes.jsonl

# 2. Run orchestrator (submits SLURM jobs)
uv run python scripts/orchestrate_3hop_analysis.py \
  --edges /path/to/edges.jsonl \
  --nodes /path/to/nodes.jsonl

# 3. Merge results
uv run python scripts/merge_results.py

# 4. Group by 1-hop metapath
uv run python scripts/group_by_onehop.py
```

See [docs/README.md](docs/README.md) for detailed workflow documentation.

## Input Format

Requires KGX (Knowledge Graph Exchange) format:
- **nodes.jsonl**: One JSON object per line with `id`, `category`, `name`
- **edges.jsonl**: One JSON object per line with `subject`, `predicate`, `object`

Example ROBOKOP graph: `/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/`

## Output Format

**Metapath notation**: `NodeType|predicate|direction|NodeType|...`
- `F` = forward, `R` = reverse, `A` = any (symmetric)
- Example: `Disease|treats|R|SmallMolecule|affects|F|Gene`

**Merged results** (`results/all_3hop_overlaps.tsv`):
```
3hop_metapath | 3hop_count | 1hop_metapath | 1hop_count | overlap | total_possible
```

**Grouped results** (`grouped_by_1hop/*.tsv`):
- One file per 1-hop metapath
- Includes metrics: Precision, Recall, F1, MCC, TPR, FPR, etc.

## Library Usage

```python
from metapath_counts import get_most_specific_type, get_symmetric_predicates

# Get most specific biolink type
categories = ["biolink:ChemicalEntity", "biolink:SmallMolecule"]
specific = get_most_specific_type(categories)
# Returns: "biolink:SmallMolecule"

# Get symmetric predicates
symmetric = get_symmetric_predicates()
# Returns: {'directly_physically_interacts_with', 'associated_with', ...}
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=metapath_counts --cov-report=html
```

## Documentation

- [docs/README.md](docs/README.md) - Complete workflow guide
- [docs/WORKFLOW.md](docs/WORKFLOW.md) - Quick reference
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) - Design details
- [CLAUDE.md](CLAUDE.md) - Development guide for AI assistants

## Requirements

- Python >= 3.12
- SLURM cluster environment
- Dependencies: graphblas, bmt, psutil

## Use Cases

- **Rule Mining**: Generate metapath statistics for association rule discovery
- **Pattern Discovery**: Identify common path patterns in knowledge graphs
- **Link Prediction**: Evaluate which 3-hop patterns predict 1-hop relationships

## Performance

Typical run on ROBOKOP graph:
- ~2,879 matrix multiplication jobs
- ~100-200 concurrent SLURM jobs
- 2-4 hours total runtime
- ~22 GB final output

## License

[To be determined]

## Contributing

This repository uses uv for environment management. See CLAUDE.md for development guidelines.
