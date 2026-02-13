# METAPATH COUNTS

## Setup

- **uv** for package/environment management. **pytest** for tests with high coverage.
- Use TSV not CSV. Visualizations: PNG only, never PDF.

### Environment Management - CRITICAL
**NEVER EVER INSTALL ANYTHING INTO SYSTEM LIBRARIES OR ANACONDA BASE ENVIRONMENT**
- ALWAYS use `uv run` to execute commands
- The virtual environment at `.venv/` is sacred.

## Purpose

Calculates N-hop metapath overlaps in large biolink knowledge graphs using GraphBLAS sparse matrices. Runs on SLURM clusters with automatic memory-tiered retry. The output is metapath statistics for rule mining.

## Project Structure

```
src/
├── library/                    # Importable library
│   ├── type_utils.py           # Biolink type utilities
│   ├── type_assignment.py      # Single-type-per-node assignment
│   ├── hierarchy.py            # Hierarchy inference for aggregation
│   ├── aggregation.py          # Metapath aggregation logic
│   ├── slurm.py                # SLURM orchestration utilities
│   └── path_tracker.py         # Per-path OOM recovery tracking
├── pipeline/                   # Analysis workflow
│   ├── prebuild_matrices.py    # Pre-build matrices (one-time)
│   ├── prepare_analysis.py     # Initialize SLURM job manifest
│   ├── orchestrate_analysis.py # SLURM orchestrator
│   ├── prepare_grouping.py     # Precompute aggregated counts
│   ├── orchestrate_grouping.py # Distributed grouping orchestrator
│   ├── merge_results.py        # Combine result files
│   └── workers/                # SLURM worker scripts
│       ├── run_overlap.py      # Core analysis engine
│       ├── run_grouping.py     # Grouping worker
│       ├── run_overlap.sh
│       └── run_grouping.sh
└── analysis/                   # Exploratory/one-off scripts
    └── benchmark/
tests/
```

## Domain Context

### KGX Files

Knowledge graph input is two JSONL files (nodes + edges):

**Node:** `{"id": "CURIE", "category": ["biolink:SmallMolecule", "biolink:ChemicalEntity", ...], ...}`
- `category` is hierarchical; first element is most specific

**Edge:** `{"subject": "CURIE", "predicate": "biolink:treats", "object": "CURIE", ...}`

### Metapath Format

Pipe-separated: `NodeType|predicate|direction|NodeType|...`
- `F` = forward, `R` = reverse, `A` = any (symmetric predicates)
- Example: `Disease|treats|R|SmallMolecule|affects|F|Gene`

### Key Design Decisions

- **Single type assignment:** Each node gets exactly one type (most specific leaf). Nodes with multiple unrelated leaf types get a pseudo-type like `Gene+SmallMolecule`.
- **Post-processing aggregation:** Hierarchical type/predicate expansion happens during grouping, not matrix building. This keeps matrix count linear with edges.
- **Approximate metrics:** Aggregated metrics use sum semantics (not set union) for scalability. Precision > 1.0 can occur in aggregated results. Use metrics for ranking, not absolute values.
- **No repeated-node filtering:** Matrix multiplication counts all paths including those that revisit nodes. Filtering would require path enumeration.

## Running the Pipeline

```bash
# 0. Pre-build matrices (once per KG)
uv run python src/pipeline/prebuild_matrices.py \
  --edges /path/to/edges.jsonl --nodes /path/to/nodes.jsonl --output matrices

# 1. Initialize manifest
uv run python src/pipeline/prepare_analysis.py --matrices-dir matrices --n-hops 3

# 2. Run SLURM orchestrator
uv run python src/pipeline/orchestrate_analysis.py --n-hops 3

# 3. Prepare grouping
uv run python src/pipeline/prepare_grouping.py --n-hops 3

# 4. Run distributed grouping
uv run python src/pipeline/orchestrate_grouping.py --n-hops 3 \
  --min-count 10 --min-precision 0.001
```

## Testing

```bash
uv run pytest
```

## Rules of the Road

- Ask clarifying questions
- Don't make classes just to group code. It is non-pythonic and hard to test.
- Do not implement bandaids - treat the root cause of problems
- Don't use try/except as a way to hide problems. Let things fail so we can figure out why.
- Once we have a test, do not delete it without explicit permission.
- Do not return made up results if an API fails. Let it fail.
- If you cannot access a file, do not just guess at its contents - come back and ask for help
- When changing code, don't make duplicate functions - just change the function. We can always roll back.
- Keep the directories clean, don't leave junk laying around.
- When making pull requests, NEVER mention `co-authored-by` or the tool used to create the commit/PR.
- Check git status before commits
