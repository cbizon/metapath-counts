# METAPATH COUNTS

## Basic Setup

* github: This project will have a github repo (to be created)
* uv: we are using uv for package and environment management and an isolated environment
* tests: we are using pytest, and want to maintain high code coverage

### Environment Management - CRITICAL
**NEVER EVER INSTALL ANYTHING INTO SYSTEM LIBRARIES OR ANACONDA BASE ENVIRONMENT**
- ALWAYS use the isolated virtual environment at `.venv/`
- ALWAYS use `uv run` to execute commands, which automatically uses the isolated environment
- The virtual environment is sacred. System packages are not your garbage dump.

## Purpose

This package calculates 3-hop metapath overlaps in large biolink knowledge graphs using GraphBLAS sparse matrices. It runs on SLURM clusters with automatic memory-tiered retry for parallel processing.

The primary use case is generating metapath statistics that can be used for rule mining in biolink knowledge graphs.

## Project Structure

```
metapath-counts/
├── src/metapath_counts/         # Source code (library)
│   ├── __init__.py
│   └── type_utils.py            # Biolink type utilities
├── scripts/                     # Analysis scripts (CLI)
│   ├── prebuild_matrices.py     # Pre-build matrices (one-time setup)
│   ├── analyze_hop_overlap.py  # Core analysis engine
│   ├── prepare_analysis.py      # Initialize SLURM job manifest
│   ├── orchestrate_hop_analysis.py # SLURM orchestrator
│   ├── merge_results.py         # Combine result files
│   ├── group_by_onehop.py       # Group by 1-hop metapath
│   └── run_single_matrix1.sh    # SLURM worker script
├── tests/                       # Test suite
├── docs/                        # Documentation
│   ├── README.md                # Main workflow documentation
│   ├── WORKFLOW.md              # Quick reference
│   └── IMPLEMENTATION_PLAN.md   # Design details
├── results/                     # Output files (gitignored)
├── logs/                        # SLURM logs (gitignored)
└── grouped_by_1hop/             # Grouped results (gitignored)
```

## KGX Files

Our Knowledge Graph (KG) files come as two jsonlines files: a nodes file and an edges file.

**Example node:**
```json
{"id":"PUBCHEM.COMPOUND:3009304","name":"1H-1,3-Diazepine...","category":["biolink:SmallMolecule","biolink:MolecularEntity","biolink:ChemicalEntity"],"equivalent_identifiers":["PUBCHEM.COMPOUND:3009304","CHEMBL.COMPOUND:CHEMBL29089"]}
```

Important elements:
- `id`: CURIE identifier for the node
- `category`: List of biolink categories (hierarchical, first is most specific)
- `equivalent_identifiers`: Alternative identifiers

**Example edge:**
```json
{"subject":"NCBITaxon:1661386","predicate":"biolink:subclass_of","object":"NCBITaxon:286","primary_knowledge_source":"infores:ubergraph"}
```

Important attributes:
- `subject`, `object`: Node IDs from the nodes file
- `predicate`: Relation type
- Qualifiers: Additional edge attributes (e.g., `object_direction_qualifier`)

## Running Metapath Analysis

### Quick Start

See `docs/README.md` for detailed instructions. Basic workflow:

```bash
# 0. One-time: Pre-build matrices (only needs to be done once per KG)
uv run python scripts/prebuild_matrices.py \
  --edges /path/to/edges.jsonl \
  --nodes /path/to/nodes.jsonl \
  --output matrices \
  --config config/type_expansion.yaml

# 1. Initialize (creates manifest and directories)
uv run python scripts/prepare_analysis.py \
  --matrices-dir matrices \
  --n-hops 3

# 2. Run orchestrator (submit and monitor SLURM jobs)
uv run python scripts/orchestrate_hop_analysis.py \
  --n-hops 3

# 3. Merge results (after all jobs complete)
uv run python scripts/merge_results.py --n-hops 3

# 4. Group by 1-hop metapath with performance metrics
uv run python scripts/group_by_onehop.py --n-hops 3
```

### Output Format

**Merged results** (`results/all_3hop_overlaps.tsv`):
```
3hop_metapath | 3hop_count | 1hop_metapath | 1hop_count | overlap | total_possible
SmallMolecule|affects|F|Gene|affects|R|SmallMolecule|affects|F|Gene | 6170000000 | SmallMolecule|regulates|F|Gene | 500000 | 450000 | 201000000000
```

**Grouped results** (`grouped_by_1hop/*.tsv`):
- One file per unique 1-hop metapath
- Contains all 3-hop paths that predict that 1-hop
- Includes performance metrics: Precision, Recall, F1, MCC, etc.

### Metapath Format

Pipe-separated: `NodeType|predicate|direction|NodeType|...`
- `F` = forward, `R` = reverse, `A` = any (symmetric predicates)
- Example: `Disease|treats|R|SmallMolecule|affects|F|Gene`
- Symmetric example: `Gene|directly_physically_interacts_with|A|Gene`

## Key Implementation Details

### Hierarchical Type Expansion

**Default behavior:** Nodes participate in metapaths as ALL their types in the Biolink hierarchy, not just the most specific type.

**Example:**
- Node with categories: `["SmallMolecule", "ChemicalEntity", "MolecularEntity"]`
- Participates as: `SmallMolecule`, `ChemicalEntity`, AND `MolecularEntity`
- Creates paths like:
  - `SmallMolecule → affects → Gene`
  - `ChemicalEntity → affects → Gene`
  - `MolecularEntity → affects → Gene`

**Configuration** (`config/type_expansion.yaml`):
```yaml
type_expansion:
  enabled: true
  exclude_types: [ThingWithTaxon, SubjectOfInvestigation, ...]  # 8 abstract types excluded by default
  max_depth: null  # Unlimited depth
  include_most_specific: true  # Always include most specific type
```

**Impact:**
- Matrix count: ~15-20x more matrices (depends on type diversity)
- More comprehensive rule mining (captures patterns at multiple abstraction levels)
- Slight memory increase per job (handled by tiering)

**CLI Usage:**
```bash
# Pre-build matrices with default config
uv run python scripts/prebuild_matrices.py \
  --edges edges.jsonl \
  --nodes nodes.jsonl \
  --output matrices

# Pre-build with custom config
uv run python scripts/prebuild_matrices.py \
  --edges edges.jsonl \
  --nodes nodes.jsonl \
  --output matrices \
  --config custom.yaml
```

**Note:** Type expansion configuration is baked into the pre-built matrices. All analysis scripts (`prepare_analysis.py`, `orchestrate_hop_analysis.py`, `analyze_hop_overlap.py`) use the configuration from the pre-built matrices.

### Per-Path OOM Recovery

The system tracks completion at **individual path granularity**, not just job-level:

**Tracking:**
- `results_Nhop/matrix1_XXX/.tracking/completed_paths.txt` - Successfully computed paths
- `results_Nhop/matrix1_XXX/.tracking/failed_paths.jsonl` - Failed paths with memory tier info
- `results_Nhop/matrix1_XXX/.tracking/path_in_progress.txt` - Currently computing path

**Retry Logic:**
1. Job completes some paths at 180GB, some OOM
2. Orchestrator detects partial completion via path tracking
3. Job retries at 250GB
4. Only failed paths attempted, completed paths skipped
5. Continues escalating tiers (250GB → 500GB → 1TB → 1.5TB)

**Result:** Maximum path completion despite memory constraints. Jobs can complete with partial results at max tier.

### Node Revisiting

**Note:** The current implementation does NOT filter out paths with repeated nodes. Path counts include all paths regardless of whether nodes are revisited (e.g., `A → B → A → C` is counted). This is intentional because:

1. Matrix-based diagonal zeroing can only prevent revisiting the **start** node, not intermediate nodes
2. Properly filtering all repeated nodes would require path enumeration, which is computationally expensive
3. For rule mining purposes, the statistical signal from repeated-node paths is often still useful

If distinct-node paths are required, post-processing filters can be applied.

### Duplicate Elimination

Each path can be computed from either end (A→B→C→D or D→C→B→A). The system eliminates duplicates by only computing from one direction:

**Rule:** Only process path if `M3.nvals >= M1.nvals`

**Example:**
- Matrix1 has 1,000 edges (M1.nvals = 1000)
- Matrix3 has 5,000 edges (M3.nvals = 5000)
- Path A→B→C→D: When M1=A, M3=D → 5000 >= 1000 ✓ → Compute
- Path D→C→B→A: When M1=D, M3=A → 1000 >= 5000 ✗ → Skip

**Result:** Each unique path computed exactly once (~2x speedup)

### 1-Hop Job Filtering

For 1-hop analysis, duplicate elimination uses alphabetical ordering instead of size-based comparison (since there's only one matrix). `prepare_analysis.py` pre-filters the job list to only include canonical directions:

**Rule:** Only create jobs where `src_type <= tgt_type` (alphabetically)

**Example:**
- `Gene|affects|Protein`: Gene < Protein alphabetically → Job created
- `Protein|affects|Gene`: Protein > Gene alphabetically → Job skipped (reverse will be handled by first job)

**Result:** ~50% fewer jobs for 1-hop analysis, no "0 paths completed" jobs

### Optimized Matrix Loading

Scripts only load matrices actually needed for the analysis:

1. **Path Enumeration Simulation:** `determine_needed_matrices()` simulates the recursive path-building logic on metadata (without loading matrices) to determine which base triples can participate
2. **Filtered Loading:** Only loads matrices in the needed set
3. **Type Constraints:** Matrices are filtered based on type compatibility for N-hop chains

**Example for 1-hop with matrix1_index=42:**
- Reads job manifest to get `(src_type, pred, direction, tgt_type)` for job 42
- Determines all matrices with matching `(src_type, tgt_type)` for comparison
- Loads only those matrices instead of all 21,815

**Result:** Faster startup, lower memory usage per job

### SLURM Memory Tiering

Jobs start at 180GB. On OOM failure, orchestrator:
1. Detects exit code 137 or SLURM state "OUT_OF_MEMORY"
2. Increments memory tier: 180GB → 250GB → 500GB → 1TB → 1.5TB
3. Resets job status to "pending"
4. Resubmits automatically

If job fails at 1.5TB, it's marked permanently failed for manual investigation.

**Rationale:** Starting at 180GB allows jobs to use most cluster nodes (~191GB RAM), maximizing parallelism. Higher tiers are used only when needed.

### Performance

**Sequential (original):** ~5 days
**Parallel (with ~100-200 concurrent SLURM jobs):** ~2-4 hours
**Speedup:** ~50-100x

## Testing

**Fast tests (for development):**
```bash
uv run pytest  # Run all tests
```

**Integration tests:**
Add `@pytest.mark.slow` decorator for tests that use real data files or make network calls.

## Library Usage

The package can also be used as a library for programmatic access:

```python
from metapath_counts import get_most_specific_type, get_symmetric_predicates

# Get most specific biolink type from a list
categories = ["biolink:ChemicalEntity", "biolink:SmallMolecule"]
specific = get_most_specific_type(categories)  # Returns "biolink:SmallMolecule"

# Check if predicate is symmetric
symmetric = get_symmetric_predicates()
if "directly_physically_interacts_with" in symmetric:
    print("Symmetric predicate - use direction 'A'")
```

## ***RULES OF THE ROAD***

- Ask clarifying questions

- Don't make classes just to group code. It is non-pythonic and hard to test.

- Do not implement bandaids - treat the root cause of problems

- Don't use try/except as a way to hide problems. It is often good just to let something fail and figure out why.

- Once we have a test, do not delete it without explicit permission.

- Do not return made up results if an API fails. Let it fail.

- If you cannot access a file, do not just guess at its contents - come back and ask for help

- When changing code, don't make duplicate functions - just change the function. We can always roll back changes if needed.

- Keep the directories clean, don't leave a bunch of junk laying around.

- When making pull requests, NEVER ever mention a `co-authored-by` or similar aspects. In particular, never mention the tool used to create the commit message or PR.

- Check git status before commits
- Use tsv not csv.
- For visualizations, output PNG only. Never create PDF files.
