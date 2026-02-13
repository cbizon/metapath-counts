# Fix: Path Counting for Same-Type Endpoints

## Status: IMPLEMENTED

Branch: `improve_e2et`
Relevant test: `tests/test_end_to_end/test_pipeline_2hop.py::TestGrouped2HopSpecificPaths::test_gene_product_affects_biological_affects_gene_product_predicts_interacts`

## The Test

This test checks that `GeneOrGeneProduct|affects|F|BiologicalEntity|affects|R|GeneOrGeneProduct`
predicts `GeneOrGeneProduct|interacts_with|A|GeneOrGeneProduct`.

**Verified values:**
- `predictor_count = 5`
- `overlap = 3`
- `predicted_count = 4`

## The Problem

When an N-hop path has the same source and target type (e.g., `Gene|affects|F|Disease|affects|R|Gene`),
the accumulated matrix can contain two kinds of spurious entries:

1. **Self-pairs (diagonal):** A node reaching itself through intermediate hops (e.g., Gene_A → Disease_P → Gene_A). These are meaningless as predictions.

2. **Symmetric duplicates (palindromic paths only):** When a path is palindromic (reads the same
   forward and backward), the accumulated matrix is symmetric — (A,B) and (B,A) both exist for
   every pair. This double-counts undirected pairs, inconsistent with how `should_process_path`
   handles non-palindromic path deduplication at the path level.

## What Is a Palindromic Path?

A path is palindromic when it equals its own reverse. The reverse is computed using the same
pattern as `canonicalize_metapath` in `aggregation.py`: reverse nodes, reverse predicates,
flip F↔R directions (symmetric predicates use effective direction 'A', and opposite('A') = 'A').

**Palindromic examples:**
- `Gene|affects|F|Disease|affects|R|Gene` — same predicate, F mirrors R
- `Gene|interacts_with|A|Protein|interacts_with|A|Gene` — symmetric predicates, A mirrors A
- 1-hop: `Gene|interacts_with|A|Gene` — symmetric, same types

**NOT palindromic:**
- `Gene|regulates|F|Gene|interacts_with|A|Gene` — different predicates
- 1-hop: `Gene|regulates|F|Gene` — F ≠ opposite(F) = R
- 3-hop with non-symmetric middle: the middle predicate of an odd-length path must be
  symmetric (eff_dir must satisfy eff_dir == opposite(eff_dir), only true for 'A')

**Why this distinction matters:** Non-palindromic same-type paths have asymmetric accumulated
matrices — entries may exist only in the lower triangle with no upper-triangle counterpart.
Applying upper-triangle filtering to these would discard valid pairs.

## What Changed

In `scripts/analyze_hop_overlap.py`:

### New function: `is_palindromic_path()`

Checks if a path equals its reverse using the same reversal logic as `canonicalize_metapath`:

```python
def is_palindromic_path(node_types, predicates, directions, symmetric_predicates):
    eff_dirs = ['A' if predicates[i] in symmetric_predicates else directions[i]
                for i in range(len(predicates))]
    rev_nodes = list(reversed(node_types))
    rev_preds = list(reversed(predicates))
    rev_dirs = [{'F': 'R', 'R': 'F', 'A': 'A'}[d] for d in reversed(eff_dirs)]
    return (list(node_types) == rev_nodes
            and list(predicates) == rev_preds
            and eff_dirs == rev_dirs)
```

### Modified: `process_path()` at `depth == n_hops`

Before computing `nhop_count` and `overlap`:

```python
work_matrix = accumulated_matrix
if src_type_final == tgt_type_final:
    if is_palindromic_path(node_types, predicates, directions, symmetric_predicates):
        # Palindromic: symmetric matrix. triu(k=1) removes diagonal + lower triangle.
        work_matrix = accumulated_matrix.select(gb.select.triu, 1).new()
    else:
        # Non-palindromic same-type: only remove self-pairs (diagonal)
        work_matrix = accumulated_matrix.select(gb.select.offdiag).new()
    if work_matrix.nvals == 0:
        return
```

Then `work_matrix` is used for both `nhop_count` and `overlap_matrix` computation.
The 1-hop comparison matrix (`onehop_matrix`) is NOT modified.

## Why predictor_count = 5

The aggregated predictor `GeneOrGeneProduct|affects|F|BiologicalEntity|affects|R|GeneOrGeneProduct`
sums three explicit 2-hop paths from the golden graph:

### 1. `Gene|affects|F|Disease|affects|R|Gene` (palindromic: count=1)

Matrix product (Gene×Gene):
- (A,A): self-pair → removed by triu(k=1)
- (A,B): upper triangle → kept
- (B,A): lower triangle → removed (symmetric duplicate of (A,B))
- (B,B): self-pair → removed by triu(k=1)

Count after dedup: **1**

### 2. `Protein|affects|F|Disease|affects|R|Gene` (count=2)

src_type=Protein ≠ tgt_type=Gene, no deduplication needed. Count: **2**

### 3. `Gene+Protein|affects|F|Disease|affects|R|Gene` (count=2)

src_type=Gene+Protein ≠ tgt_type=Gene, no deduplication needed. Count: **2**

### Total: 1 + 2 + 2 = **5**

## Why overlap=3

Aggregated sum from raw rows (each contributing 1):
- `Protein|...|Gene` vs `Protein|interacts_with|A|Gene`: pair (M,A) in both → 1
- `Gene+Protein|...|Gene` vs `Gene+Protein|interacts_with|A|Gene`: pair (Z,B) in both → 1
- `Gene|...|Gene` vs `Gene|regulates|F|Gene`: pair (A,B) in both → 1
  (regulates is a descendant of interacts_with in biolink predicate hierarchy)

After the fix, the Gene path's work_matrix has only (A,B) in the upper triangle. This
matches `Gene|regulates|F|Gene` which has (A,B). So overlap stays 1 for this path.

## Why predicted_count=4

The aggregated `GeneOrGeneProduct|interacts_with|A|GeneOrGeneProduct` counts:
- `Protein|interacts_with|A|Gene` (count=2)
- `Gene+Protein|interacts_with|A|Gene` (count=1)
- `Gene|regulates|F|Gene` (count=1, via predicate hierarchy)

Total = **4**. These 1-hop matrices are not modified by the fix.

## Golden Graph Reference

Nodes: Gene_A (Gene), Gene_B (Gene), Protein_M (Protein), Protein_N (Protein),
       GeneProtein_Z (Gene+Protein), Disease_P (Disease), Disease_Q (Disease),
       SmallMolecule_X (SmallMolecule), SmallMolecule_Y (SmallMolecule)

Affects edges (8): Gene_A→Disease_P, Gene_A→Disease_Q, Gene_B→Disease_P, Gene_B→Disease_Q,
                   Protein_M→Disease_P, GeneProtein_Z→Disease_Q,
                   SmallMolecule_X→Gene_A, SmallMolecule_Y→Gene_B

interacts_with edges (3, symmetric): Gene_A↔Protein_M, Gene_B↔Protein_N, GeneProtein_Z↔Gene_B

regulates edge (1): Gene_A→Gene_B
