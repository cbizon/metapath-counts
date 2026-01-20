# Diagonal Zeroing in 3-Hop Path Analysis

## Overview

The 3-hop path analysis implements "diagonal zeroing" at multiple stages to reduce certain types of repeated nodes in paths. However, it's important to understand both what this approach **does** and **does not** accomplish.

## What Diagonal Zeroing Does

Diagonal zeroing removes entries on the diagonal of square matrices (where row index == column index). In our path analysis, this occurs at three levels:

### Level 1: Input Edge Filtering
```python
# Removes self-loops from input edges (A→A)
if nrows == ncols:
    matrix = matrix.select(gb.select.offdiag).new()
```
**Effect:** Prevents paths that contain single-edge self-loops (e.g., Drug_A → Drug_A)

### Level 2: After A @ B (2-hop intermediate)
```python
# Prevents paths where start node reappears at position 2
if result_AB.nrows == result_AB.ncols:
    result_AB = result_AB.select(gb.select.offdiag).new()
```
**Effect:** For a path Node_i → ... → Node_j, prevents Node_i from reappearing as Node_j in the 2-hop result

### Level 3: After (A @ B) @ C (3-hop final)
```python
# Prevents paths where start node reappears at end
if result_ABC.nrows == result_ABC.ncols:
    result_ABC = result_ABC.select(gb.select.offdiag).new()
```
**Effect:** For a 3-hop path, prevents the start node from being the same as the end node (only if start and end types match)

## What Diagonal Zeroing Does NOT Do

**Diagonal zeroing does NOT compute simple paths** (paths with no repeated nodes).

### Example of Undetected Repetition

Consider the path: Drug_A → Disease_X → Drug_B → Disease_X

This path has Disease_X appearing twice (positions 1 and 3), but:
- ✓ Level 1 doesn't catch it (not a self-loop edge)
- ✓ Level 2 doesn't catch it (checks position 0 vs 2, not 1 vs 3)
- ✓ Level 3 doesn't catch it (checks position 0 vs 3, which are different nodes)

**This path is counted** even though Disease_X repeats.

### Why Complete Simple Path Detection is Infeasible

Computing the number of **simple paths** (no repeated nodes) between all pairs of nodes is **NP-hard**. The matrix multiplication approach we use inherently loses information about intermediate nodes, making it impossible to detect all repetitions without exponential computational complexity.

## Direction-Dependent Filtering

**CRITICAL:** Diagonal zeroing produces **different results** depending on traversal direction.

### Example: Drug ↔ Disease paths

**Forward Direction** (starting from Drug):
- Path: Drug_A → Disease_X → Drug_A → Disease_Y
- Level 2 removes this (Drug_A at positions 0 and 2)
- **Count: 4,643,717**

**Backward Direction** (starting from Disease):
- Same path reversed: Disease_Y → Drug_A → Disease_X → Drug_A
- Level 2 does NOT remove this (Disease_Y ≠ Disease_X at positions 0 and 2)
- **Count: 4,644,230**

The 513-path difference arises because each direction filters out paths that loop back to **its specific starting node type**.

## When to Use --no-zeroing

Use the `--no-zeroing` flag when you want to count **all paths** between nodes, including:
- Paths that return to the starting node
- Paths with repeated intermediate nodes
- Maximum path counts without filtering

**Tradeoffs:**
- ✓ Counts are symmetric (forward == reverse)
- ✓ Easier to interpret (pure path counts)
- ✗ Includes many "low quality" paths with repetitions
- ✗ Higher path counts

## Implementation Details

### Code Locations

Diagonal zeroing is implemented in `analyze_3hop_overlap.py`:
- Line ~196: Level 1 (input matrices)
- Line ~350: Level 2 (after A @ B)
- Line ~370: Level 3 (after A @ B @ C)

### Conditional Execution

All three levels can be disabled with the `--no-zeroing` flag:
```bash
uv run python analyze_3hop_overlap.py --no-zeroing
```

## Recommendations

1. **For exploratory analysis:** Use default (with zeroing) to reduce low-quality paths
2. **For symmetric counts:** Use `--no-zeroing` to ensure forward/reverse agreement
3. **For publication:** Clearly document which mode was used and its implications
4. **For filtering comparisons:** Consider running both modes to understand impact

## Summary

| Aspect | With Zeroing | Without Zeroing |
|--------|-------------|-----------------|
| Prevents start→...→start | ✓ Yes | ✗ No |
| Prevents all repetitions | ✗ No | ✗ No |
| Forward == Reverse counts | ✗ No | ✓ Yes |
| Simple path guarantee | ✗ No | ✗ No |
| Reduces low-quality paths | ✓ Partial | ✗ No |
