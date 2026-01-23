# Plan: Hierarchical Type Expansion

## Current System Analysis

### Node Type Assignment
- **Current**: Each node gets ONE type via `get_most_specific_type()`
- Example: Node with `[SmallMolecule, ChemicalEntity, NamedThing]` → `SmallMolecule` only
- Location: `load_node_types()` in `analyze_hop_overlap.py:135-155`

### Matrix Building
- **Current**: Each edge appears in exactly ONE matrix
- Example: `Drug1(SmallMolecule) -treats-> Disease1` → `SmallMolecule|treats|Disease` matrix only
- Location: `build_matrices()` in `analyze_hop_overlap.py:158-244`

### Current Scope
- If graph has P predicates and T types:
  - Number of matrices ≈ P × T² (one type per node)
  - Each node participates in matrices for its single assigned type

## Proposed System

### Node Type Assignment
- **Proposed**: Each node gets ALL its types from the hierarchy
- Example: Same node → participates as `SmallMolecule`, `ChemicalEntity`, AND `NamedThing`
- Implementation: Return list of types instead of single type

### Matrix Building
- **Proposed**: Each edge appears in MULTIPLE matrices (one per type combination)
- Example: Same edge appears in:
  - `SmallMolecule|treats|Disease`
  - `ChemicalEntity|treats|Disease`
  - `NamedThing|treats|Disease` (maybe filter very abstract types?)
  - Plus all combinations if target also has multiple types

### New Scope
- Number of matrices ≈ P × T² × (avg_types_per_node)²
- If average node has 3 types: **~9× more matrices**
- Each matrix is larger (more nodes qualify for broader types)

## Impact Analysis

### 1. Matrix Explosion
**Problem**: Combinatorial explosion of matrices
- Current: ~180 matrices (estimated for biolink graph)
- Proposed: ~1,600+ matrices (if avg 3 types per node)

**Mitigation**:
- Filter out very abstract types (e.g., `NamedThing`, `Entity`)
- Only expand up to certain level in hierarchy (e.g., max 2-3 ancestors)
- Add configuration for which types to expand

### 2. Increased Matrix Size
**Problem**: Broader type categories = more nodes per matrix
- `ChemicalEntity` matrix includes all `SmallMolecule`, `Drug`, `Metabolite` nodes
- Can be 10-100× larger than specific type matrices

**Mitigation**:
- Memory estimation per matrix (predict before building)
- Skip matrices predicted to exceed memory limits
- Log skipped matrices for later processing on larger nodes

### 3. Computational Cost
**Problem**: More starting points × larger matrices = exponentially more work
- More matrix1 choices for parallel jobs
- Larger matrix multiplications (N-hop path building)
- More comparisons in terminal step

**Mitigation**:
- Better job scheduling (prioritize by predicted cost)
- Time limits per individual path computation
- Checkpoint progress within jobs

### 4. Job Failure Modes
**Problem**: Current system kills entire SLURM job if ANY path fails
- All-or-nothing: one OOM path → lose all results from that job
- Expensive retry on larger nodes

**Need**: Graceful degradation within jobs

## Implementation Plan

### Phase 1: Hierarchical Type Support (Core Changes)

#### 1.1 Update Type Assignment
**File**: `type_utils.py`
- Add `get_all_types(categories: list[str]) -> list[str]`
  - Returns full hierarchy for a node
  - Option to filter by max depth or exclude abstract types
- Add `filter_abstract_types(types: list[str]) -> list[str]`
  - Remove types like `NamedThing`, `Entity` (configurable)

**File**: `analyze_hop_overlap.py`
- Modify `load_node_types()` to return `dict[node_id, list[str]]`
- Keep backward compatibility option for single type

#### 1.2 Update Matrix Building
**File**: `analyze_hop_overlap.py`
- Modify `build_matrices()` to handle nodes with multiple types
- For each edge `(subject, predicate, object)`:
  - Get all types for subject: `[TypeA1, TypeA2, ...]`
  - Get all types for object: `[TypeB1, TypeB2, ...]`
  - Create matrices for all combinations: `TypeAi|predicate|TypeBj`
  - Node indexing per type (subject may have different indices in different type matrices)

**Challenges**:
- Node indexing: A node has different index in `SmallMolecule` vs `ChemicalEntity` matrix
- Same node pair may appear in multiple matrices
- Need to track which nodes belong to which type matrices

### Phase 2: Error Handling & Graceful Degradation

#### 2.1 Memory Prediction
**New File**: `scripts/memory_estimator.py`
- Estimate memory for matrix multiplication before executing
- Formula: `memory ≈ (nnz1 + nnz2 + nnz_result) × bytes_per_element`
- Heuristic for result sparsity based on input patterns
- Skip paths predicted to exceed available memory

#### 2.2 Per-Path Error Handling
**File**: `analyze_hop_overlap.py`
- Wrap path computation in try/except at appropriate granularity
- Current granularity: entire matrix1 job (all paths from one starting matrix)
- Proposed granularity: individual N-hop path computation

**Implementation**:
```python
def process_path(...):
    try:
        # Existing path building logic
        ...
    except (MemoryError, gb.exceptions.OutOfMemory):
        # Log failure but continue
        log_skipped_path(reason="OOM", ...)
        return
    except Exception as e:
        # Log unexpected errors
        log_skipped_path(reason=str(e), ...)
        return
```

#### 2.3 Skipped Path Tracking
**New File**: `results_{N}hop/skipped_paths_matrix1_{XXX}.jsonl`
- JSON Lines format for each skipped path:
  ```json
  {
    "matrix1": "SmallMolecule|treats|Disease",
    "depth": 2,
    "accumulated_types": ["SmallMolecule", "Gene"],
    "next_matrix": "Gene|associated_with|Disease",
    "reason": "OOM",
    "estimated_memory_gb": 450,
    "timestamp": "..."
  }
  ```

#### 2.4 Partial Results
- Job writes results incrementally (already doing this with file writes)
- Even if job fails later, early results are preserved
- Manifest tracks: `completed_paths`, `failed_paths`, `skipped_paths`

### Phase 3: Job Management Improvements

#### 3.1 Enhanced Manifest
**File**: `prepare_analysis.py`
- Add per-job metadata:
  ```json
  "matrix1_042": {
    "status": "completed",
    "matrix_info": {
      "src_type": "SmallMolecule",
      "pred": "treats",
      "tgt_type": "Disease",
      "nvals": 15000,
      "estimated_memory_gb": 180
    },
    "results": {
      "paths_computed": 2500,
      "paths_skipped": 150,
      "reason_counts": {"OOM": 120, "timeout": 30}
    }
  }
  ```

#### 3.2 Smarter Job Scheduling
**File**: `orchestrate_hop_analysis.py`
- Sort jobs by estimated cost (smallest first)
- Start with jobs most likely to succeed
- Defer large jobs until more resources available
- Option to skip jobs predicted to be infeasible

#### 3.3 Timeout per Path
- Add configurable timeout for individual path computation
- Prevents one infinite loop from blocking progress
- Log timeout events for analysis

### Phase 4: Configuration & Controls

#### 4.1 Configuration File
**New File**: `config/type_expansion.yaml`
```yaml
type_expansion:
  enabled: true
  max_depth: 2  # How far up hierarchy to expand
  exclude_types:  # Types to skip
    - NamedThing
    - Entity
    - PhysicalEssenceOrOccurrent
  include_most_specific: true  # Always include leaf type

error_handling:
  skip_on_oom: true
  max_memory_gb: 1400  # Don't attempt if predicted >1.4TB
  timeout_per_path_sec: 300  # 5 min per path

output:
  write_skipped_paths: true
  checkpoint_frequency: 1000  # Flush every N paths
```

#### 4.2 CLI Options
**All scripts**: Add `--config` option
```bash
uv run python scripts/prepare_analysis.py \
  --edges edges.jsonl \
  --nodes nodes.jsonl \
  --n-hops 3 \
  --config config/type_expansion.yaml
```

### Phase 5: Testing Strategy

#### 5.1 Unit Tests
- Test `get_all_types()` with various hierarchies
- Test matrix building with multi-type nodes
- Test error handling (mock OOM conditions)
- Test skipped path logging

#### 5.2 Integration Tests
- Small synthetic graph with known type hierarchy
- Verify all type combinations produce matrices
- Verify graceful failure on resource constraints
- Verify partial results are usable

#### 5.3 Incremental Rollout
1. Run on small graph with type expansion (verify correctness)
2. Run on medium graph with error handling (verify robustness)
3. Run on full graph with monitoring (verify scalability)

## Migration Strategy

### Backward Compatibility
- Keep single-type mode as default (`type_expansion.enabled: false`)
- Existing workflows continue to work unchanged
- Opt-in to hierarchical mode via config

### Gradual Transition
1. **Phase 1**: Implement core type expansion (no error handling)
   - Test on small graphs only
   - Validate output quality

2. **Phase 2**: Add error handling
   - Test on medium graphs
   - Tune memory estimation heuristics

3. **Phase 3**: Add job management improvements
   - Test on full graphs
   - Monitor performance

4. **Phase 4**: Make hierarchical mode default
   - After validation on production data
   - Document breaking changes

## Success Metrics

### Correctness
- [ ] All type combinations are generated
- [ ] Matrix contents match expected edges
- [ ] No duplicate paths in output
- [ ] Skipped paths are properly logged

### Robustness
- [ ] Jobs complete with partial results on OOM
- [ ] Timeouts don't kill entire jobs
- [ ] Failed jobs can be retried
- [ ] Manifest accurately reflects job state

### Performance
- [ ] 90%+ of feasible paths complete successfully
- [ ] <10% of compute time wasted on infeasible paths
- [ ] Memory estimation accuracy >80%
- [ ] Job throughput remains acceptable

## Open Questions

1. **Type filtering**: What's the right cutoff for "too abstract"?
   - Need to analyze type distribution in actual graphs
   - Balance between coverage and computational cost

2. **Index management**: How to efficiently handle node indices across multiple type matrices?
   - Option 1: Global node→index, then filter per type
   - Option 2: Separate index space per type
   - Tradeoff: memory vs lookup complexity

3. **Result interpretation**: How to handle overlapping results?
   - Same path appears at multiple type granularities
   - Post-processing to deduplicate or aggregate?
   - Keep all levels for different use cases?

4. **Memory estimation**: What heuristics work best?
   - Depends heavily on graph structure
   - May need calibration phase on subset of jobs
   - Update estimates based on actual outcomes

## Timeline Estimate

- **Phase 1** (Core changes): 2-3 weeks
  - Type utilities: 2 days
  - Matrix building refactor: 1 week
  - Testing: 3 days
  - Buffer: 4 days

- **Phase 2** (Error handling): 1-2 weeks
  - Memory estimation: 3 days
  - Per-path try/except: 2 days
  - Skipped path tracking: 2 days
  - Testing: 3 days

- **Phase 3** (Job management): 1 week
  - Manifest enhancements: 2 days
  - Scheduling logic: 2 days
  - Testing: 3 days

- **Phase 4** (Config): 3 days
  - Config file structure: 1 day
  - CLI integration: 1 day
  - Documentation: 1 day

- **Phase 5** (Testing): 1-2 weeks
  - Unit tests: 3 days
  - Integration tests: 3 days
  - Production validation: 5 days

**Total**: 6-9 weeks for full implementation

## Risks & Mitigation

### Risk 1: Matrix explosion makes system unusable
**Likelihood**: Medium
**Impact**: High
**Mitigation**:
- Start with aggressive type filtering
- Implement memory prediction early
- Add kill switches in config

### Risk 2: Error handling doesn't catch all failure modes
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:
- Test with intentional resource constraints
- Monitor real jobs for unexpected failures
- Iterate on exception handling

### Risk 3: Performance degradation too severe
**Likelihood**: Low-Medium
**Impact**: High
**Mitigation**:
- Profile before/after
- Optimize hot paths
- Consider sampling strategies if needed

### Risk 4: Results too complex to interpret
**Likelihood**: Low
**Impact**: Medium
**Mitigation**:
- Document clearly what each level means
- Provide aggregation tools
- Validate with domain experts

## Next Steps (After Current Jobs Complete)

1. Review this plan with stakeholders
2. Decide on scope for initial implementation
3. Create feature branch for development
4. Start with Phase 1 (core type expansion)
5. Test on small graph before proceeding
