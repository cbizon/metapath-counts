# Phase 2 Implementation: Per-Path Tracking & OOM Recovery

## Summary

Implements per-path completion tracking to enable incremental computation with OOM recovery. When a job OOMs, we:
1. Never recompute completed paths
2. Skip paths that failed at the current memory tier
3. Retry failed paths at higher memory tiers
4. Handle branch failures by marking all downstream paths as failed

## Key Components Created

### 1. Path Tracker Module (`scripts/path_tracker.py`)

**Path Identification:**
- `generate_path_id()`: Create unique ID from node_types, predicates, directions
- Format: `Type1|pred1|dir1|Type2__Type2|pred2|dir2|Type3`
- Example: `SmallMolecule|treats|F|Disease__Disease|affects|F|Gene`

**Tracking Files per Matrix1 Job:**
- `completed_paths.txt`: One path ID per line (append-only)
- `failed_paths.jsonl`: Failed paths with memory tier and depth
- `path_in_progress.txt`: Current path being computed (overwritten)

**Key Functions:**
- `load_completed_paths()`: Load set of completed path IDs
- `load_failed_paths()`: Load failed paths at specific memory tier
- `record_completed_path()`: Append to completed_paths.txt
- `record_failed_path()`: Append to failed_paths.jsonl
- `record_path_in_progress()`: Overwrite with current path
- `enumerate_downstream_paths()`: Find all N-hop completions of partial path

### 2. Modified `analyze_hop_overlap.py`

**New Parameters:**
- `current_memory_gb=180`: Current SLURM memory tier
- `enable_path_tracking=True`: Toggle tracking (can disable for testing)

**Initialization:**
- Load completed_paths and failed_at_current_tier sets
- Report skip statistics

**Modified `process_path()` Function:**

**At Final Depth (depth == n_hops):**
```python
# 1. Generate path ID
path_id = generate_path_id(node_types, predicates, directions)

# 2. Check if already done
if path_id in completed_paths:
    return  # Skip, already have results

# 3. Check if failed at this tier
if path_id in failed_at_current_tier:
    return  # Skip, will retry at higher tier

# 4. Record in progress
if enable_path_tracking:
    record_path_in_progress(path_id, results_dir, matrix1_index, n_hops, current_memory_gb)

# 5. Compute overlap (existing code)
# ... calculate overlaps with 1-hop matrices ...

# 6. Write results and FLUSH
f.write(result_row)
f.flush()  # CRITICAL: Flush after every path

# 7. Record completion
if enable_path_tracking:
    record_completed_path(path_id, results_dir, matrix1_index)
    clear_path_in_progress(results_dir, matrix1_index)
```

**At Intermediate Depth (depth < n_hops):**
```python
# Wrap matrix multiplication in try/except
for next_matrix in by_source_type[current_target_type]:
    try:
        # Record in-progress (partial path)
        partial_path_id = generate_path_id(node_types, predicates, directions)
        if enable_path_tracking:
            record_path_in_progress(partial_path_id, results_dir, matrix1_index,
                                   depth, current_memory_gb)

        # Multiply matrices (may OOM here)
        next_result = accumulated_matrix.mxm(matrix, gb.semiring.any_pair).new()

        if next_result.nvals == 0:
            continue

        # Recurse to next depth
        process_path(depth + 1, next_result, ...)

    except (MemoryError, Exception) as e:
        # OOM during intermediate multiplication
        # This means ALL downstream N-hop paths are impossible to compute

        if enable_path_tracking and "memory" in str(e).lower():
            # Enumerate all possible N-hop completions
            partial_path_id = generate_path_id(node_types, predicates, directions)
            downstream_paths = enumerate_downstream_paths(
                partial_path_id, all_matrices, n_hops, depth
            )

            # Mark all as failed at this tier
            for complete_path_id in downstream_paths:
                record_failed_path(complete_path_id, results_dir, matrix1_index,
                                 current_memory_gb, depth=depth, reason="branch_oom")

        # Continue to next matrix (don't crash entire job)
        continue
```

### 3. Orchestrator Changes (`scripts/orchestrate_hop_analysis.py`)

**On Job Completion (exit_code 137 - OOM):**
```python
def handle_oom_job(matrix1_id, memory_gb):
    # Read which path was in progress
    in_progress = read_path_in_progress(results_dir, matrix1_id)

    if in_progress:
        # Record as failed at this tier
        record_failed_path(
            in_progress['path_id'],
            results_dir,
            matrix1_id,
            memory_gb,
            depth=in_progress['depth'],
            reason="oom"
        )

    # Count paths
    stats = get_path_statistics(results_dir, matrix1_id)
    completed = stats['completed']
    failed_this_tier = stats['failed_by_tier'].get(memory_gb, 0)

    # Estimate remaining (total - completed - failed)
    # Note: Total may not be known exactly, estimate from manifest

    # Decide retry strategy
    if has_untried_paths:
        # Resubmit at SAME tier to finish untried paths
        submit_job(matrix1_id, memory_gb)

    if failed_this_tier > 0:
        # Resubmit at NEXT tier for failed paths
        next_tier = get_next_memory_tier(memory_gb)
        submit_job(matrix1_id, next_tier)
```

**Memory Tier Progression:**
- 180GB → 250GB → 500GB → 1TB → 1.5TB
- Same as before

### 4. Config Changes

**`config/type_expansion.yaml`:**
```yaml
output:
  checkpoint_frequency: "per_path"  # Flush after every path (was 10000 rows)
```

## Critical Design Decisions

### Branch Failure Handling

When intermediate multiplication OOMs at depth D < N:
1. Generate partial path ID (D hops so far)
2. Enumerate ALL possible N-hop completions
3. Mark each completion as failed at current tier
4. Continue with next matrix (don't crash)

**Example:**
- Depth 1, partial path: `SmallMolecule|treats|F|Disease`
- OOM during: `Disease|affects|F|Gene` multiplication
- Enumerate 100 possible Gene→* completions
- Mark all 100 as failed at 180GB
- Skip them on same-tier retry
- Attempt them at 250GB retry

### Flush Frequency

**Changed from 10,000 rows to per-path:**
- Old: `if rows_written % 10000 == 0: f.flush()`
- New: `f.write(result_row); f.flush()`

**Rationale:**
- One path may write 100+ result rows (comparing with all 1-hop matrices)
- If we flush per 10k rows, losing 100 rows = losing 1 path
- With hierarchical types (15-30x more paths), OOM more likely
- Per-path flush ensures completed work is NEVER lost

**Performance Impact:**
- Modern filesystems use write-back cache
- Flush ~1000 times = ~1ms overhead total
- Negligible compared to matrix multiplication cost

### Path ID Format

Using `__` as segment separator:
- `SmallMolecule|treats|F|Disease__Disease|affects|F|Gene`
- Easy to parse, no conflicts with `|` in metapath format
- Can split on `__` to get individual hops

## Testing Plan

### Unit Tests

1. **Path Tracker:**
   - Test path ID round-trip (generate → parse → verify)
   - Test 1-hop, 2-hop, 3-hop paths
   - Test completed/failed path loading
   - Test enumeration of downstream paths

2. **Modified analyze_hop_overlap:**
   - Test path skipping (completed paths not recomputed)
   - Test branch failure handling
   - Test flush frequency

### Integration Tests

1. **Simulated OOM:**
   - Kill job mid-run
   - Verify completed_paths.txt has entries
   - Verify path_in_progress.txt shows last path
   - Restart job, verify skips completed paths

2. **Multi-tier Retry:**
   - Run at 180GB, let some paths fail
   - Run at 180GB again, verify skips failed
   - Run at 250GB, verify retries only failed paths

3. **Branch Failure:**
   - Simulate OOM at depth=1
   - Verify all downstream paths marked as failed
   - Verify enumeration is complete

## Rollout Plan

### Phase 2a: Core Path Tracking (Current)
- ✅ Path tracker module
- ✅ Config changes (flush per path)
- 🔄 Modify analyze_hop_overlap.py
- 🔄 Unit tests

### Phase 2b: Orchestrator Integration
- Modify orchestrate_hop_analysis.py
- Handle OOM detection
- Implement retry logic
- Enhanced manifest tracking

### Phase 2c: Testing & Validation
- Integration tests on synthetic data
- Small production run (Matrix1 000-010)
- Verify path tracking works correctly
- Tune if needed

### Phase 2d: Full Deployment
- Run on full production graph
- Monitor completion rates
- Analyze failed paths
- Document patterns

## Success Metrics

- [ ] Completed paths never recomputed
- [ ] 90%+ of feasible paths complete successfully
- [ ] Failed paths properly tracked by memory tier
- [ ] Branch failures correctly enumerate downstream paths
- [ ] Multi-tier retry successfully recovers failed paths
- [ ] Flush frequency doesn't impact performance (<1% overhead)

## Open Questions

1. **Total path enumeration**: Should we pre-enumerate ALL possible N-hop paths upfront for progress tracking? Or estimate based on matrix count?
   - Pre-enumerate: Accurate but slow
   - Estimate: Fast but approximate

2. **Branch failure optimization**: Should we skip entire branches once one path fails at a given depth?
   - Pro: Avoid repeated enumerations
   - Con: Miss paths that might succeed with different matrix order

3. **Memory tier escalation**: Should we automatically escalate ALL remaining paths to next tier after N failures?
   - Current: Retry individually at same tier first
   - Alternative: After 10% failure rate, move all to next tier

## Next Steps

1. Complete modifications to analyze_hop_overlap.py
2. Write unit tests for path tracking
3. Modify orchestrator for multi-tier retry
4. Integration test on synthetic data
5. Small production validation run
