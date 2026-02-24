# On-The-Fly DAG Compression Plan

## Goal
Build smaller 1-hop and multihop DAGs by skipping filtered hierarchy levels during DAG construction, while preserving reachability for inference.

## Semantics
- Compressed DAG edges mean `nearest allowed parent`, not immediate ontology parent.
- Filtered types/predicates are not emitted as DAG nodes.
- Inference over the compressed DAG should preserve reachability among retained nodes.

## Scope
- Apply the same compression semantics to:
  - `src/pipeline/build_onehop_dag.py`
  - `src/pipeline/build_multihop_dag.py`
- Compression should be driven by the same excluded type/predicate sets used elsewhere in the pipeline.

## Plan

### 1. Define compression semantics and provenance
- Record in `provenance.json` that edges are compressed (`nearest allowed parent`).
- Record excluded type/predicate inputs and normalization rules used for compression.

### 2. Standardize filter inputs
- Decide shared CLI inputs for excluded types and excluded predicates.
- Normalize names consistently across builders (`normalize_type`, `normalize_predicate`).
- Ensure onehop and multihop interpret the same filter sets identically.

### 3. Implement shared compressed-parent helpers
- Add shared memoized helpers:
  - `get_allowed_type_parents(type_name, excluded_types)`
  - `get_allowed_predicate_parents(predicate, excluded_predicates)`
- Behavior:
  - If immediate parent is allowed, return it.
  - If immediate parent is filtered, recurse upward until nearest allowed ancestors are found.
  - Return a de-duplicated set/list of nearest allowed ancestors.

### 4. Integrate into 1-hop DAG construction
- Node generation:
  - Skip nodes containing filtered types/predicates.
- Edge generation:
  - Replace immediate type/predicate parent lookups with compressed-parent helpers.
  - Preserve reverse and symmetric path behavior.
  - Preserve dedup.
- Result:
  - Smaller base DAG before any multihop expansion.

### 5. Integrate into multihop DAG construction
- Primary path:
  - Build multihop DAGs from compressed onehop inputs, inheriting compressed semantics.
- Robustness path:
  - Ensure multihop builder does not emit filtered nodes if given mixed or legacy inputs.
  - Preserve compressed semantics for:
    - N-side parent edges
    - 1-hop-side parent edges
    - paired join-parent edges

### 6. Validate reachability and measure impact
- Add tests on small fixtures:
  - filtered intermediate type chain is skipped but reachability is preserved
  - filtered predicate chain is skipped but reachability is preserved
  - paired join-parent case under compression
- Compare uncompressed vs compressed:
  - 1-hop DAG node/edge counts
  - 2-hop/3-hop shard sizes
  - shard memory usage / OOM frequency

## Notes
- Post-build DAG contraction is semantically clean but does not solve peak memory/work because it still materializes the larger DAG first.
- The objective here is direct construction of the compressed DAG to reduce peak cost.
