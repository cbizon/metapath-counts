# Pass B Performance Notes

## Summary
Pass B (SLURM precompute for aggregated counts) is slow because the metapath expansion step is combinatorial. Caching type and predicate expansions does not materially improve runtime.

## What Pass B Does
Each Pass B task:
- Loads one shard of `explicit_counts` (list of `(path, count)` pairs).
- Calls `expand_metapath_to_variants(path)` for each explicit path.
- Accumulates `variant_path -> count` into a local dict.
- Writes `results_{n}hop/_tmp_prepare_grouping/agg_XXXXX.pkl`.

Reduce B then merges all `agg_*.pkl` shards into a single `aggregated_nhop_counts.json`.

## Observations
- Variant counts are very large even on a small sample:
  - Sample size: 200 paths
  - Variants per path: min 21, max 563,040, avg 40,521
- Explicit paths are numerous:
  - `num_explicit_paths`: 31,150,220 (3-hop)
  - `passb_shards`: 512
  - Avg paths per shard: ~60,842
- `agg_*.pkl` shards are huge (11G to 29G observed), so Reduce B is heavy in memory and I/O.

## Caching Experiment
We benchmarked a cached version of expansion (caching type and predicate variant lists) vs current expansion:

- Baseline: 19.22s
- Cached: 18.31s
- Speedup: ~1.05x

Conclusion: caching type/predicate expansions is not a meaningful improvement. The cost is dominated by the combinatorial expansion itself, not ancestor lookup.

## Next Directions to Explore
- Reduce the number of expansions (prune or constrain variant generation without changing semantics).
- Change representation to avoid constructing the full set in memory.
- Reconsider aggregation semantics or move parts of aggregation to later stages.

## Idea: Bubble-Up Aggregation via Child->Parent Map
Concept:
- Precompute a graph of type/predicate child->parent relationships (many-to-many).
- For each explicit path count, insert once at the leaf; then propagate upward through parents instead of generating full Cartesian products.

Potential Pros:
- Avoids full combinatorial expansion per path.
- Might allow incremental aggregation without building huge variant sets per path.

Risks/Costs:
- Building the propagation structure can be expensive, especially single-threaded.
- Update cost could still be large if propagation breadth is high.
- Needs careful handling for symmetry and canonicalization semantics to match current output.

Status:
- Proposed, not implemented. Needs feasibility check against current expansion semantics.

## Experiment: Edge-Pair DP Reuse
Goal:
- Reuse expansions of the first two edges in a 3-hop path and stitch with the last edge.
- Hypothesis: reduce repeated expansion work via cached edge-pair variants.

Method:
- Cache expansions for `(node0, pred0, dir0, node1, pred1, dir1, node2)` signatures.
- For each path, expand the last edge separately and stitch with cached edge-pair variants.
- Benchmark on 200 paths from `explicit_counts_shard_00000.pkl`.

Result:
- Baseline expansion: 19.28s
- DP edge-pair expansion: 22.13s
- Speedup: 0.87x (slower)

Conclusion:
- Edge-pair reuse did not help. Overhead of stitching and canonicalization outweighed any reuse benefits.
- Full-path signatures are unique, and partial reuse does not reduce combinatorial output size.

## 1-Hop DAG Build (Biolink-Only vs Data-Filtered)
We implemented `src/pipeline/build_onehop_dag.py` to build a 1-hop DAG from Biolink constraints, with optional data filtering.

Behavior:
- Uses Biolink domain/range constraints to permit (type, predicate, type) triples.
- Expands predicate variants (including qualifiers) using `get_predicate_variants`.
- Emits:
  - non-symmetric paths in both traversal directions (`F` and `R`)
  - symmetric paths in both endpoint orders with direction `A`
- Edges point to immediate parents (types/predicates) and preserve reverse/symmetric forms.
- Optional data filtering:
  - Current behavior is data-driven by observed `(src_type, predicate, tgt_type)` triples from KGX (`edges.jsonl` + `nodes.jsonl`).
  - For each observed triple, predicate ancestors and type ancestors are added.
  - This avoids the earlier overgeneration caused by predicate-only/type-only filtering.
- Uses dedup for nodes and edges.
- Data-filtered triple expansion now de-duplicates observed triples before ancestor expansion (large runtime improvement on big edge files).

Outputs:
- `nodes.tsv` with `metapath` column: `SrcType|predicate|dir|TgtType`
- `edges.tsv` with `child\tparent` edges (immediate parent types/predicates)
- `provenance.json`

Observed sizes:
- Filtered DAG (data-constrained):
  - nodes: 161,119
  - edges: 590,003
- Full Biolink-only DAG:
  - nodes: 43,302,598
  - edges: 164,501,745

Notes:
- Filtered DAG is tractable; full DAG is large but still manageable on disk.
- Higher-hop DAGs will grow combinatorially.

## Multihop DAG Build (Join Composition of N-Hop + 1-Hop)
We implemented `src/pipeline/build_multihop_dag.py` to compose an `N`-hop DAG with a 1-hop DAG into an `(N+1)`-hop DAG.

Definition:
- Nodes are legal concatenations where `end(N-hop) == start(1-hop)`.
- Edges are direct-parent edges in the combined DAG:
  - N-side parent only (join unchanged)
  - 1-hop-side parent only (join unchanged)
  - paired parent change on the join node (both sides change to the same join parent)

Current behavior:
- Inputs are directories containing `nodes.tsv` and `edges.tsv`.
- Outputs are `nodes.tsv`, `edges.tsv`, and `provenance.json`.
- Node and edge dedup are on by default.
- Includes shard-level profiling (counts and timing breakdowns).

Performance notes:
- A single-process 2-hop build ran out of memory due to large in-memory parent edge maps and dedup sets.
- Join-type sharding (`--shard-by-join`) fixed the dedup memory blow-up by scoping dedup to a shard.
- Additional optimization reduced parent-loop parsing overhead by using string slicing/cached start/end types and string concatenation for parent metapaths.

## SLURM Shard Workflow for Multihop DAGs
To handle large DAG2/DAG3 builds, we added a shard-job workflow:

- `scripts/submit_multihop_shard_jobs.sh`
  - prepares join-type shards if needed
  - submits one SLURM job per join shard
  - waits for all shard jobs to finish
  - prints periodic progress
  - reports failed shards with join type, state, exit code, shard path
  - automatically retries `OUT_OF_MEMORY` shards at higher memory tiers
- `scripts/merge_multihop_shard_outputs.sh`
  - validates all shard outputs exist
  - merges shard `nodes.tsv` / `edges.tsv` into final outputs
- `scripts/build_dags_feb13.sh`
  - wrapper to run DAG1, DAG2, DAG3 end-to-end using the shard workflow for multihop stages

Memory retry behavior (shard jobs):
- Base memory is script-configurable (e.g., DAG3 uses 128G in `build_dags_feb13.sh`).
- Heavy join types (`Entity`, `NamedThing`, `PhysicalEssence`) start at 2x the base memory.
- OOM retries escalate through tiers (`64, 128, 256, 500, 1000, 1400` GB).
- Jobs >1000 GB on `lowpri` are resubmitted to `largemem`.

## Current DAG Direction (Compression)
The current DAG approach still includes hierarchy levels that are filtered out elsewhere in the pipeline, which increases DAG size.

Why this happens:
- DAG edges currently preserve immediate-parent semantics.
- If filtered hierarchy nodes are simply removed, parent chains can break.

Current direction:
- Build a compressed DAG on the fly where edges point to the nearest allowed parent (skipping filtered hierarchy levels).
- This preserves reachability while reducing node/edge counts earlier, before multihop expansion.
- See `docs/DAG_COMPRESSION_PLAN.md` for the implementation plan.
