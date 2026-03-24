# Optimization Notes

## Implemented Grouping Optimizations

This section records the sequence of grouping hot-path fixes that are now in
the codebase. The main files are:
- [src/library/aggregation.py](/projects/sequence_analysis/vol3/bizon/sub/metapath-counts/src/library/aggregation.py)
- [src/pipeline/workers/run_grouping.py](/projects/sequence_analysis/vol3/bizon/sub/metapath-counts/src/pipeline/workers/run_grouping.py)

### 1. Early Target Filtering

Implemented:
- excluded 1-hop targets are filtered before candidate-building
- pseudo-type 1-hop targets are filtered before candidate-building

Why:
- the worker was spending hours building candidates for targets that would be
  discarded later during output

Effect:
- removes guaranteed wasted work
- does not change semantics

### 2. Ordered Predictor Processing

Implemented:
- per target, explicit predictors are processed in descending explicit count

Why:
- lower-bound precision pruning is monotone in predictor count accumulation
- large predictors are the ones most likely to trigger pruning early

Effect:
- improves early pruning behavior, especially at higher precision thresholds

### 3. Real Branch Cut-Off

Bug that existed:
- the traversal callback could signal prune, but traversal still continued in
  cases where it should have stopped descending

Implemented:
- `visit_variant(...) -> True` now actually stops descent from that lattice
  point

Effect:
- large throughput improvement versus the broken callback path

### 4. State-Level Pruning

Bug / limitation that existed:
- pruning only happened after an output-valid type-pair variant was materialized
- intermediate lattice states that were already too broad were still explored

Implemented:
- pruning now happens at lattice-state level before output-valid endpoint
  materialization
- state-level counts are aggregated and compared against the precision ceiling

Effect:
- corrects the pruning model
- makes pruning possible before endpoint validity

### 5. Compact State Identities

Problem that existed:
- state pruning used canonical metapath strings as keys
- that was expensive in both memory and CPU

Implemented:
- state identities are now stored as compact integer-tuples derived from token
  IDs rather than full strings

Effect:
- lowers RSS substantially
- removes repeated tuple -> string -> hash churn in the hot loop

### 6. Cross-Target Pruned-State Carryover

Implemented:
- targets are processed in descending target-count order
- globally pruned lattice states are carried forward across targets

Why this is valid:
- if a state is too broad for a larger target count `N1`, it is also too broad
  for any later smaller target count `N2 < N1`

Effect:
- enables monotone reuse of prune facts across targets
- benefit only appears once a run reaches target 2 and later

### 7. Direct Endpoint Promotion

Problem that existed:
- type-pair traversal still crawled endpoint type hierarchies upward until the
  endpoints matched the job pair
- that created many endpoint states that were never outputable for the job

Implemented:
- for a job `(T1, T2)`, explicit endpoints are promoted directly to the valid
  endpoint assignment(s)
- endpoint hierarchy crawling is removed from the type-pair traversal

Semantic rule:
- valid endpoint assignments are exactly:
  - left -> `T1`, right -> `T2`
  - left -> `T2`, right -> `T1`
- keep whichever assignments the explicit endpoint types are descendants of

This is exact for all type pairs. It does not rely on a special case like
`ChemicalEntity / DiseaseOrPhenotypicFeature`.

Effect:
- removes useless endpoint-dimension traversal
- produced the first major throughput jump in profiling

### 8. Aggregate Predictors After Endpoint Promotion

Problem that remained after endpoint promotion:
- the worker still iterated once per explicit predictor even when many explicit
  predictors collapsed to the same promoted starting form

Implemented:
- predictors are now grouped by their promoted starting metapath before
  traversal
- overlaps and explicit counts are summed on that promoted-start key

Effect:
- `predictor_total` for the ChemicalEntity/DiseaseOrPhenotypicFeature sample
  target dropped from `2,207,143` to `491,215`
- this produced the largest overall gain so far

### Profiling Summary

On the ChemicalEntity / DiseaseOrPhenotypicFeature sample runs, the “whole
ass” version includes:
- compact state IDs
- target sorting + carried prune state
- direct endpoint promotion
- predictor aggregation by promoted starting form

Observed improvement versus the earlier carryover-only version on target 1:
- `0.001`: about `32/s` vs `5/s`
- `0.1`: about `59/s` vs `8/s`
- `0.99`: about `92/s` vs `12/s`
- RSS dropped from roughly `22-28 GB` into the `6-12 GB` range during the same
  phase

This does not mean the broad jobs are solved, but it moved the hot path from
"multi-day and memory-heavy" to "operationally tractable for target 1".

## Endpoint Expansion Ceiling

Current grouping jobs expand metapaths all the way up the type hierarchy and
then filter variants back down to those whose endpoints exactly match the
job's type pair.

Example:
- job type pair: `(ChemicalEntity, Disease)`
- explicit path already starts at `ChemicalEntity`
- current code still generates broader endpoint variants like `NamedThing`
  and `Entity`, only to discard them later because the endpoints no longer
  match `(ChemicalEntity, Disease)`

Possible optimization:
- when expanding endpoint types inside a type-pair job `(type1, type2)`,
  stop endpoint expansion once an end has reached `type1` or `type2`
- do not generate endpoint ancestors broader than the job ceiling

Expected benefit:
- reduces useless variant generation
- lowers CPU and temporary expansion volume in:
  - local count-map building
  - predictor expansion during aggregation

Limitation:
- this will not materially help the largest broad jobs, because jobs like
  `(NamedThing, NamedThing)` or similarly broad endpoint pairs still have no
  meaningful endpoint ceiling below the top of the hierarchy
- so this is a worthwhile optimization, but not a solution to the worst-case
  broad-shard runtime problem

## Reuse Predictor Expansion Between Count Build And Aggregation

Current grouping worker behavior effectively expands explicit predictor paths
twice within the same type-pair job:

1. During local count-map construction:
   - explicit N-hop predictor paths are expanded to build
     `predictor_variant_counts`
2. During per-target overlap aggregation:
   - the same explicit predictor paths are expanded again the first time they
     are encountered in aggregation, then cached for reuse within that phase

So the worker does not recompute predictor expansion for every target bucket,
but it still duplicates the expansion work across the two main phases.

Possible optimization:
- while building `predictor_variant_counts` in the first phase, also populate
  the per-job predictor expansion cache
- then, during target aggregation, reuse the already-built expansion list for
  each explicit predictor path instead of expanding it again on first use

Expected benefit:
- removes one full predictor expansion pass per explicit predictor path
- reduces CPU spent in the aggregation phase
- may reduce timeouts for jobs where repeated predictor expansion is a
  meaningful fraction of runtime

Limitation:
- this does not address the cost of scanning overlap result files
- this does not reduce the size of broad type-pair jobs
- it is an efficiency improvement, not a fix for the largest pathological jobs

## Turn Precision Filter Into Branch Pruning

Current precision filtering is only a late exact rejection:
- for a given target 1-hop variant, if a predictor variant has
  `predictor_count > target_count / min_precision`, that exact predictor
  variant is skipped

What it does **not** do:
- it does not prevent broader ancestor predictor variants from being generated
- it does not prune expansion branches upward through the hierarchy
- it does not reduce the earlier count-map construction cost

So the current implementation is safe but weak: it rejects exact variants late
instead of preventing whole super-variant branches from being explored.

Possible optimization:
- when a predictor variant fails the precision ceiling for a given target,
  prune all broader super-variants of that predictor for that target

Why this is valid:
- broader predictor variants have counts greater than or equal to narrower ones
- if variant `V` is already too broad to meet the target precision threshold,
  any ancestor/super-variant of `V` is also too broad

What this requires:
- an expansion process with explicit parent/ancestor structure or a traversal
  order from specific to broad
- enough bookkeeping to stop generating super-variants once a narrower variant
  has already failed

Expected benefit:
- this is the first precision-based optimization that can meaningfully reduce
  expansion work rather than only trimming output

Limitation:
- still target-specific: a branch pruned for one target may remain valid for a
  different target
- broad type-pair jobs can still be expensive even with correct branch pruning

## Stop Aggregating Once Only One Contributing Subpath Remains

Observation:
- for very broad aggregated paths such as `(NamedThing, treats, NamedThing)`,
  many logical subclass combinations do not actually contribute any data
- as aggregation proceeds upward, there may be a point where only one concrete
  contributing subpath remains under a broader aggregate

## Planned Redesign: Canonical Variant Traversal With One Counting Loop

This is the current redesign plan for grouping. It replaces the present
candidate-build plus exact-count split with a single traversal/count/prune
loop over canonical variants.

### Input Data

For one worker type pair `(A, B)`, the worker currently has two kinds of input:

1. Type-pair explicit-count shard
- file: `results_3hop/_tmp_prepare_grouping/typepair_explicit_paths/<A>__<B>.pkl`
- items: `(explicit_predictor_path, explicit_predictor_count)`

2. Raw overlap shards
- files: `results_3hop/results_matrix1_*.tsv`
- partitioned by first-hop `matrix1`, not by final type pair
- rows tell us that an explicit predictor path `P` overlaps an explicit target
  path `T` by `overlap(P, T)`

So after scan, the worker effectively knows:
- global explicit predictor counts `C(P)`
- sparse target overlaps `O(P, T)`

### Rolled Facts To Build Up Front

For the exact-match-only + rollup design, each explicit predictor `P` is rolled
up to one or more exact-match worker-valid predictor starts `R`.

From that mapping we build:

- `rolled_predictor_counts[predictor_start]`
  - exact rolled count
  - sum of all explicit predictor counts that roll to that start

- `target_overlaps[target][predictor_start]`
  - exact overlap between a target and a rolled predictor start
  - sum of all explicit overlaps whose predictors roll to that start

- `target_counts[target]`
  - exact target count for each canonical target

At that point, we no longer need a later repair pass for predictor counts,
because the rolled predictor start already has the count we intend to use.

### Canonical Traversal Identity

The plan is to make traversal identity and emitted identity the same thing:

- `predictor_start`
  - the rolled exact-match predictor we start from

- `variant`
  - every canonical generalized metapath reached from that start

There is no separate algorithm-level `state` noun in this design. The
traversal walks canonical variants directly.

This requires:
- canonical endpoint ordering
- symmetric predicates normalized
- reverse-equivalent forms collapsed

### Per-Target Working Structures

For each target:

- `aggregated_predictor_counts[variant]`
- `aggregated_overlap_counts[variant]`
- `pruned_variants`
  - an upward-closed pruned set

If cross-target pruning reuse remains enabled, a carried prune set may also be
seeded from larger targets processed earlier.

### One-Loop Traversal / Counting / Pruning

For each target in descending target-count order:

```python
max_predictor_count = target_counts[target] / min_precision

aggregated_predictor_counts = defaultdict(int)
aggregated_overlap_counts = defaultdict(int)
pruned_variants = inherited_pruned_variants.copy()

for predictor_start in target_predictors_sorted_by_count_desc:
    predictor_count = rolled_predictor_counts[predictor_start]
    overlap = target_overlaps[target][predictor_start]

    stack = [predictor_start]
    seen_variants = {predictor_start}

    while stack:
        variant = stack.pop()

        if variant in pruned_variants:
            continue

        proposed_count = aggregated_predictor_counts[variant] + predictor_count

        if proposed_count > max_predictor_count:
            for ancestor in ancestor_closure(variant):
                pruned_variants.add(ancestor)
                aggregated_predictor_counts.pop(ancestor, None)
                aggregated_overlap_counts.pop(ancestor, None)
            continue

        aggregated_predictor_counts[variant] = proposed_count
        aggregated_overlap_counts[variant] += overlap

        for parent_variant in more_general_canonical_children(variant):
            if parent_variant not in seen_variants:
                seen_variants.add(parent_variant)
                stack.append(parent_variant)
```

At the end of the target loop, rows are written directly from:
- `aggregated_predictor_counts`
- `aggregated_overlap_counts`

There is no separate `compute_exact_predictor_counts(...)` phase in this
design.

### Why The Pruned Set Must Be Ancestor-Closed

If variant `V` is too broad to meet the target precision threshold, every
ancestor of `V` is also too broad.

So when `V` prunes, the plan is to add:
- `V`
- every ancestor of `V`

to the pruned set.

This matters because otherwise a later traversal can reach an already-doomed
ancestor by a different path and waste time rediscovering the same pruning
fact.

With an ancestor-closed prune set:
- any hit in `pruned_variants` stops the branch immediately
- future traversals do not re-explore the same upper cone

### Intended Benefits

- remove the candidate-build / exact-count split
- use true rolled counts for pruning immediately
- align traversal identity with output identity
- reduce duplicate work from different generalization orders
- make the pruning model match the lattice mathematics more closely

### Main Open Requirement

This redesign depends on a correct canonical variant lattice:
- child generation must be correct
- duplicate reverse/symmetric forms must already be collapsed
- emitted variants and traversed variants must be the same identity

Until that canonical lattice exists, the current split architecture remains in
use.

## Current Implementation

The implementation now matches the planned architecture. The active worker path
calls `build_candidate_variants_for_targets(...)` in
`src/pipeline/workers/run_grouping.py`.

### Architecture

Target-centric, rolled-start traversal with support-map dedup:

1. **Precompute global variant predictor counts** — explicit typepair expansion
   produces the correct global predictor count for each canonical variant,
   keyed by `original_predictor_identity` to prevent double-counting
2. **Outer loop: targets** — sorted by descending target count
3. **Inner loop: rolled predictor starts** — explicit predictors promoted to
   typepair rollup keys via `promote_metapath_endpoints_to_typepair_rollup_keys`
4. **Traversal: canonical variants** — `traverse_canonical_variants_for_typepair_pruned`
   walks the variant lattice from each rolled start
5. **Overlap: per-target** — overlap is accumulated per-target during traversal
6. **Output counts** — the precomputed global variant counts are used for the
   predictor_count field in output rows, not the traversal-accumulated support.
   This is necessary because canonical traversal from rolled starts can
   over-generate variants relative to explicit expansion (documented by
   `test_promoted_start_traversal_mismatches_explicit_ancestor_set_for_failing_case`)

### Pruning

- **State-level pruning** — lower-bound count checks on intermediate lattice
  states cut entire subtrees before expanding to individual variants
- **Variant-level pruning** — ancestor-closure pruning marks a variant and all
  its ancestors as pruned, preventing further accumulation
- **Cross-target carryover** — `global_pruned_states` carries state-level prune
  decisions forward across targets (processed in descending count order, so
  high-count targets prune states that low-count targets inherit)

### Correctness Properties

Three concrete correctness issues were resolved before this architecture
could be used:

1. **Double-counting from converging starts** — one explicit predictor can
   produce multiple rolled starts that reach the same ancestor variant.
   Resolved by keying support maps on `original_predictor_identity`, so the
   same predictor contributes its count only once per variant.

2. **Same-endpoint reverse dedup** — `Gene|regulates|F|Gene` and
   `Gene|regulates|R|Gene` are directional views of the same edge, not
   distinct predictors. Resolved by `original_predictor_identity(...)` which
   collapses reverse-equivalent forms.

3. **Canonical traversal semantics** — the canonical traversal originally
   disagreed with explicit bounded expansion for same-endpoint symmetric
   cases. Fixed in `a7bb4d2`.

Example intuition:
- many subclass combinations under `(NamedThing, treats, NamedThing)` may have
  zero support
- eventually a broader aggregate may be supported by exactly one explicit or
  already-aggregated child path

Possible optimization:
- detect when an aggregate path has exactly one contributing child/subpath
- once that is true, stop performing further overlap/count aggregation work for
  the remaining broader aliases of that same lineage
- instead, propagate or materialize those broader aliases directly from the
  single contributing path without recomputing

Expected benefit:
- avoids redundant repeated aggregation where broader variants add no new data
- especially relevant for very broad endpoint/predicate combinations where the
  logical hierarchy is large but the actual support is sparse

Open questions:
- how to track "number of contributing child paths" efficiently during
  aggregation
- whether this should be applied on the target side, predictor side, or both
- how to integrate this with precision-based pruning so the two optimizations
  reinforce rather than duplicate each other

## Exact Pair-Set Tracking (Phase A + B)

### What Was Implemented

Two new phases added to the grouping worker, activated by `--matrices-dir`:

**Phase A** — after `build_target_variant_counts` (sum-based), load all base
matrices from disk and compute exact target pair counts by unioning matching
base matrices in a unified coordinate space. The unified space concatenates
per-type dense indices with offsets, so matrices for different subtypes
(e.g. SmallMolecule + Drug → ChemicalEntity) can be unioned without collision.

**Phase B** — after `build_candidate_variants_for_targets`, for each surviving
candidate predictor variant, reconstruct the N-hop matrix from base matrices
(chain of `mxm` operations), remap into the same unified space, union across
contributing predictor paths, and intersect with the Phase A target pair set.
Produces exact `predictor_count` and `overlap`.

New files:
- `src/library/unified_index.py` — offset computation, matrix remapping,
  target pair set construction, N-hop reconstruction
- `tests/test_unified_index.py` — 18 unit tests
- Integration tests in `tests/test_grouping_explicit_shards.py`

### What Phase A Showed

Run on ChemicalEntity × DiseaseOrPhenotypicFeature (SLURM, 64GB):

- Loading 5295 base matrices: **17 seconds**
- Computing exact counts for 32 target variants: **84 seconds**
- Total Phase A overhead: ~100 seconds (negligible vs 33 min file scan)

Sum semantics was overcounting by ~2× across the board:

| Target | Summed | Exact | Ratio |
|--------|--------|-------|-------|
| related_to\|A | 9,145,079 | 4,159,177 | 0.455 |
| related_to_at_instance_level\|A | 9,095,655 | 4,150,679 | 0.456 |
| associated_with\|A | 6,711,945 | 3,315,130 | 0.494 |
| treats_or_applied\|F | 1,696,872 | 751,216 | 0.443 |
| studied_to_treat\|F | 716,716 | 358,293 | 0.500 |
| treats\|F | 705,550 | 351,095 | 0.498 |
| affects\|F | 316,092 | 155,001 | 0.490 |
| (24 smaller targets) | <310K | <155K | ~0.500 |

The ~0.500 ratio on most targets means sum-based counts were roughly double
the true pair count. This is expected: each pair typically appears via two
predicates that both roll up to the same ancestor (e.g. treats + ameliorates
→ related_to).

### Orientation Bug Found and Fixed

Canonical variant form puts alphabetically smaller type first:
`SmallMolecule|treats|F|Disease` → `Disease|treats|R|SmallMolecule`.

The original implementation passed `(type1_offsets, type2_offsets)` directly
to `build_target_pair_set`, but when a variant has reversed endpoints (e.g.
Disease first when type1=ChemicalEntity), the offset tables don't contain
the right types. All exact counts came back as 0.

Fix: `_detect_metapath_orientation(metapath, type1, type2)` detects whether
a metapath's endpoints are in `(type1, type2)` or `(type2, type1)` order.
Reversed variants get swapped offsets and a transpose after construction.
Applied to both `compute_exact_target_pair_counts` and
`compute_exact_predictor_metrics`.

### What Actually Happened At Runtime

Job 1175761: ChemicalEntity × DiseaseOrPhenotypicFeature, 64GB, min_precision=0.99.

- File scan: 33 min, 52.8M rows, 16.5M matched, 5.6 GB RSS
- Phase A: ~100 seconds, exact counts computed for 32 targets
- Candidate build (Pass 3): **OOM killed after 2h24m**

The candidate build for `related_to|A` (4.2M exact pairs) was processing
530K promoted predictors, each generating ~144 ancestor variants. The pruning
threshold at 0.99 is `4,159,177 / 0.99 = 4,201,189` — essentially every
predictor's count is below this, so nothing gets branch-pruned. Accepted
variants grew at ~400 MB/min, RSS hit 64GB at ~18 GB into the build.

### The Fundamental Problem

Phase A halves the target counts and tightens pruning thresholds, but the
threshold is still proportional to target size. For the top 3 targets
(`related_to`, `related_to_at_instance_level`, `associated_with`) with
3.3M–4.2M exact pairs, the threshold at any precision ≥ 0.01 is far larger
than any individual 3-hop predictor count. This means:

1. The pruning heuristic (`predictor_count > target_count / min_precision`)
   does not fire for these targets — the threshold is in the millions while
   individual predictors have at most hundreds of thousands of pairs
2. All 530K predictors × ~144 variants survive → millions of accepted variants
3. Memory grows linearly with accepted variants, eventually OOMing

Phase A helps for medium-sized targets (tens or hundreds of thousands of
pairs) where halving the count meaningfully tightens the threshold. But for
the monster targets, the problem is that the threshold is structurally
irrelevant — it's too high to prune anything.

Phase B was never reached in this run. It would also be expensive for the
monster targets: reconstructing N-hop matrices for millions of surviving
candidate variants.

### What Needs To Change

The candidate variant expansion (Pass 3) is the bottleneck, not the
precision computation. The current architecture generates all variant
expansions first, then filters by precision during output. For targets with
millions of pairs, this means the expansion phase generates and stores
millions of variants that will never meet any useful precision threshold.

The core issue: **precision filtering happens too late.** The overlap data
needed to compute actual precision already exists in `onehop_to_overlaps`
(from the file scan), but it's only consulted after variant expansion
completes. Integrating overlap-aware filtering into the expansion loop —
or restructuring to avoid the expansion entirely for targets where it's
guaranteed to blow up — would address the fundamental scaling problem.

Possible directions:
1. **Skip variant expansion for monster targets entirely** — compute exact
   metrics directly from base matrices for the explicit predictor paths,
   without expanding to ancestor variants. This is feasible because the
   explicit paths already have their overlap data from the overlap phase.
2. **Cap accepted variants per target** — if a target's expansion exceeds
   a memory budget, fall back to explicit-path-only output for that target.
3. **Streaming variant expansion** — process one predictor at a time through
   expansion → Phase B reconstruction → output, without accumulating all
   variants in memory. This trades CPU (redundant traversal) for memory.
4. **Pre-filter predictors by overlap ratio** — before variant expansion,
   check if the explicit predictor's sum-based overlap / predictor_count
   could possibly meet the precision threshold. Predictors with no chance
   of meeting precision at even the most specific level can be skipped
   entirely, avoiding their variant expansion.
