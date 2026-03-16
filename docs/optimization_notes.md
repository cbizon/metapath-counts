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
- how to track “number of contributing child paths” efficiently during
  aggregation
- whether this should be applied on the target side, predictor side, or both
- how to integrate this with precision-based pruning so the two optimizations
  reinforce rather than duplicate each other
