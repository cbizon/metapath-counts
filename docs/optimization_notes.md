# Optimization Notes

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
