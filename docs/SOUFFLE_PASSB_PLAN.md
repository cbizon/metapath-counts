# Soufflé-Based Pass B Replacement

## Problem Recap

Pass B expands 31M explicit 3-hop metapaths into hierarchical variants via Cartesian product, then sums counts. Current Python implementation:
- 512 SLURM shards, each producing 11-29 GB pickle files
- Per-path variant counts up to 563,040
- Caching (1.05x) and edge-pair DP (0.87x) failed to help
- Bottleneck is the combinatorial expansion itself, not lookups

## Proposal

Replace the Python Pass B expansion + aggregation with a Soufflé Datalog program. The hierarchy lookups stay in Python (cheap, uses biolink-model-toolkit). The expensive 7-way join and aggregation runs in Soufflé (compiled parallel C++).

## Why Soufflé Fits

The expansion is fundamentally: for each explicit path, join each of its 7 components (4 types, 3 predicates) against an ancestor relation, then sum counts grouped by the resulting variant path. This is a textbook relational join + aggregation, which is exactly what Datalog engines are built for.

Key advantages over the Python approach:
- **No per-path Cartesian explosion** — Soufflé's join engine uses indexed relations. Shared component prefixes are reused through index structure, not recomputed per path.
- **Compiled parallel C++** — Soufflé compiles programs with `-j N` parallelism. No Python overhead, no GIL.
- **Eliminates shard infrastructure** — No 512 SLURM shards, no giant pickle files, no Reduce B merge. One Soufflé run, one output TSV.
- **Native TSV I/O** — reads/writes delimited files directly.

## Semantics to Capture

`generate_metapath_variants` in `src/library/aggregation.py` does more than simple ancestor expansion. The Soufflé program must replicate all of the following:

### 1. Type ancestor expansion
Each node type expands to itself + all Biolink ancestors. Pseudo-types like `Gene+SmallMolecule` also expand to their constituents.

### 2. Predicate ancestor expansion
Plain predicates expand to self + ancestors. Compound predicates (`base--direction--aspect`) expand each component independently (base ancestors x direction qualifier ancestors x aspect qualifier ancestors), and qualifiers can be dropped to None.

### 3. Symmetric direction adjustment
When a predicate variant is symmetric (e.g. `related_to`), the direction must change from `F`/`R` to `A`.

### 4. Canonicalization
Output metapaths are canonical: `first_type <= last_type` alphabetically. If not, the path is reversed and directions flipped (F<->R, A stays A).

### 5. Same-type dedup
When the **original** path has identical src/tgt types and direction is `R`, variants where the predicate becomes symmetric are skipped. The `F` version of the original path already generates those variants.

### 6. Different-to-same-type reversal
When different-type endpoints expand to same-type endpoints, both direction orderings are yielded (the canonical one and its reverse), because the canonical direction choice was forced by the original alphabetical ordering which no longer applies.

## Architecture

### Python pre-computation (run once, fast)

Produce flat TSV files from biolink-model-toolkit:

| File | Columns | Description |
|------|---------|-------------|
| `type_variant.tsv` | `type\tancestor` | All (type, ancestor) pairs including self. Pseudo-types pre-expanded to constituents + all ancestors. |
| `pred_variant.tsv` | `predicate\tancestor_pred` | All (predicate, variant) pairs. Compound predicates pre-expanded as cross-product with qualifier dropping. |
| `symmetric_pred.tsv` | `predicate` | Set of symmetric predicates. |
| `explicit_paths.tsv` | `t0\tp0\td0\tt1\tp1\td1\tt2\tp2\td2\tt3\tcount` | Decomposed explicit paths (pipe-separated string split into columns). |

The type and predicate variant relations are small (thousands of rows). The explicit paths are large (31M rows) but trivial to export.

### Soufflé program

```datalog
// --- Input relations (loaded from TSV) ---

.decl type_var(orig: symbol, anc: symbol)
.input type_var(IO=file, filename="type_variant.tsv", delimiter="\t")

.decl pred_var(orig: symbol, anc: symbol)
.input pred_var(IO=file, filename="pred_variant.tsv", delimiter="\t")

.decl symmetric(pred: symbol)
.input symmetric(IO=file, filename="symmetric_pred.tsv", delimiter="\t")

.decl explicit_path(t0:symbol, p0:symbol, d0:symbol,
                    t1:symbol, p1:symbol, d1:symbol,
                    t2:symbol, p2:symbol, d2:symbol,
                    t3:symbol, count:number)
.input explicit_path(IO=file, filename="explicit_paths.tsv", delimiter="\t")

// --- Direction adjustment ---
// Resolve direction: if ancestor predicate is symmetric, force "A"; otherwise keep original.

.decl resolve_dir(orig_dir: symbol, pred_anc: symbol, out_dir: symbol)
resolve_dir(D, P, "A") :- symmetric(P), D = "F".
resolve_dir(D, P, "A") :- symmetric(P), D = "R".
resolve_dir(D, P, "A") :- symmetric(P), D = "A".
resolve_dir(D, P, D)   :- !symmetric(P), D = "F".
resolve_dir(D, P, D)   :- !symmetric(P), D = "R".
resolve_dir(D, P, D)   :- !symmetric(P), D = "A".

// --- Raw expansion (before canonicalization) ---
// Join each component against its ancestor relation + adjust directions.

.decl raw_variant(t0:symbol, p0:symbol, d0:symbol,
                  t1:symbol, p1:symbol, d1:symbol,
                  t2:symbol, p2:symbol, d2:symbol,
                  t3:symbol,
                  orig_t0:symbol, orig_t3:symbol,
                  orig_d0:symbol, orig_d1:symbol, orig_d2:symbol,
                  count:number)

raw_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
            T0, T3, D0, D1, D2, Count) :-
    explicit_path(T0, P0, D0, T1, P1, D1, T2, P2, D2, T3, Count),
    type_var(T0, T0a), type_var(T1, T1a), type_var(T2, T2a), type_var(T3, T3a),
    pred_var(P0, P0a), pred_var(P1, P1a), pred_var(P2, P2a),
    resolve_dir(D0, P0a, D0out),
    resolve_dir(D1, P1a, D1out),
    resolve_dir(D2, P2a, D2out).

// --- Same-type dedup filter ---
// Skip variants where original had same src/tgt AND original direction was R
// AND the ancestor predicate became symmetric. The F version covers these.
// (Applied per-edge: if ANY edge triggers this, skip the whole variant.)

.decl skip_variant(t0:symbol, p0:symbol, d0:symbol,
                   t1:symbol, p1:symbol, d1:symbol,
                   t2:symbol, p2:symbol, d2:symbol,
                   t3:symbol,
                   orig_t0:symbol, orig_t3:symbol,
                   orig_d0:symbol, orig_d1:symbol, orig_d2:symbol,
                   count:number)

// Skip if original src==tgt and any original direction was R with symmetric ancestor
skip_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
             OT0, OT3, OD0, OD1, OD2, C) :-
    raw_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
                OT0, OT3, OD0, OD1, OD2, C),
    OT0 = OT3, OD0 = "R", D0out = "A".

skip_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
             OT0, OT3, OD0, OD1, OD2, C) :-
    raw_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
                OT0, OT3, OD0, OD1, OD2, C),
    OT0 = OT3, OD1 = "R", D1out = "A".

skip_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
             OT0, OT3, OD0, OD1, OD2, C) :-
    raw_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
                OT0, OT3, OD0, OD1, OD2, C),
    OT0 = OT3, OD2 = "R", D2out = "A".

// --- Filtered variants (not skipped) ---

.decl filtered_variant(t0:symbol, p0:symbol, d0:symbol,
                       t1:symbol, p1:symbol, d1:symbol,
                       t2:symbol, p2:symbol, d2:symbol,
                       t3:symbol,
                       orig_t0:symbol, orig_t3:symbol,
                       count:number)

filtered_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
                 OT0, OT3, C) :-
    raw_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
                OT0, OT3, OD0, OD1, OD2, C),
    !skip_variant(T0a, P0a, D0out, T1a, P1a, D1out, T2a, P2a, D2out, T3a,
                  OT0, OT3, OD0, OD1, OD2, C).

// --- Canonicalization ---
// Canonical form: first_type <= last_type. If not, reverse path and flip F<->R.

.decl canon_variant(t0:symbol, p0:symbol, d0:symbol,
                    t1:symbol, p1:symbol, d1:symbol,
                    t2:symbol, p2:symbol, d2:symbol,
                    t3:symbol,
                    orig_t0:symbol, orig_t3:symbol,
                    count:number)

// Already canonical: t0 <= t3
canon_variant(T0, P0, D0, T1, P1, D1, T2, P2, D2, T3, OT0, OT3, C) :-
    filtered_variant(T0, P0, D0, T1, P1, D1, T2, P2, D2, T3, OT0, OT3, C),
    T0 <= T3.

// Needs reversal: t0 > t3 — reverse nodes, predicates, flip directions
.decl flip(orig: symbol, flipped: symbol)
flip("F", "R"). flip("R", "F"). flip("A", "A").

canon_variant(T3, P2, D2f, T2, P1, D1f, T1, P0, D0f, T0, OT0, OT3, C) :-
    filtered_variant(T0, P0, D0, T1, P1, D1, T2, P2, D2, T3, OT0, OT3, C),
    T0 > T3,
    flip(D0, D0f), flip(D1, D1f), flip(D2, D2f).

// --- Different-to-same-type reversal ---
// When original had different endpoints but variant has same endpoints,
// also emit the reversed-direction form (unless all directions are A).

.decl has_directional(d0: symbol, d1: symbol, d2: symbol)
has_directional(D0, D1, D2) :- D0 != "A".
has_directional(D0, D1, D2) :- D1 != "A".
has_directional(D0, D1, D2) :- D2 != "A".

// Extra reversed variant for different->same case
canon_variant(T0, P2, D2f, T1, P1, D1f, T2, P0, D0f, T3, OT0, OT3, C) :-
    canon_variant(T0, P0, D0, T1, P1, D1, T2, P2, D2, T3, OT0, OT3, C),
    OT0 != OT3,      // original had different types
    T0 = T3,          // variant has same types
    has_directional(D0, D1, D2),
    flip(D0, D0f), flip(D1, D1f), flip(D2, D2f).

// --- Aggregation ---
// Sum counts over all source paths, dropping the orig tracking columns.

.decl agg_count(t0:symbol, p0:symbol, d0:symbol,
                t1:symbol, p1:symbol, d1:symbol,
                t2:symbol, p2:symbol, d2:symbol,
                t3:symbol, total:number)
.output agg_count(IO=file, filename="aggregated_counts.tsv", delimiter="\t")

agg_count(T0, P0, D0, T1, P1, D1, T2, P2, D2, T3, Total) :-
    Total = sum C : {
        canon_variant(T0, P0, D0, T1, P1, D1, T2, P2, D2, T3, _, _, C)
    }.
```

**Note:** This Soufflé sketch is a starting point. The same-type dedup, different-to-same reversal, and `has_directional` rules need validation against the Python implementation with test cases before production use.

### Python wrapper

The Python side of the replacement:

1. **Export hierarchy relations** — call `get_type_variants()` for all types seen in explicit paths, `get_predicate_variants()` for all predicates, `get_symmetric_predicates()`. Write as TSV.
2. **Export explicit paths** — read the existing `explicit_counts` shards, decompose pipe-separated paths into columns, write as one big TSV.
3. **Run Soufflé** — `souffle -j <cores> passb.dl` (or pre-compile with `souffle -o passb passb.dl` then run the binary).
4. **Read output** — `aggregated_counts.tsv` is the final result. Reassemble into pipe-separated metapath strings if downstream expects that format.

### Integration points

- Replaces: `src/pipeline/prepare_grouping.py` Pass B map step, the shard infrastructure, and Reduce B merge.
- Keeps: everything upstream (matrix building, overlap computation, explicit count generation) and everything downstream (grouping workers that consume aggregated counts).
- The Soufflé program is 3-hop specific. For N-hop generality, template-generate the `.dl` file (different column counts per hop).

## Risks and Open Questions

1. **Semantic fidelity** — the Soufflé rules must exactly match `generate_metapath_variants`. Validate on a small sample (e.g. 200 paths from a shard) and diff the output against Python.
2. **Output size** — the aggregated result is inherently large. Soufflé handles it but disk I/O may still be significant.
3. **Intermediate relation size** — `raw_variant` could be very large (31M paths x avg 40K variants). Soufflé manages memory well but this needs profiling. If memory is tight, Soufflé supports disk-backed relations via SQLite (`-DUSE_SQLITE`).
4. **Soufflé availability on cluster** — need to install or load a module. It's a single binary; can also compile from source or use conda/spack.
5. **Compound predicate handling** — pre-expanding compound predicate variants in Python is cleaner than expressing the qualifier cross-product logic in Datalog.
6. **Pseudo-type handling** — pre-expanding pseudo-types into constituent+ancestor pairs in Python keeps Soufflé simple.
