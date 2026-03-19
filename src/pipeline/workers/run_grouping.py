#!/usr/bin/env python3
"""
Worker script to group results for all 1-hop metapaths between a type pair.

This worker no longer relies on a global aggregated_nhop_counts.json. Instead it:
1. Loads explicit paths/counts for the current type pair shard
2. Builds local hierarchical counts for 1-hop targets and N-hop predictors
3. Streams relevant overlap rows
4. Applies precision pruning during predictor aggregation
"""

import argparse
import pickle
import json
import os
import time
from collections import defaultdict

import zstandard

from library.hierarchy import get_type_ancestors
from library.aggregation import (
    canonical_variant_ancestor_closure_state_ids_for_typepair,
    canonical_variant_metapath_from_state_ids,
    canonical_variant_state_ids,
    expand_metapath_to_typepair_variants,
    expand_metapath_to_variants,
    promote_metapath_endpoints_to_typepair_rollup_keys,
    promote_metapath_endpoints_to_typepair_starts,
    traverse_canonical_variants_for_typepair_pruned,
    get_type_variants,
    calculate_metrics,
    parse_compound_predicate,
    parse_metapath,
    original_predictor_identity,
)


def get_rss_mb():
    """Return resident memory in MB from /proc/self/status when available."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def now_s(start_time):
    return round(time.time() - start_time, 1)


def print_profile_event(stage, start_time, **metrics):
    parts = [f"{k}={v}" for k, v in metrics.items()]
    rss = get_rss_mb()
    parts.append(f"elapsed_s={now_s(start_time)}")
    if rss is not None:
        parts.append(f"rss_mb={rss:.1f}")
    print(f"[grouping] {stage}: " + " ".join(parts), flush=True)


def write_progress_file(progress_file, payload):
    if not progress_file:
        return
    tmp_path = f"{progress_file}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, progress_file)


def build_progress_payload(type1, type2, stage, start_time, counters, stage_timings, **extra):
    payload = {
        "type1": type1,
        "type2": type2,
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": now_s(start_time),
        "rss_mb": None if get_rss_mb() is None else round(get_rss_mb(), 1),
        "counters": counters,
        "stage_timings_s": {k: round(v, 3) for k, v in stage_timings.items()},
    }
    payload.update(extra)
    return payload


def get_max_predictor_count_for_precision(onehop_count, min_precision):
    """Return the largest predictor count that can still meet min_precision."""
    if min_precision <= 0:
        return None
    return onehop_count / min_precision


def compute_total_possible(type1, type2, type_node_counts):
    """Compute total possible pairs for a type pair."""
    count1 = type_node_counts.get(type1, 0)
    count2 = type_node_counts.get(type2, 0)
    return count1 * count2


def check_type_match(onehop_path, type1, type2):
    """Check if a 1-hop metapath aggregates to a type pair."""
    parts = onehop_path.split('|')
    if len(parts) != 4:
        return False

    src_type, _, _, tgt_type = parts
    src_ancestors = get_type_ancestors(src_type)
    tgt_ancestors = get_type_ancestors(tgt_type)
    match_forward = (type1 in src_ancestors and type2 in tgt_ancestors)
    match_reverse = (type2 in src_ancestors and type1 in tgt_ancestors)
    return match_forward or match_reverse


def contains_pseudo_type(metapath):
    """Check if a metapath contains any pseudo-type nodes."""
    parts = metapath.split('|')
    for i in range(0, len(parts), 3):
        if '+' in parts[i]:
            return True
    return False


def should_exclude_metapath(metapath, excluded_types, excluded_predicates):
    """Check if a metapath should be excluded based on types or predicates."""
    if not excluded_types and not excluded_predicates:
        return False

    parts = metapath.split('|')
    nodes = []
    predicates = []
    for i, part in enumerate(parts):
        if i % 3 == 0:
            nodes.append(part)
        elif i % 3 == 1:
            predicates.append(part)

    if excluded_types:
        for node in nodes:
            if node in excluded_types:
                return True

    if excluded_predicates:
        for pred in predicates:
            if pred in excluded_predicates:
                return True
            base_pred = parse_compound_predicate(pred)[0]
            if base_pred != pred and base_pred in excluded_predicates:
                return True

    return False


def typepair_shard_path(explicit_shards_dir, type1, type2):
    """Return the path to the explicit-count shard for a type pair."""
    return os.path.join(explicit_shards_dir, f"{type1}__{type2}.pkl")


def load_typepair_explicit_counts(explicit_shards_dir, type1, type2):
    """Load explicit path counts relevant to the current type pair."""
    shard_path = typepair_shard_path(explicit_shards_dir, type1, type2)
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Explicit-count shard not found: {shard_path}")

    with open(shard_path, "rb") as f:
        return pickle.load(f)


def build_target_variant_counts(explicit_items, type1, type2, start_time=None,
                                progress_file=None, counters=None, stage_timings=None):
    """Build local hierarchical counts for 1-hop target variants only."""
    target_variant_support = defaultdict(dict)
    target_expansion_cache = {}
    counters = counters if counters is not None else {}
    stage_timings = stage_timings if stage_timings is not None else {}

    t0 = time.time()
    next_progress_at = t0 + 30.0

    for idx, (path, count) in enumerate(explicit_items, start=1):
        _, predicates, _ = parse_metapath(path)
        if len(predicates) != 1:
            continue
        predictor_id = original_predictor_identity(path)

        counters["target_expansion_requests"] = counters.get("target_expansion_requests", 0) + 1
        if path in target_expansion_cache:
            variants = target_expansion_cache[path]
            counters["target_expansion_cache_hits"] = counters.get("target_expansion_cache_hits", 0) + 1
        else:
            variants = list(expand_metapath_to_typepair_variants(path, type1, type2))
            target_expansion_cache[path] = variants
            counters["target_expansion_cache_misses"] = counters.get("target_expansion_cache_misses", 0) + 1
            counters["target_variants_generated"] = counters.get("target_variants_generated", 0) + len(variants)

        for variant in variants:
            target_variant_support[variant][predictor_id] = count

        should_log = False
        now = time.time()
        if idx % 100000 == 0 or idx == len(explicit_items):
            should_log = True
        elif start_time and now >= next_progress_at:
            should_log = True
            next_progress_at = now + 30.0

        if start_time and should_log:
            target_paths = counters.get("target_expansion_cache_misses", 0)
            avg_target_variants = round(counters.get("target_variants_generated", 0) / target_paths, 2) if target_paths else 0.0
            print_profile_event(
                "build_target_variant_counts",
                start_time,
                explicit_done=idx,
                explicit_total=len(explicit_items),
                target_variants=len(target_variant_support),
                target_cache=len(target_expansion_cache),
                avg_target_variants=avg_target_variants,
            )
            write_progress_file(
                progress_file,
                build_progress_payload(
                    type1,
                    type2,
                    "build_target_variant_counts",
                    start_time,
                    counters,
                    stage_timings,
                    explicit_done=idx,
                    explicit_total=len(explicit_items),
                    target_variants=len(target_variant_support),
                    target_cache=len(target_expansion_cache),
                    avg_target_variants=avg_target_variants,
                ),
            )

    stage_timings["build_target_variant_counts"] = stage_timings.get("build_target_variant_counts", 0.0) + (time.time() - t0)
    target_variant_counts = {
        variant: sum(support.values())
        for variant, support in target_variant_support.items()
    }
    counters["target_variant_count_size"] = len(target_variant_counts)
    return dict(target_variant_counts), target_expansion_cache


def expand_predictor_variants_for_typepair(path, type1, type2, predictor_expansion_cache, counters):
    """Expand one predictor path through the full hierarchy and cache the result."""
    counters["predictor_expansion_requests"] = counters.get("predictor_expansion_requests", 0) + 1
    if path in predictor_expansion_cache:
        counters["predictor_expansion_cache_hits"] = counters.get("predictor_expansion_cache_hits", 0) + 1
        return predictor_expansion_cache[path]

    variants = list(expand_metapath_to_variants(path))
    predictor_expansion_cache[path] = variants
    counters["predictor_expansion_cache_misses"] = counters.get("predictor_expansion_cache_misses", 0) + 1
    counters["predictor_variants_generated"] = counters.get("predictor_variants_generated", 0) + len(variants)
    return variants


def group_predictor_items_by_promoted_start(explicit_items, n_hops, type1, type2, counters=None):
    """Aggregate explicit predictors by their worker-valid promoted exact-match form(s)."""
    grouped_support = defaultdict(dict)
    counters = counters if counters is not None else {}

    for path, count in explicit_items:
        if len(parse_metapath(path)[1]) != n_hops:
            continue
        predictor_id = original_predictor_identity(path)
        promoted_starts = promote_metapath_endpoints_to_typepair_rollup_keys(path, type1, type2)
        counters["predictor_endpoint_promotions"] = counters.get("predictor_endpoint_promotions", 0) + len(promoted_starts)
        for promoted_key in promoted_starts:
            grouped_support[promoted_key][predictor_id] = count

    grouped_counts = {
        key: sum(support.values())
        for key, support in grouped_support.items()
    }
    counters["grouped_predictor_count"] = len(grouped_counts)
    return dict(grouped_counts)


def build_global_variant_predictor_counts(
    explicit_items,
    n_hops,
    type1,
    type2,
    counters,
):
    """Accumulate exact global counts via a semantics-preserving variant-signature rollup."""
    global_variant_support = defaultdict(dict)

    for explicit_path, predictor_count in explicit_items:
        if len(parse_metapath(explicit_path)[1]) != n_hops:
            continue
        if not promote_metapath_endpoints_to_typepair_starts(explicit_path, type1, type2):
            continue
        predictor_id = original_predictor_identity(explicit_path)
        counters["predictor_expansion_requests"] = counters.get("predictor_expansion_requests", 0) + 1

        for variant in expand_metapath_to_typepair_variants(
            explicit_path,
            type1,
            type2,
        ):
            global_variant_support[canonical_variant_state_ids(variant)][predictor_id] = predictor_count

    counters["predictor_signature_rollup_count"] = len(global_variant_support)

    global_variant_predictor_counts = {
        variant_state_ids: sum(support.values())
        for variant_state_ids, support in global_variant_support.items()
    }

    counters["global_variant_predictor_count_size"] = len(global_variant_predictor_counts)
    return dict(global_variant_predictor_counts)


def get_metapath_endpoints(metapath):
    nodes, _, _ = parse_metapath(metapath)
    return nodes[0], nodes[-1]


def endpoints_within_target_pair(left_type, right_type, type1, type2):
    left_reachable = set(get_type_variants(left_type))
    right_reachable = set(get_type_variants(right_type))
    return (
        (type1 in left_reachable and type2 in right_reachable)
        or (type2 in left_reachable and type1 in right_reachable)
    )


def endpoints_exact_match(left_type, right_type, type1, type2):
    return tuple(sorted((left_type, right_type))) == tuple(sorted((type1, type2)))


def build_candidate_variants_for_targets(onehop_to_overlaps, rolled_predictor_counts, original_predictor_counts, target_variant_counts,
                                         type1, type2, min_precision,
                                         start_time, progress_file, counters, stage_timings):
    """Build target-aware predictor candidates with rolled-predictor support maps."""
    t0 = time.time()
    next_progress_at = t0 + 30.0
    candidate_rows_by_target = {}
    all_candidate_variants = set()
    target_items = sorted(
        onehop_to_overlaps.items(),
        key=lambda item: target_variant_counts.get(item[0], 0),
        reverse=True,
    )
    total_targets = len(target_items)
    counters["targets_sorted_by_target_count"] = total_targets
    global_pruned_states = set()
    state_token_ids = {}
    next_state_token_id = 0

    def rolled_count_for_key(key):
        if key in rolled_predictor_counts:
            return rolled_predictor_counts[key]
        if isinstance(key, tuple):
            return rolled_predictor_counts.get(key[0], 0)
        return 0

    for target_index, (onehop_path, nhop_overlaps) in enumerate(target_items, start=1):
        target_t0 = time.time()
        onehop_count_global = target_variant_counts.get(onehop_path, 0)
        target_type1, target_type2 = get_metapath_endpoints(onehop_path)
        max_predictor_count = get_max_predictor_count_for_precision(onehop_count_global, min_precision)
        aggregated_predictor_support = defaultdict(dict)
        aggregated_overlap_support = defaultdict(dict)
        pruned_variants = set()
        lower_bound_state_counts = defaultdict(int)
        variants_pruned_lower_bound = 0
        target_branch_pruned_at_start = counters.get("candidate_variants_branch_pruned", 0)
        target_accepted_at_start = counters.get("candidate_variants_accepted", 0)
        raw_target_predictors = sorted(
            nhop_overlaps.items(),
            key=lambda item: original_predictor_counts.get(original_predictor_identity(item[0]), 0),
            reverse=True,
        )
        promoted_target_predictors = defaultdict(lambda: {"predictor": {}, "overlap": {}})
        for nhop_path, overlap in raw_target_predictors:
            predictor_id = original_predictor_identity(nhop_path)
            promoted_starts = promote_metapath_endpoints_to_typepair_rollup_keys(nhop_path, type1, type2)
            counters["predictor_endpoint_promotions"] = counters.get("predictor_endpoint_promotions", 0) + len(promoted_starts)
            for promoted_key in promoted_starts:
                promoted_target_predictors[promoted_key]["predictor"][predictor_id] = original_predictor_counts[predictor_id]
                promoted_target_predictors[promoted_key]["overlap"][predictor_id] = overlap

        target_predictors = sorted(
            promoted_target_predictors.items(),
            key=lambda item: sum(item[1]["predictor"].values()),
            reverse=True,
        )
        explicit_predictor_total = len(target_predictors)
        max_explicit_predictor_count = (
            sum(target_predictors[0][1]["predictor"].values()) if target_predictors else 0
        )
        min_explicit_predictor_count = (
            sum(target_predictors[-1][1]["predictor"].values()) if target_predictors else 0
        )
        counters["targets_sorted_by_predictor_count"] = counters.get("targets_sorted_by_predictor_count", 0) + 1

        for predictor_index, (rollup_key, rollup_support) in enumerate(target_predictors, start=1):
            if isinstance(rollup_key, tuple):
                nhop_path, force_same_endpoint_reverse = rollup_key
            else:
                nhop_path, force_same_endpoint_reverse = rollup_key, False
            explicit_predictor_count = sum(rollup_support["predictor"].values())
            counters["predictor_expansion_requests"] = counters.get("predictor_expansion_requests", 0) + 1

            def visit_state(state_signature):
                nonlocal variants_pruned_lower_bound
                nonlocal next_state_token_id

                if not endpoints_within_target_pair(state_signature[0], state_signature[-1], target_type1, target_type2):
                    return False

                state_id_parts = []
                for token in state_signature:
                    token_id = state_token_ids.get(token)
                    if token_id is None:
                        token_id = next_state_token_id
                        state_token_ids[token] = token_id
                        next_state_token_id += 1
                    state_id_parts.append(token_id)
                state_id = tuple(state_id_parts)

                if state_id in global_pruned_states:
                    counters["candidate_state_revisits_pruned"] = counters.get("candidate_state_revisits_pruned", 0) + 1
                    return True
                proposed = lower_bound_state_counts[state_id] + explicit_predictor_count
                if max_predictor_count is not None and proposed > max_predictor_count:
                    variants_pruned_lower_bound += 1
                    global_pruned_states.add(state_id)
                    lower_bound_state_counts.pop(state_id, None)
                    counters["candidate_states_branch_pruned"] = counters.get("candidate_states_branch_pruned", 0) + 1
                    return True
                lower_bound_state_counts[state_id] = proposed
                return False

            def visit_variant(variant_state_ids):
                nonlocal variants_pruned_lower_bound
                if variant_state_ids in pruned_variants:
                    counters["candidate_variant_revisits_pruned"] = counters.get("candidate_variant_revisits_pruned", 0) + 1
                    return True
                support_map = aggregated_predictor_support[variant_state_ids]
                proposed = sum(support_map.values())
                for predictor_id, predictor_count in rollup_support["predictor"].items():
                    if predictor_id not in support_map:
                        proposed += predictor_count
                if max_predictor_count is not None and proposed > max_predictor_count:
                    variants_pruned_lower_bound += 1
                    for pruned_state_ids in canonical_variant_ancestor_closure_state_ids_for_typepair(
                        variant_state_ids,
                        target_type1,
                        target_type2,
                    ):
                        pruned_variants.add(pruned_state_ids)
                        aggregated_predictor_support.pop(pruned_state_ids, None)
                        aggregated_overlap_support.pop(pruned_state_ids, None)
                    counters["candidate_variants_branch_pruned"] = counters.get("candidate_variants_branch_pruned", 0) + 1
                    return True
                for predictor_id, predictor_count in rollup_support["predictor"].items():
                    support_map[predictor_id] = predictor_count
                for predictor_id, overlap in rollup_support["overlap"].items():
                    aggregated_overlap_support[variant_state_ids][predictor_id] = overlap
                counters["candidate_variants_accepted"] = counters.get("candidate_variants_accepted", 0) + 1
                return False

            traverse_canonical_variants_for_typepair_pruned(
                nhop_path,
                target_type1,
                target_type2,
                visit_variant,
                visit_state=visit_state,
                force_same_endpoint_reverse=force_same_endpoint_reverse,
            )

            now = time.time()
            if start_time and now >= next_progress_at:
                next_progress_at = now + 30.0
                elapsed_target = max(now - target_t0, 1e-9)
                accepted_delta = counters.get("candidate_variants_accepted", 0) - target_accepted_at_start
                branch_pruned_delta = counters.get("candidate_variants_branch_pruned", 0) - target_branch_pruned_at_start
                print_profile_event(
                    "build_target_candidates",
                    start_time,
                    target_index=target_index,
                    total_targets=total_targets,
                    current_target=onehop_path,
                    predictors_done=predictor_index,
                    predictor_total=explicit_predictor_total,
                    predictor_rate=round(predictor_index / elapsed_target, 2),
                    latest_predictor_explicit_count=explicit_predictor_count,
                    max_predictor_explicit_count=max_explicit_predictor_count,
                    min_predictor_explicit_count=min_explicit_predictor_count,
                    active_variants=len(aggregated_overlap_support),
                    pruned_variants=len(pruned_variants),
                    lower_bound_pruned=variants_pruned_lower_bound,
                    accepted_variants=accepted_delta,
                    branch_pruned_variants=branch_pruned_delta,
                )
                write_progress_file(
                    progress_file,
                    build_progress_payload(
                        type1,
                        type2,
                        "build_target_candidates",
                        start_time,
                        counters,
                        stage_timings,
                        target_index=target_index,
                        total_targets=total_targets,
                        current_target=onehop_path,
                        predictors_done=predictor_index,
                        predictor_total=explicit_predictor_total,
                        predictor_rate=round(predictor_index / elapsed_target, 2),
                        latest_predictor_explicit_count=explicit_predictor_count,
                        max_predictor_explicit_count=max_explicit_predictor_count,
                        min_predictor_explicit_count=min_explicit_predictor_count,
                        active_variants=len(aggregated_overlap_support),
                        pruned_variants=len(pruned_variants),
                        lower_bound_pruned=variants_pruned_lower_bound,
                        accepted_variants=accepted_delta,
                        branch_pruned_variants=branch_pruned_delta,
                        candidate_build_elapsed_s=round(time.time() - target_t0, 3),
                    ),
                )

        candidate_rows = {
            canonical_variant_metapath_from_state_ids(variant_state_ids): (
                sum(predictor_support.values()),
                sum(aggregated_overlap_support[variant_state_ids].values()),
            )
            for variant_state_ids, predictor_support in aggregated_predictor_support.items()
            if predictor_support
        }
        candidate_rows_by_target[onehop_path] = candidate_rows
        all_candidate_variants.update(candidate_rows)

        counters["targets_candidate_built"] = counters.get("targets_candidate_built", 0) + 1
        counters["candidate_variants"] = counters.get("candidate_variants", 0) + len(candidate_rows)
        counters["precision_pruned_lower_bound"] = counters.get("precision_pruned_lower_bound", 0) + variants_pruned_lower_bound
        elapsed_target = max(time.time() - target_t0, 1e-9)
        accepted_delta = counters.get("candidate_variants_accepted", 0) - target_accepted_at_start
        branch_pruned_delta = counters.get("candidate_variants_branch_pruned", 0) - target_branch_pruned_at_start

        print_profile_event(
            "build_target_candidates",
            start_time,
            target_index=target_index,
            total_targets=total_targets,
            explicit_predictors=len(nhop_overlaps),
            predictor_rate=round(explicit_predictor_total / elapsed_target, 2),
            max_predictor_explicit_count=max_explicit_predictor_count,
            min_predictor_explicit_count=min_explicit_predictor_count,
            candidate_variants=len(candidate_rows),
            lower_bound_pruned=variants_pruned_lower_bound,
            accepted_variants=accepted_delta,
            branch_pruned_variants=branch_pruned_delta,
            candidate_build_elapsed_s=round(time.time() - target_t0, 3),
        )
        write_progress_file(
            progress_file,
            build_progress_payload(
                type1,
                type2,
                "build_target_candidates",
                start_time,
                counters,
                stage_timings,
                target_index=target_index,
                total_targets=total_targets,
                current_target=onehop_path,
                explicit_predictors=len(nhop_overlaps),
                predictor_rate=round(explicit_predictor_total / elapsed_target, 2),
                max_predictor_explicit_count=max_explicit_predictor_count,
                min_predictor_explicit_count=min_explicit_predictor_count,
                candidate_variants=len(candidate_rows),
                lower_bound_pruned=variants_pruned_lower_bound,
                accepted_variants=accepted_delta,
                branch_pruned_variants=branch_pruned_delta,
                candidate_build_elapsed_s=round(time.time() - target_t0, 3),
            ),
        )

    stage_timings["build_target_candidates"] = stage_timings.get("build_target_candidates", 0.0) + (time.time() - t0)
    counters["candidate_variant_union_size"] = len(all_candidate_variants)
    return candidate_rows_by_target, all_candidate_variants


def build_candidate_variants_for_targets_single_pass(
    onehop_to_overlaps,
    explicit_items,
    original_predictor_counts,
    target_variant_counts,
    type1,
    type2,
    min_precision,
    counters,
):
    """Build exact global predictor counts and target overlaps in one predictor-centric pass."""
    candidate_rows_by_target = {target: {} for target in onehop_to_overlaps}
    all_candidate_variants = set()

    predictor_paths_by_id = defaultdict(set)
    for path, count in explicit_items:
        if len(parse_metapath(path)[1]) == 0:
            continue
        if not promote_metapath_endpoints_to_typepair_starts(path, type1, type2):
            continue
        predictor_paths_by_id[original_predictor_identity(path)].add(path)

    predictor_target_overlaps = defaultdict(dict)
    for target, nhop_overlaps in onehop_to_overlaps.items():
        for nhop_path, overlap in nhop_overlaps.items():
            predictor_id = original_predictor_identity(nhop_path)
            previous = predictor_target_overlaps[predictor_id].get(target)
            if previous is None or overlap > previous:
                predictor_target_overlaps[predictor_id][target] = overlap

    max_predictor_count_by_target = {
        target: get_max_predictor_count_for_precision(target_variant_counts.get(target, 0), min_precision)
        for target in onehop_to_overlaps
    }
    target_endpoint_types = {
        target: get_metapath_endpoints(target)
        for target in onehop_to_overlaps
    }

    global_variant_support = defaultdict(dict)
    target_variant_overlap_support = defaultdict(lambda: defaultdict(dict))
    active_targets_by_variant = defaultdict(set)
    pruned_variants_by_target = defaultdict(set)

    predictor_items = sorted(
        predictor_paths_by_id.keys(),
        key=lambda predictor_id: original_predictor_counts.get(predictor_id, 0),
        reverse=True,
    )
    counters["predictor_single_pass_count"] = len(predictor_items)

    def prune_variant_for_target(variant_state_ids, target):
        target_type1, target_type2 = target_endpoint_types[target]
        for pruned_state_ids in canonical_variant_ancestor_closure_state_ids_for_typepair(
            variant_state_ids,
            target_type1,
            target_type2,
        ):
            pruned_variants_by_target[target].add(pruned_state_ids)
            target_variant_overlap_support[target].pop(pruned_state_ids, None)
            active_targets_by_variant[pruned_state_ids].discard(target)

    for predictor_id in predictor_items:
        predictor_count = original_predictor_counts.get(predictor_id, 0)
        target_overlaps = predictor_target_overlaps.get(predictor_id, {})
        seen_variants = set()

        def visit_variant(variant_state_ids):
            if variant_state_ids in seen_variants:
                counters["candidate_variant_revisits_pruned"] = counters.get("candidate_variant_revisits_pruned", 0) + 1
                return True

            seen_variants.add(variant_state_ids)
            support_map = global_variant_support[variant_state_ids]
            if predictor_id not in support_map:
                support_map[predictor_id] = predictor_count

            current_predictor_count = sum(support_map.values())

            if active_targets_by_variant[variant_state_ids]:
                for target in list(active_targets_by_variant[variant_state_ids]):
                    if variant_state_ids in pruned_variants_by_target[target]:
                        continue
                    max_predictor_count = max_predictor_count_by_target[target]
                    if max_predictor_count is not None and current_predictor_count > max_predictor_count:
                        prune_variant_for_target(variant_state_ids, target)

            for target, overlap in target_overlaps.items():
                if variant_state_ids in pruned_variants_by_target[target]:
                    continue
                max_predictor_count = max_predictor_count_by_target[target]
                if max_predictor_count is not None and current_predictor_count > max_predictor_count:
                    prune_variant_for_target(variant_state_ids, target)
                    continue
                target_variant_overlap_support[target][variant_state_ids][predictor_id] = overlap
                active_targets_by_variant[variant_state_ids].add(target)
                counters["candidate_variants_accepted"] = counters.get("candidate_variants_accepted", 0) + 1
            return False

        for nhop_path in sorted(predictor_paths_by_id.get(predictor_id, ())):
            traverse_canonical_variants_for_typepair_pruned(
                nhop_path,
                type1,
                type2,
                visit_variant,
            )

    for target, variant_overlap_support in target_variant_overlap_support.items():
        rows = {}
        for variant_state_ids, overlap_support in variant_overlap_support.items():
            if not overlap_support:
                continue
            predictor_support = global_variant_support.get(variant_state_ids)
            if not predictor_support:
                continue
            rows[canonical_variant_metapath_from_state_ids(variant_state_ids)] = (
                sum(predictor_support.values()),
                sum(overlap_support.values()),
            )
        candidate_rows_by_target[target] = rows
        all_candidate_variants.update(rows)

    counters["candidate_variant_union_size"] = len(all_candidate_variants)
    return candidate_rows_by_target, all_candidate_variants


def group_type_pair(type1, type2, file_list, output_dir, n_hops, explicit_items,
                    type_node_counts=None, min_count=0, min_precision=0.0,
                    excluded_types=None, excluded_predicates=None,
                    progress_file=None):
    """Group all N-hop results for 1-hop metapaths between a type pair."""
    excluded_types = excluded_types or set()
    excluded_predicates = excluded_predicates or set()
    start_time = time.time()
    counters = {
        "file_count": len(file_list),
        "explicit_item_count": len(explicit_items),
        "min_count": min_count,
        "min_precision": min_precision,
    }
    stage_timings = {}

    print(f"Grouping for type pair: ({type1}, {type2})")
    print(f"Min count filter: {min_count}")
    print(f"Min precision filter: {min_precision}")
    if excluded_types:
        print(f"Excluded types: {sorted(excluded_types)}")
    if excluded_predicates:
        print(f"Excluded predicates: {sorted(excluded_predicates)}")
    print(f"Using explicit path shard: {len(explicit_items)} paths")
    write_progress_file(
        progress_file,
        build_progress_payload(type1, type2, "start", start_time, counters, stage_timings),
    )

    onehop_to_overlaps = defaultdict(lambda: defaultdict(int))
    target_expansion_cache = {}
    global_explicit_count_by_path = {}
    files_processed = 0
    rows_found = 0
    rows_scanned = 0
    t_scan = time.time()

    for file_path in file_list:
        files_processed += 1
        with open(file_path, 'r') as f:
            f.readline()
            for line in f:
                rows_scanned += 1
                parts = line.strip().split('\t')
                if len(parts) != 6:
                    continue

                nhop_path = parts[0]
                nhop_count = int(parts[1])
                onehop_path = parts[2]
                onehop_count = int(parts[3])
                overlap = int(parts[4])

                previous_nhop_count = global_explicit_count_by_path.get(nhop_path)
                if previous_nhop_count is None:
                    global_explicit_count_by_path[nhop_path] = nhop_count
                else:
                    assert previous_nhop_count == nhop_count, (
                        f"Inconsistent explicit count for {nhop_path}: "
                        f"{previous_nhop_count} vs {nhop_count}"
                    )

                previous_onehop_count = global_explicit_count_by_path.get(onehop_path)
                if previous_onehop_count is None:
                    global_explicit_count_by_path[onehop_path] = onehop_count
                else:
                    assert previous_onehop_count == onehop_count, (
                        f"Inconsistent explicit count for {onehop_path}: "
                        f"{previous_onehop_count} vs {onehop_count}"
                    )

                onehop_left_type, onehop_right_type = get_metapath_endpoints(onehop_path)
                if not endpoints_within_target_pair(onehop_left_type, onehop_right_type, type1, type2):
                    continue

                rows_found += 1
                onehop_variants = target_expansion_cache.setdefault(
                    onehop_path,
                    list(expand_metapath_to_typepair_variants(onehop_path, type1, type2)),
                )
                for onehop_variant in onehop_variants:
                    onehop_to_overlaps[onehop_variant][nhop_path] += overlap

        if files_processed % 100 == 0 or files_processed == len(file_list):
            print_profile_event(
                "scan_overlap_files",
                start_time,
                files_processed=files_processed,
                file_total=len(file_list),
                rows_scanned=rows_scanned,
                rows_matched=rows_found,
                target_buckets=len(onehop_to_overlaps),
            )
            write_progress_file(
                progress_file,
                build_progress_payload(
                    type1,
                    type2,
                    "scan_overlap_files",
                    start_time,
                    counters,
                    stage_timings,
                    files_processed=files_processed,
                    file_total=len(file_list),
                    rows_scanned=rows_scanned,
                    rows_matched=rows_found,
                    target_buckets=len(onehop_to_overlaps),
                ),
            )

    stage_timings["scan_overlap_files"] = stage_timings.get("scan_overlap_files", 0.0) + (time.time() - t_scan)
    counters["rows_scanned"] = rows_scanned
    counters["rows_matched"] = rows_found
    counters["target_bucket_count"] = len(onehop_to_overlaps)

    global_explicit_items = sorted(global_explicit_count_by_path.items())
    original_predictor_counts = {}
    for path, count in global_explicit_items:
        predictor_id = original_predictor_identity(path)
        previous = original_predictor_counts.get(predictor_id)
        if previous is None:
            original_predictor_counts[predictor_id] = count
        else:
            assert previous == count, (
                f"Inconsistent explicit count for original predictor identity {predictor_id}: "
                f"{previous} vs {count}"
            )
    rolled_predictor_counts = group_predictor_items_by_promoted_start(
        global_explicit_items,
        n_hops,
        type1,
        type2,
        counters,
    )
    target_variant_counts, target_expansion_cache = build_target_variant_counts(
        global_explicit_items,
        type1,
        type2,
        start_time=start_time,
        progress_file=progress_file,
        counters=counters,
        stage_timings=stage_timings,
    )
    print(f"Local target variants: {len(target_variant_counts)}")
    total_possible_for_pair = compute_total_possible(type1, type2, type_node_counts) if type_node_counts else 0
    print(f"Total possible for ({type1}, {type2}): {total_possible_for_pair:,}")

    print(f"\n✓ Found {rows_found} total rows")
    print(f"  Unique 1-hop metapaths: {len(onehop_to_overlaps)}")
    if not onehop_to_overlaps:
        print("\nWARNING: No matching 1-hop metapaths found. Skipping output.")
        return

    filtered_onehop_to_overlaps = {}
    skipped_excluded_targets = 0
    skipped_pseudo_targets = 0
    skipped_targets = []
    for onehop_path, nhop_overlaps in onehop_to_overlaps.items():
        if should_exclude_metapath(onehop_path, excluded_types, excluded_predicates):
            skipped_excluded_targets += 1
            skipped_targets.append((onehop_path, "excluded"))
            continue
        if contains_pseudo_type(onehop_path):
            skipped_pseudo_targets += 1
            skipped_targets.append((onehop_path, "pseudo"))
            continue
        filtered_onehop_to_overlaps[onehop_path] = nhop_overlaps

    if skipped_excluded_targets or skipped_pseudo_targets:
        print(
            f"  Filtered target buckets before candidate build: "
            f"excluded={skipped_excluded_targets} pseudo={skipped_pseudo_targets}"
        )
        for target, reason in skipped_targets[:10]:
            print(f"    Skipped target ({reason}): {target}")

    onehop_to_overlaps = filtered_onehop_to_overlaps
    counters["target_buckets_filtered_excluded"] = skipped_excluded_targets
    counters["target_buckets_filtered_pseudo"] = skipped_pseudo_targets
    counters["target_bucket_count_after_filter"] = len(onehop_to_overlaps)

    if not onehop_to_overlaps:
        print("\nWARNING: No matching 1-hop metapaths remain after target filtering. Skipping output.")
        write_progress_file(
            progress_file,
            build_progress_payload(
                type1,
                type2,
                "complete",
                start_time,
                counters,
                stage_timings,
                total_targets=0,
            ),
        )
        return

    write_progress_file(
        progress_file,
        build_progress_payload(
            type1,
            type2,
            "pre_build_target_candidates",
            start_time,
            counters,
            stage_timings,
            target_buckets=len(onehop_to_overlaps),
            rows_scanned=rows_scanned,
            rows_matched=rows_found,
            target_buckets_filtered_excluded=skipped_excluded_targets,
            target_buckets_filtered_pseudo=skipped_pseudo_targets,
        ),
    )

    candidate_rows_by_target, candidate_variants = build_candidate_variants_for_targets_single_pass(
        onehop_to_overlaps,
        global_explicit_items,
        original_predictor_counts,
        target_variant_counts,
        type1,
        type2,
        min_precision,
        counters,
    )
    print(f"\nCandidate predictor variants across targets: {len(candidate_variants)}")
    write_progress_file(
        progress_file,
        build_progress_payload(
            type1,
            type2,
            "pre_process_targets",
            start_time,
            counters,
            stage_timings,
            total_targets=len(onehop_to_overlaps),
            candidate_variants=len(candidate_variants),
        ),
    )

    counters["targets_processed"] = 0
    counters["output_files_written"] = 0
    counters["output_rows_written"] = 0

    total_targets = len(onehop_to_overlaps)
    for target_index, (onehop_path, nhop_overlaps) in enumerate(onehop_to_overlaps.items(), start=1):
        print(f"\nProcessing 1-hop: {onehop_path}")
        print(f"  {len(nhop_overlaps)} explicit N-hop paths with overlap")
        t_target = time.time()

        onehop_count_global = target_variant_counts.get(onehop_path, 0)
        aggregated_rows = candidate_rows_by_target.get(onehop_path, {})
        explicit_predictors_for_target = len(nhop_overlaps)

        print(f"  Aggregated predictor variants: {len(aggregated_rows)}")

        if should_exclude_metapath(onehop_path, excluded_types, excluded_predicates):
            print("  Skipping (excluded type/predicate in 1-hop)")
            continue
        if contains_pseudo_type(onehop_path):
            print("  Skipping (pseudo-type in 1-hop)")
            continue

        safe_filename = onehop_path.replace('|', '_').replace(':', '_').replace(' ', '_')
        output_file = f"{output_dir}/{safe_filename}.tsv.zst"

        rows_written = 0
        rows_filtered_excluded = 0
        rows_filtered_pseudo = 0
        rows_filtered_count = 0
        rows_filtered_precision = 0
        t_write = time.time()

        with zstandard.open(output_file, 'wt') as out:
            out.write("predictor_metapath\tpredictor_count\toverlap\ttotal_possible\t")
            out.write("precision\trecall\tf1\tmcc\tspecificity\tnpv\n")

            sorted_items = sorted(aggregated_rows.items(), key=lambda x: x[1][1], reverse=True)
            for nhop_variant, (nhop_count, overlap) in sorted_items:
                if should_exclude_metapath(nhop_variant, excluded_types, excluded_predicates):
                    rows_filtered_excluded += 1
                    continue
                if contains_pseudo_type(nhop_variant):
                    rows_filtered_pseudo += 1
                    continue

                if nhop_count <= 0 or nhop_count < min_count:
                    rows_filtered_count += 1
                    continue

                metrics = calculate_metrics(nhop_count, onehop_count_global, overlap, total_possible_for_pair)
                if metrics['precision'] < min_precision:
                    rows_filtered_precision += 1
                    continue

                out.write(f"{nhop_variant}\t{nhop_count}\t{overlap}\t{total_possible_for_pair}\t")
                out.write(f"{metrics['precision']:.6f}\t{metrics['recall']:.6f}\t")
                out.write(f"{metrics['f1']:.6f}\t{metrics['mcc']:.6f}\t")
                out.write(f"{metrics['specificity']:.6f}\t{metrics['npv']:.6f}\n")
                rows_written += 1
        stage_timings["write_output"] = stage_timings.get("write_output", 0.0) + (time.time() - t_write)

        print(f"  Written {rows_written} rows to {output_file}")
        if rows_filtered_excluded > 0:
            print(f"    Filtered (excluded): {rows_filtered_excluded}")
        if rows_filtered_pseudo > 0:
            print(f"    Filtered (pseudo-type): {rows_filtered_pseudo}")
        if rows_filtered_count > 0:
            print(f"    Filtered (count < {min_count}): {rows_filtered_count}")
        if rows_filtered_precision > 0:
            print(f"    Filtered (precision < {min_precision}): {rows_filtered_precision}")

        stage_timings["process_targets_total"] = stage_timings.get("process_targets_total", 0.0) + (time.time() - t_target)
        counters["targets_processed"] += 1
        counters["output_files_written"] += 1
        counters["output_rows_written"] += rows_written
        counters["rows_filtered_excluded"] = counters.get("rows_filtered_excluded", 0) + rows_filtered_excluded
        counters["rows_filtered_pseudo"] = counters.get("rows_filtered_pseudo", 0) + rows_filtered_pseudo
        counters["rows_filtered_count"] = counters.get("rows_filtered_count", 0) + rows_filtered_count
        counters["rows_filtered_precision"] = counters.get("rows_filtered_precision", 0) + rows_filtered_precision

        print_profile_event(
            "process_target",
            start_time,
            target_index=target_index,
            total_targets=total_targets,
            explicit_predictors=explicit_predictors_for_target,
            aggregated_predictor_variants=len(aggregated_rows),
            rows_written=rows_written,
        )
        write_progress_file(
            progress_file,
            build_progress_payload(
                type1,
                type2,
                "process_target",
                start_time,
                counters,
                stage_timings,
                target_index=target_index,
                total_targets=total_targets,
                current_target=onehop_path,
                explicit_predictors=explicit_predictors_for_target,
                aggregated_predictor_variants=len(aggregated_rows),
                rows_written=rows_written,
            ),
        )

    print(f"\n✓ All 1-hop metapaths processed for type pair ({type1}, {type2})")
    write_progress_file(
        progress_file,
        build_progress_payload(
            type1,
            type2,
            "complete",
            start_time,
            counters,
            stage_timings,
            total_targets=total_targets,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description='Group results for all 1-hop metapaths between a type pair')
    parser.add_argument('--type1', type=str, required=True, help='First type')
    parser.add_argument('--type2', type=str, required=True, help='Second type')
    parser.add_argument('--file-list', type=str, required=True, help='Path to file containing list of result files to scan')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for grouped results')
    parser.add_argument('--n-hops', type=int, required=True, help='Number of hops')
    parser.add_argument('--explicit-shards-dir', type=str, required=True, help='Directory containing type-pair explicit-count shards')
    parser.add_argument('--type-node-counts', type=str, required=True, help='Path to type_node_counts.json')
    parser.add_argument('--min-count', type=int, default=0, help='Minimum N-hop count to include')
    parser.add_argument('--min-precision', type=float, default=0.0, help='Minimum precision to include')
    parser.add_argument('--exclude-types', type=str, default='Entity,ThingWithTaxon', help='Comma-separated list of node types to exclude')
    parser.add_argument('--exclude-predicates', type=str, default='related_to_at_instance_level,related_to_at_concept_level', help='Comma-separated list of predicates to exclude')
    parser.add_argument('--progress-file', type=str, default=None, help='Optional path to write per-job profiling/progress JSON')

    args = parser.parse_args()

    excluded_types = set(t.strip() for t in args.exclude_types.split(',') if t.strip())
    excluded_predicates = set(p.strip() for p in args.exclude_predicates.split(',') if p.strip())

    with open(args.file_list, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    explicit_items = load_typepair_explicit_counts(args.explicit_shards_dir, args.type1, args.type2)
    with open(args.type_node_counts, 'r') as f:
        type_node_counts = json.load(f)

    group_type_pair(
        type1=args.type1,
        type2=args.type2,
        file_list=file_list,
        output_dir=args.output_dir,
        n_hops=args.n_hops,
        explicit_items=explicit_items,
        type_node_counts=type_node_counts,
        min_count=args.min_count,
        min_precision=args.min_precision,
        excluded_types=excluded_types,
        excluded_predicates=excluded_predicates,
        progress_file=args.progress_file,
    )


if __name__ == "__main__":
    main()
