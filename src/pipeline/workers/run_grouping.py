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
    expand_metapath_to_typepair_variants,
    expand_metapath_to_variants,
    promote_metapath_endpoints_to_typepair_starts,
    traverse_metapath_variants_for_typepair_pruned,
    get_type_variants,
    calculate_metrics,
    parse_compound_predicate,
    parse_metapath,
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
    target_variant_counts = defaultdict(int)
    target_expansion_cache = {}
    counters = counters if counters is not None else {}
    stage_timings = stage_timings if stage_timings is not None else {}

    t0 = time.time()
    next_progress_at = t0 + 30.0

    for idx, (path, count) in enumerate(explicit_items, start=1):
        _, predicates, _ = parse_metapath(path)
        if len(predicates) != 1:
            continue

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
            target_variant_counts[variant] += count

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
                target_variants=len(target_variant_counts),
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
                    target_variants=len(target_variant_counts),
                    target_cache=len(target_expansion_cache),
                    avg_target_variants=avg_target_variants,
                ),
            )

    stage_timings["build_target_variant_counts"] = stage_timings.get("build_target_variant_counts", 0.0) + (time.time() - t0)
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
    grouped_counts = defaultdict(int)
    counters = counters if counters is not None else {}

    for path, count in explicit_items:
        if len(parse_metapath(path)[1]) != n_hops:
            continue
        promoted_starts = promote_metapath_endpoints_to_typepair_starts(path, type1, type2)
        counters["predictor_endpoint_promotions"] = counters.get("predictor_endpoint_promotions", 0) + len(promoted_starts)
        for promoted_path in promoted_starts:
            grouped_counts[promoted_path] += count

    counters["grouped_predictor_count"] = len(grouped_counts)
    return dict(grouped_counts)


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


def build_candidate_variants_for_targets(onehop_to_overlaps, explicit_count_by_path, target_variant_counts,
                                         type1, type2, min_precision, predictor_expansion_cache,
                                         start_time, progress_file, counters, stage_timings):
    """Build target-aware predictor candidates using overlap lower bounds."""
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

    for target_index, (onehop_path, nhop_overlaps) in enumerate(target_items, start=1):
        target_t0 = time.time()
        onehop_count_global = target_variant_counts.get(onehop_path, 0)
        target_type1, target_type2 = get_metapath_endpoints(onehop_path)
        max_predictor_count = get_max_predictor_count_for_precision(onehop_count_global, min_precision)
        aggregated_overlaps = defaultdict(int)
        lower_bound_counts = defaultdict(int)
        pruned_variants = set()
        lower_bound_state_counts = defaultdict(int)
        variants_pruned_lower_bound = 0
        target_branch_pruned_at_start = counters.get("candidate_variants_branch_pruned", 0)
        target_accepted_at_start = counters.get("candidate_variants_accepted", 0)
        raw_target_predictors = sorted(
            nhop_overlaps.items(),
            key=lambda item: explicit_count_by_path.get(item[0], 0),
            reverse=True,
        )
        promoted_target_predictors = {}
        for nhop_path, overlap in raw_target_predictors:
            explicit_predictor_count = explicit_count_by_path.get(nhop_path, 0)
            promoted_starts = promote_metapath_endpoints_to_typepair_starts(nhop_path, type1, type2)
            counters["predictor_endpoint_promotions"] = counters.get("predictor_endpoint_promotions", 0) + len(promoted_starts)
            for promoted_path in promoted_starts:
                if promoted_path not in promoted_target_predictors:
                    promoted_target_predictors[promoted_path] = [0, 0]
                promoted_target_predictors[promoted_path][0] += overlap
                promoted_target_predictors[promoted_path][1] += explicit_predictor_count

        target_predictors = sorted(
            promoted_target_predictors.items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
        explicit_predictor_total = len(target_predictors)
        max_explicit_predictor_count = (
            target_predictors[0][1][1] if target_predictors else 0
        )
        min_explicit_predictor_count = (
            target_predictors[-1][1][1] if target_predictors else 0
        )
        counters["targets_sorted_by_predictor_count"] = counters.get("targets_sorted_by_predictor_count", 0) + 1

        for predictor_index, (nhop_path, predictor_payload) in enumerate(target_predictors, start=1):
            overlap, explicit_predictor_count = predictor_payload
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

            def visit_variant(variant):
                nonlocal variants_pruned_lower_bound
                variant_left_type, variant_right_type = get_metapath_endpoints(variant)
                if not endpoints_exact_match(
                    variant_left_type,
                    variant_right_type,
                    target_type1,
                    target_type2,
                ):
                    return False
                if variant in pruned_variants:
                    counters["candidate_variant_revisits_pruned"] = counters.get("candidate_variant_revisits_pruned", 0) + 1
                    return True
                proposed = lower_bound_counts[variant] + explicit_predictor_count
                if max_predictor_count is not None and proposed > max_predictor_count:
                    variants_pruned_lower_bound += 1
                    pruned_variants.add(variant)
                    lower_bound_counts.pop(variant, None)
                    aggregated_overlaps.pop(variant, None)
                    counters["candidate_variants_branch_pruned"] = counters.get("candidate_variants_branch_pruned", 0) + 1
                    return True
                lower_bound_counts[variant] = proposed
                aggregated_overlaps[variant] += overlap
                counters["candidate_variants_accepted"] = counters.get("candidate_variants_accepted", 0) + 1
                return False

            traverse_metapath_variants_for_typepair_pruned(
                nhop_path,
                target_type1,
                target_type2,
                visit_variant,
                visit_state=visit_state,
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
                    active_variants=len(aggregated_overlaps),
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
                        active_variants=len(aggregated_overlaps),
                        pruned_variants=len(pruned_variants),
                        lower_bound_pruned=variants_pruned_lower_bound,
                        accepted_variants=accepted_delta,
                        branch_pruned_variants=branch_pruned_delta,
                        candidate_build_elapsed_s=round(time.time() - target_t0, 3),
                    ),
                )

        candidate_rows = {
            variant: overlap
            for variant, overlap in aggregated_overlaps.items()
            if lower_bound_counts.get(variant, 0) > 0
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


def compute_exact_predictor_counts(explicit_items, candidate_variants, n_hops, type1, type2,
                                   predictor_expansion_cache, start_time, progress_file,
                                   counters, stage_timings):
    """Compute exact predictor counts only for candidate variants that survived lower-bound pruning."""
    t0 = time.time()
    next_progress_at = t0 + 30.0
    candidate_variants = set(candidate_variants)
    exact_counts = defaultdict(int)
    predictor_items = [
        (path, count) for path, count in explicit_items if len(parse_metapath(path)[1]) == n_hops
    ]

    for idx, (path, count) in enumerate(predictor_items, start=1):
        variants = expand_predictor_variants_for_typepair(path, type1, type2, predictor_expansion_cache, counters)
        matched = False
        for variant in variants:
            if variant in candidate_variants:
                exact_counts[variant] += count
                matched = True
        if matched:
            counters["predictor_items_contributing_to_candidates"] = counters.get("predictor_items_contributing_to_candidates", 0) + 1

        now = time.time()
        if start_time and (idx == len(predictor_items) or now >= next_progress_at):
            next_progress_at = now + 30.0
            print_profile_event(
                "compute_exact_predictor_counts",
                start_time,
                predictors_done=idx,
                predictor_total=len(predictor_items),
                exact_variants=len(exact_counts),
                candidate_variants=len(candidate_variants),
            )
            write_progress_file(
                progress_file,
                build_progress_payload(
                    type1,
                    type2,
                    "compute_exact_predictor_counts",
                    start_time,
                    counters,
                    stage_timings,
                    predictors_done=idx,
                    predictor_total=len(predictor_items),
                    exact_variants=len(exact_counts),
                    candidate_variants=len(candidate_variants),
                ),
            )

    stage_timings["compute_exact_predictor_counts"] = stage_timings.get("compute_exact_predictor_counts", 0.0) + (time.time() - t0)
    counters["exact_predictor_variant_count_size"] = len(exact_counts)
    return dict(exact_counts)


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

    explicit_count_by_path = dict(explicit_items)
    target_variant_counts, target_expansion_cache = build_target_variant_counts(
        explicit_items,
        type1,
        type2,
        start_time=start_time,
        progress_file=progress_file,
        counters=counters,
        stage_timings=stage_timings,
    )
    print(f"Local target variants: {len(target_variant_counts)}")
    predictor_expansion_cache = {}

    total_possible_for_pair = compute_total_possible(type1, type2, type_node_counts) if type_node_counts else 0
    print(f"Total possible for ({type1}, {type2}): {total_possible_for_pair:,}")

    onehop_to_overlaps = defaultdict(lambda: defaultdict(int))
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
                onehop_path = parts[2]
                overlap = int(parts[4])

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

    candidate_rows_by_target, candidate_variants = build_candidate_variants_for_targets(
        onehop_to_overlaps,
        explicit_count_by_path,
        target_variant_counts,
        type1,
        type2,
        min_precision,
        predictor_expansion_cache,
        start_time,
        progress_file,
        counters,
        stage_timings,
    )
    print(f"\nCandidate predictor variants across targets: {len(candidate_variants)}")
    write_progress_file(
        progress_file,
        build_progress_payload(
            type1,
            type2,
            "pre_compute_exact_predictor_counts",
            start_time,
            counters,
            stage_timings,
            target_buckets=len(onehop_to_overlaps),
            candidate_variants=len(candidate_variants),
        ),
    )

    exact_predictor_counts = compute_exact_predictor_counts(
        explicit_items,
        candidate_variants,
        n_hops,
        type1,
        type2,
        predictor_expansion_cache,
        start_time,
        progress_file,
        counters,
        stage_timings,
    )
    print(f"Exact candidate predictor counts computed: {len(exact_predictor_counts)}")
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
            exact_variants=len(exact_predictor_counts),
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
        aggregated_overlaps = candidate_rows_by_target.get(onehop_path, {})
        explicit_predictors_for_target = len(nhop_overlaps)

        print(f"  Aggregated predictor variants: {len(aggregated_overlaps)}")

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

            sorted_items = sorted(aggregated_overlaps.items(), key=lambda x: x[1], reverse=True)
            for nhop_variant, overlap in sorted_items:
                if should_exclude_metapath(nhop_variant, excluded_types, excluded_predicates):
                    rows_filtered_excluded += 1
                    continue
                if contains_pseudo_type(nhop_variant):
                    rows_filtered_pseudo += 1
                    continue

                nhop_count = exact_predictor_counts.get(nhop_variant, 0)
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
            aggregated_predictor_variants=len(aggregated_overlaps),
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
                aggregated_predictor_variants=len(aggregated_overlaps),
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
