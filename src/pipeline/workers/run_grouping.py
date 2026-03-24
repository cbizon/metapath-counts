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

import graphblas as gb
import zstandard

from library.hierarchy import get_type_ancestors
from library.aggregation import (
    expand_metapath_to_typepair_variants,
    get_type_variants,
    calculate_metrics,
    parse_compound_predicate,
    parse_metapath,
    original_predictor_identity,
)
from library.unified_index import (
    build_target_pair_set,
    build_unified_type_offsets,
    load_base_matrices,
    reconstruct_nhop_matrix,
    reconstruct_prefix_matrix,
    remap_nhop_to_unified,
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


def _detect_metapath_orientation(metapath, type1, type2):
    """Detect if a metapath's endpoints are in (type1, type2) or (type2, type1) order.

    Returns:
        'forward' if metapath src is under type1 and tgt is under type2.
        'reversed' if metapath src is under type2 and tgt is under type1.
        None if neither orientation matches.
    """
    nodes, _, _ = parse_metapath(metapath)
    src_ancestors = get_type_ancestors(nodes[0])
    tgt_ancestors = get_type_ancestors(nodes[-1])

    if type1 in src_ancestors and type2 in tgt_ancestors:
        return 'forward'
    if type2 in src_ancestors and type1 in tgt_ancestors:
        return 'reversed'
    return None


def _build_prefix_string(nhop_path, n_prefix_hops):
    """Extract the prefix string (first n_prefix_hops hops) from a metapath."""
    parts = nhop_path.split('|')
    # Each hop is 3 parts (pred, dir, type), prefix has n_prefix_hops hops
    # plus the starting type. Total parts = 1 + 3 * n_prefix_hops
    prefix_end = 1 + 3 * n_prefix_hops
    return '|'.join(parts[:prefix_end])


def _get_final_hop_key(nhop_path, n_hops):
    """Extract the final hop's (src_type, predicate, tgt_type, direction) from a metapath."""
    nodes, predicates, directions = parse_metapath(nhop_path)
    i = n_hops - 1
    return nodes[i], predicates[i], nodes[i + 1], directions[i]


def compute_direct_metrics(onehop_to_overlaps, target_pair_sets, exact_target_pair_counts,
                           base_matrices, manifest_matrices,
                           type1, type2,
                           start_time, progress_file, counters, stage_timings):
    """Directly evaluate each (explicit predictor path, target) pair via matrix operations.

    Replaces the old candidate-build + Phase B pipeline. Groups N-hop predictor
    paths by their (N-1)-hop prefix to share prefix matrix reconstruction, then
    computes exact predictor_count and overlap for each path against each target.

    Args:
        onehop_to_overlaps: {target_variant: {nhop_path: overlap_sum}}
        target_pair_sets: {target_variant: GraphBLAS matrix in (type1, type2) unified space}
        exact_target_pair_counts: {target_variant: int}
        base_matrices: {(src_type, pred, tgt_type): GraphBLAS matrix}
        manifest_matrices: List of matrix metadata dicts
        type1, type2: Type pair
        start_time: Start time for profiling
        progress_file: Progress file path
        counters: Counters dict
        stage_timings: Stage timings dict

    Returns:
        {target_variant: [(nhop_path, predictor_count, overlap)]}
    """
    t0 = time.time()

    # Step 1: Transpose onehop_to_overlaps to get path_to_targets
    path_to_targets = defaultdict(dict)
    for target_variant, nhop_overlaps in onehop_to_overlaps.items():
        for nhop_path, overlap_sum in nhop_overlaps.items():
            path_to_targets[nhop_path][target_variant] = overlap_sum

    counters["direct_unique_predictor_paths"] = len(path_to_targets)
    counters["direct_unique_targets"] = len(onehop_to_overlaps)

    # Step 2: Group paths by prefix (N-1 hops for N-hop paths, or identity for 1-hop)
    prefix_groups = defaultdict(list)
    for nhop_path in path_to_targets:
        nodes, predicates, _ = parse_metapath(nhop_path)
        n_hops = len(predicates)
        if n_hops <= 1:
            # 1-hop path: no prefix reconstruction needed
            prefix_groups["__1hop__"].append(nhop_path)
        else:
            n_prefix_hops = n_hops - 1
            prefix_key = _build_prefix_string(nhop_path, n_prefix_hops)
            prefix_groups[prefix_key].append(nhop_path)

    counters["direct_prefix_groups"] = len(prefix_groups)
    print(f"  Direct evaluation: {len(path_to_targets)} paths in {len(prefix_groups)} prefix groups")

    # Step 3: Build unified offsets (same as Phase A)
    type1_offsets, type1_total = build_unified_type_offsets(type1, manifest_matrices)
    type2_offsets, type2_total = build_unified_type_offsets(type2, manifest_matrices)

    # Step 4: Process each prefix group
    results = defaultdict(list)  # target_variant -> [(nhop_path, predictor_count, overlap)]
    total_paths_processed = 0
    total_reconstructions = 0
    total_reconstruction_failures = 0
    next_progress_at = time.time() + 30.0

    for prefix_idx, (prefix_key, paths_in_group) in enumerate(sorted(prefix_groups.items()), start=1):
        if prefix_key == "__1hop__":
            # 1-hop paths: reconstruct directly (just a base matrix lookup)
            for nhop_path in paths_in_group:
                total_reconstructions += 1
                nhop_matrix = reconstruct_nhop_matrix(nhop_path, base_matrices)
                if nhop_matrix is None:
                    total_reconstruction_failures += 1
                    total_paths_processed += 1
                    continue

                orientation = _detect_metapath_orientation(nhop_path, type1, type2)
                if orientation == 'forward':
                    remapped = remap_nhop_to_unified(
                        nhop_matrix, nhop_path,
                        type1_offsets, type2_offsets,
                        type1_total, type2_total,
                    )
                elif orientation == 'reversed':
                    remapped = remap_nhop_to_unified(
                        nhop_matrix, nhop_path,
                        type2_offsets, type1_offsets,
                        type2_total, type1_total,
                    )
                    if remapped is not None and remapped.nvals > 0:
                        remapped = remapped.T.new()
                else:
                    remapped = None

                if remapped is None or remapped.nvals == 0:
                    total_paths_processed += 1
                    continue

                predictor_count = remapped.nvals
                for target_variant in path_to_targets[nhop_path]:
                    target_ps = target_pair_sets.get(target_variant)
                    if target_ps is None:
                        continue
                    intersection = remapped.ewise_mult(target_ps, gb.binary.pair).new()
                    exact_overlap = intersection.nvals
                    if exact_overlap > 0:
                        results[target_variant].append((nhop_path, predictor_count, exact_overlap))

                total_paths_processed += 1
        else:
            # Multi-hop paths: reconstruct prefix once, then multiply by each final hop
            sample_path = paths_in_group[0]
            nodes, predicates, _ = parse_metapath(sample_path)
            n_hops = len(predicates)
            n_prefix_hops = n_hops - 1

            total_reconstructions += 1
            prefix_matrix = reconstruct_prefix_matrix(sample_path, n_prefix_hops, base_matrices)
            if prefix_matrix is None:
                total_reconstruction_failures += 1
                total_paths_processed += len(paths_in_group)
                continue

            for nhop_path in paths_in_group:
                total_reconstructions += 1
                # Get final hop matrix
                final_src, final_pred, final_tgt, final_dir = _get_final_hop_key(nhop_path, n_hops)

                key = (final_src, final_pred, final_tgt)
                if key in base_matrices:
                    final_hop = base_matrices[key]
                else:
                    reverse_key = (final_tgt, final_pred, final_src)
                    if reverse_key in base_matrices:
                        final_hop = base_matrices[reverse_key].T
                    else:
                        total_reconstruction_failures += 1
                        total_paths_processed += 1
                        continue

                # Full N-hop matrix = prefix × final_hop
                full_matrix = prefix_matrix.mxm(final_hop, gb.semiring.any_pair).new()

                if full_matrix.nvals == 0:
                    total_paths_processed += 1
                    continue

                # Remap to unified coords
                orientation = _detect_metapath_orientation(nhop_path, type1, type2)
                if orientation == 'forward':
                    remapped = remap_nhop_to_unified(
                        full_matrix, nhop_path,
                        type1_offsets, type2_offsets,
                        type1_total, type2_total,
                    )
                elif orientation == 'reversed':
                    remapped = remap_nhop_to_unified(
                        full_matrix, nhop_path,
                        type2_offsets, type1_offsets,
                        type2_total, type1_total,
                    )
                    if remapped is not None and remapped.nvals > 0:
                        remapped = remapped.T.new()
                else:
                    remapped = None

                if remapped is None or remapped.nvals == 0:
                    total_paths_processed += 1
                    continue

                predictor_count = remapped.nvals

                for target_variant in path_to_targets[nhop_path]:
                    target_ps = target_pair_sets.get(target_variant)
                    if target_ps is None:
                        continue
                    intersection = remapped.ewise_mult(target_ps, gb.binary.pair).new()
                    exact_overlap = intersection.nvals
                    if exact_overlap > 0:
                        results[target_variant].append((nhop_path, predictor_count, exact_overlap))

                total_paths_processed += 1

            # Prefix matrix goes out of scope here — GC can reclaim it

        now = time.time()
        if now >= next_progress_at or prefix_idx == len(prefix_groups):
            next_progress_at = now + 30.0
            print_profile_event(
                "compute_direct_metrics",
                start_time,
                prefix_groups_done=prefix_idx,
                prefix_groups_total=len(prefix_groups),
                paths_processed=total_paths_processed,
                paths_total=len(path_to_targets),
                reconstructions=total_reconstructions,
                reconstruction_failures=total_reconstruction_failures,
            )
            write_progress_file(
                progress_file,
                build_progress_payload(
                    type1, type2, "compute_direct_metrics",
                    start_time, counters, stage_timings,
                    prefix_groups_done=prefix_idx,
                    prefix_groups_total=len(prefix_groups),
                    paths_processed=total_paths_processed,
                    paths_total=len(path_to_targets),
                ),
            )

    stage_timings["compute_direct_metrics"] = time.time() - t0
    counters["direct_reconstructions"] = total_reconstructions
    counters["direct_reconstruction_failures"] = total_reconstruction_failures
    counters["direct_paths_processed"] = total_paths_processed
    counters["direct_result_rows"] = sum(len(v) for v in results.values())

    return dict(results)


def compute_exact_target_pair_counts(target_variant_counts, base_matrices, manifest_matrices,
                                     type1, type2, start_time, progress_file, counters, stage_timings):
    """Compute exact target pair counts using unified index pair-set union.

    For each target variant, unions all matching base matrices in unified
    coordinate space and returns the exact .nvals as the pair count.

    All returned pair sets are in standard (type1, type2) orientation:
    rows correspond to type1 unified nodes, cols to type2 unified nodes.
    Variants whose canonical form reverses the endpoints are transposed
    after construction so they share the same coordinate space.

    Args:
        target_variant_counts: {variant: summed_count} from build_target_variant_counts
        base_matrices: {(src_type, pred, tgt_type): GraphBLAS matrix}
        manifest_matrices: List of matrix metadata dicts from manifest.json
        type1, type2: Type pair
        start_time: Start time for profiling
        progress_file: Progress file path
        counters: Counters dict for profiling
        stage_timings: Stage timings dict

    Returns:
        {variant: exact_pair_count} dict, and the target pair set cache
        {variant: GraphBLAS matrix in (type1, type2) unified space}.
    """
    t0 = time.time()
    # Build unified offsets for both endpoint types
    type1_offsets, type1_total = build_unified_type_offsets(type1, manifest_matrices)
    type2_offsets, type2_total = build_unified_type_offsets(type2, manifest_matrices)
    counters["unified_src_total"] = type1_total
    counters["unified_tgt_total"] = type2_total

    exact_counts = {}
    target_pair_sets = {}
    total_variants = len(target_variant_counts)

    for idx, variant in enumerate(target_variant_counts, start=1):
        orientation = _detect_metapath_orientation(variant, type1, type2)

        if orientation == 'forward':
            # Variant src matches type1, tgt matches type2
            pair_set = build_target_pair_set(
                variant, base_matrices, manifest_matrices,
                type1_offsets, type2_offsets,
                type1_total, type2_total,
            )
        elif orientation == 'reversed':
            # Variant src matches type2, tgt matches type1 — build in reversed
            # space then transpose to standard (type1, type2) orientation
            pair_set = build_target_pair_set(
                variant, base_matrices, manifest_matrices,
                type2_offsets, type1_offsets,
                type2_total, type1_total,
            )
            if pair_set.nvals > 0:
                pair_set = pair_set.T.new()
            else:
                pair_set = gb.Matrix(gb.dtypes.BOOL, nrows=type1_total, ncols=type2_total)
        else:
            pair_set = gb.Matrix(gb.dtypes.BOOL, nrows=type1_total, ncols=type2_total)

        exact_count = pair_set.nvals
        exact_counts[variant] = exact_count
        target_pair_sets[variant] = pair_set

        if idx % 100 == 0 or idx == total_variants:
            summed = target_variant_counts[variant]
            print_profile_event(
                "compute_exact_target_counts",
                start_time,
                done=idx,
                total=total_variants,
                latest_exact=exact_count,
                latest_summed=summed,
            )
            write_progress_file(
                progress_file,
                build_progress_payload(
                    type1, type2, "compute_exact_target_counts",
                    start_time, counters, stage_timings,
                    done=idx, total=total_variants,
                ),
            )

    stage_timings["compute_exact_target_counts"] = time.time() - t0
    counters["exact_target_variants_computed"] = len(exact_counts)

    # Log the ratio of exact vs summed counts
    for variant in list(target_variant_counts)[:5]:
        summed = target_variant_counts[variant]
        exact = exact_counts.get(variant, 0)
        ratio = exact / summed if summed > 0 else 0.0
        print(f"  Target {variant}: summed={summed:,} exact={exact:,} ratio={ratio:.3f}")

    return exact_counts, target_pair_sets


def group_type_pair(type1, type2, file_list, output_dir, n_hops, explicit_items,
                    type_node_counts=None, min_count=0, min_precision=0.0,
                    excluded_types=None, excluded_predicates=None,
                    progress_file=None, matrices_dir=None):
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

    # Phase A: compute exact target pair counts using unified index
    base_matrices = None
    manifest_matrices = None
    exact_target_pair_counts = None
    target_pair_sets = None
    if matrices_dir:
        print(f"\n[Phase A] Loading base matrices from {matrices_dir}...")
        t_load = time.time()
        manifest_matrices, base_matrices = load_base_matrices(matrices_dir)
        stage_timings["load_base_matrices"] = time.time() - t_load
        counters["base_matrices_loaded"] = len(base_matrices)
        print(f"  Loaded {len(base_matrices)} base matrices in {time.time() - t_load:.1f}s")

        print(f"[Phase A] Computing exact target pair counts...")
        exact_target_pair_counts, target_pair_sets = compute_exact_target_pair_counts(
            target_variant_counts, base_matrices, manifest_matrices,
            type1, type2, start_time, progress_file, counters, stage_timings,
        )
        print(f"  Exact target pair counts computed for {len(exact_target_pair_counts)} variants")

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

    # Direct matrix evaluation: compute exact metrics for each (predictor path, target) pair
    if base_matrices is not None and target_pair_sets is not None:
        print(f"\n[Direct Evaluation] Computing exact metrics via matrix operations...")
        direct_results = compute_direct_metrics(
            onehop_to_overlaps, target_pair_sets, exact_target_pair_counts,
            base_matrices, manifest_matrices,
            type1, type2,
            start_time, progress_file, counters, stage_timings,
        )
        print(f"  Direct evaluation complete: {counters.get('direct_result_rows', 0)} result rows "
              f"across {len(direct_results)} targets")
    else:
        direct_results = {}

    # Use exact target counts for output metrics when available
    output_target_counts = exact_target_pair_counts if exact_target_pair_counts else target_variant_counts

    counters["targets_processed"] = 0
    counters["output_files_written"] = 0
    counters["output_rows_written"] = 0

    total_targets = len(onehop_to_overlaps)
    for target_index, (onehop_path, nhop_overlaps) in enumerate(onehop_to_overlaps.items(), start=1):
        t_target = time.time()

        onehop_count_global = output_target_counts.get(onehop_path, 0)
        target_rows = direct_results.get(onehop_path, [])

        print(f"\nProcessing 1-hop: {onehop_path}")
        print(f"  {len(nhop_overlaps)} explicit N-hop paths with overlap, {len(target_rows)} exact results")

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

            # Sort by overlap descending
            sorted_rows = sorted(target_rows, key=lambda x: x[2], reverse=True)
            for nhop_path, predictor_count, overlap in sorted_rows:
                if should_exclude_metapath(nhop_path, excluded_types, excluded_predicates):
                    rows_filtered_excluded += 1
                    continue
                if contains_pseudo_type(nhop_path):
                    rows_filtered_pseudo += 1
                    continue

                if predictor_count <= 0 or predictor_count < min_count:
                    rows_filtered_count += 1
                    continue

                metrics = calculate_metrics(predictor_count, onehop_count_global, overlap, total_possible_for_pair)
                if metrics['precision'] < min_precision:
                    rows_filtered_precision += 1
                    continue

                out.write(f"{nhop_path}\t{predictor_count}\t{overlap}\t{total_possible_for_pair}\t")
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
            explicit_predictors=len(nhop_overlaps),
            exact_result_rows=len(target_rows),
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
                exact_result_rows=len(target_rows),
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
    parser.add_argument('--matrices-dir', type=str, default=None, help='Matrices directory for exact pair-set tracking')

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
        matrices_dir=args.matrices_dir,
    )


if __name__ == "__main__":
    main()
