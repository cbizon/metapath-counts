#!/usr/bin/env python3
"""
Trace candidate-building pruning across multiple real predictors.

This script replays the per-target lower-bound pruning logic for the top-N
explicit predictors from a type-pair shard. It carries shared target-local
state across predictors, just like the grouping worker, and writes every
visited variant plus the prune decision math to a text file.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from library.aggregation import (
    _build_typepair_variant_dimensions,
    _state_signature_for_dimension_indexes,
    _variants_for_dimension_indexes,
)


CHEM_DIS_TARGET_COUNTS = {
    "ChemicalEntity|related_to|A|DiseaseOrPhenotypicFeature": 9145079,
    "ChemicalEntity|related_to_at_instance_level|A|DiseaseOrPhenotypicFeature": 9095655,
    "ChemicalEntity|associated_with|A|DiseaseOrPhenotypicFeature": 6711945,
}


def load_top_predictors(shard_path: Path, limit: int) -> list[tuple[str, int]]:
    with shard_path.open("rb") as f:
        items = pickle.load(f)

    predictors: list[tuple[str, int]] = []
    for path, count in items:
        if path.count("|") == 9:
            predictors.append((path, int(count)))

    predictors.sort(key=lambda x: x[1], reverse=True)
    return predictors[:limit]


def trace_variants_for_predictor(
    path: str,
    type1: str,
    type2: str,
    visit_state,
    visit_variant,
    out,
    stop_after_visits: int | None = None,
) -> None:
    nodes, predicates, directions, assignment_dimensions, node_dim_count, symmetric_preds, required_endpoints = (
        _build_typepair_variant_dimensions(path, type1, type2)
    )
    yielded_variants = set()
    visits_emitted = 0

    def describe_state(dimensions, indexes: tuple[int, ...]) -> str:
        node_choices = []
        for i in range(node_dim_count):
            node_choices.append(f"node{i}={dimensions[i][indexes[i]]}")
        pred_choices = []
        for i in range(len(predicates)):
            pred_choices.append(f"pred{i}={dimensions[node_dim_count + i][indexes[node_dim_count + i]]}")
        return " ".join(node_choices + pred_choices)

    for assignment_index, dimensions in enumerate(assignment_dimensions, start=1):
        out.write(f"ASSIGNMENT {assignment_index}\n")
        start = tuple(0 for _ in dimensions)
        stack = [start]
        seen_indexes = {start}
        parents: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}

        while stack:
            indexes = stack.pop()
            state_signature = _state_signature_for_dimension_indexes(
                nodes,
                predicates,
                directions,
                dimensions,
                node_dim_count,
                symmetric_preds,
                indexes,
            )
            variants = _variants_for_dimension_indexes(
                nodes,
                predicates,
                directions,
                dimensions,
                node_dim_count,
                symmetric_preds,
                required_endpoints,
                indexes,
            )
            out.write(
                f"POP state={indexes} parent={parents[indexes]} "
                f"{describe_state(dimensions, indexes)} raw_variants={len(variants)}\n"
            )
            if state_signature is not None:
                if visit_state(state_signature):
                    out.write("  STATE_BRANCH_STOP\n")
                    continue

            branch_pruned = False
            emitted_new_variant = False
            for variant in variants:
                if variant in yielded_variants:
                    out.write(f"  DUP variant={variant}\n")
                    continue
                yielded_variants.add(variant)
                emitted_new_variant = True
                out.write(
                    f"state indexes={indexes} parent={parents[indexes]} variant={variant} "
                )
                if visit_variant(variant):
                    branch_pruned = True
                    out.write("branch_stop=yes\n")
                else:
                    out.write("branch_stop=no\n")
                visits_emitted += 1
                if stop_after_visits is not None and visits_emitted >= stop_after_visits:
                    return

            if not variants:
                out.write("  NO_VARIANT\n")
            elif not emitted_new_variant:
                out.write("  ONLY_DUPLICATES\n")

            if branch_pruned:
                continue

            for dim_idx in range(len(dimensions) - 1, -1, -1):
                next_idx = indexes[dim_idx] + 1
                if next_idx >= len(dimensions[dim_idx]):
                    continue
                child = list(indexes)
                child[dim_idx] = next_idx
                child = tuple(child)
                if child in seen_indexes:
                    continue
                seen_indexes.add(child)
                parents[child] = indexes
                stack.append(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        default="results_3hop/_tmp_prepare_grouping/typepair_explicit_paths/ChemicalEntity__DiseaseOrPhenotypicFeature.pkl",
    )
    parser.add_argument(
        "--target",
        choices=sorted(CHEM_DIS_TARGET_COUNTS),
        default="ChemicalEntity|related_to|A|DiseaseOrPhenotypicFeature",
    )
    parser.add_argument("--min-precision", type=float, default=0.99)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--stop-after-visits", type=int, default=None)
    parser.add_argument(
        "--output",
        default="logs_grouping_3hop/trace_predictor_pruning_related_to_p099.txt",
    )
    args = parser.parse_args()

    shard_path = Path(args.shard)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_count = CHEM_DIS_TARGET_COUNTS[args.target]
    max_predictor_count = None if args.min_precision <= 0 else target_count / args.min_precision
    predictors = load_top_predictors(shard_path, args.limit)

    lower_bound_counts: dict[str, int] = {}
    lower_bound_state_counts = {}
    aggregated_overlaps: dict[str, int] = {}
    pruned_variants: set[str] = set()
    pruned_states = set()
    state_token_ids: dict[str, int] = {}
    next_state_token_id = 0

    total_visited = 0
    total_pruned = 0
    total_revisit_pruned = 0
    total_accepted = 0

    with output_path.open("w") as out:
        out.write(f"target={args.target}\n")
        out.write(f"target_count={target_count}\n")
        out.write(f"min_precision={args.min_precision}\n")
        out.write(f"max_predictor_count={max_predictor_count}\n")
        out.write(f"predictor_limit={args.limit}\n")
        out.write("\n")

        for predictor_index, (path, explicit_count) in enumerate(predictors, start=1):
            predictor_visited = 0
            predictor_pruned = 0
            predictor_revisit_pruned = 0
            predictor_accepted = 0

            out.write(f"=== predictor {predictor_index} ===\n")
            out.write(f"path={path}\n")
            out.write(f"explicit_predictor_count={explicit_count}\n")
            out.write(f"pruned_variants_before={len(pruned_variants)}\n")
            out.write(f"pruned_states_before={len(pruned_states)}\n")
            out.write(f"active_variants_before={len(aggregated_overlaps)}\n")

            def visit_state(state_signature: tuple[str, ...]) -> bool:
                nonlocal next_state_token_id
                state_id_parts = []
                for token in state_signature:
                    token_id = state_token_ids.get(token)
                    if token_id is None:
                        token_id = next_state_token_id
                        state_token_ids[token] = token_id
                        next_state_token_id += 1
                    state_id_parts.append(token_id)
                state_id = tuple(state_id_parts)
                state_label = "|".join(state_signature)
                proposed = lower_bound_state_counts.get(state_id, 0) + explicit_count
                if state_id in pruned_states:
                    out.write(
                        f"  STATE state_key={state_label} state_id={state_id} previous={lower_bound_state_counts.get(state_id, 0)} "
                        f"proposed={proposed} threshold={max_predictor_count} action=revisit_pruned\n"
                    )
                    return True
                if max_predictor_count is not None and proposed > max_predictor_count:
                    pruned_states.add(state_id)
                    lower_bound_state_counts.pop(state_id, None)
                    out.write(
                        f"  STATE state_key={state_label} state_id={state_id} previous=0 proposed={proposed} "
                        f"threshold={max_predictor_count} action=prune\n"
                    )
                    return True
                lower_bound_state_counts[state_id] = proposed
                out.write(
                    f"  STATE state_key={state_label} state_id={state_id} previous=0 proposed={proposed} "
                    f"threshold={max_predictor_count} action=accept\n"
                )
                return False

            def visit_variant(variant: str) -> bool:
                nonlocal total_visited, total_pruned, total_revisit_pruned, total_accepted
                nonlocal predictor_visited, predictor_pruned, predictor_revisit_pruned, predictor_accepted

                predictor_visited += 1
                total_visited += 1

                previous = lower_bound_counts.get(variant, 0)
                proposed = previous + explicit_count

                if variant in pruned_variants:
                    predictor_revisit_pruned += 1
                    total_revisit_pruned += 1
                    out.write(
                        f"visit {predictor_visited}: variant={variant} "
                        f"previous={previous} proposed={proposed} threshold={max_predictor_count} "
                        f"action=revisit_pruned\n"
                    )
                    return True

                if max_predictor_count is not None and proposed > max_predictor_count:
                    pruned_variants.add(variant)
                    lower_bound_counts.pop(variant, None)
                    aggregated_overlaps.pop(variant, None)
                    predictor_pruned += 1
                    total_pruned += 1
                    out.write(
                        f"visit {predictor_visited}: variant={variant} "
                        f"previous={previous} proposed={proposed} threshold={max_predictor_count} "
                        f"action=prune\n"
                    )
                    return True

                lower_bound_counts[variant] = proposed
                aggregated_overlaps[variant] = aggregated_overlaps.get(variant, 0) + 1
                predictor_accepted += 1
                total_accepted += 1
                out.write(
                    f"visit {predictor_visited}: variant={variant} "
                    f"previous={previous} proposed={proposed} threshold={max_predictor_count} "
                    f"action=accept\n"
                )
                return False

            trace_variants_for_predictor(
                path,
                "ChemicalEntity",
                "DiseaseOrPhenotypicFeature",
                visit_state,
                visit_variant,
                out,
                stop_after_visits=args.stop_after_visits,
            )

            out.write(
                f"summary: visited={predictor_visited} accepted={predictor_accepted} "
                f"pruned={predictor_pruned} revisit_pruned={predictor_revisit_pruned} "
                f"pruned_variants_after={len(pruned_variants)} active_variants_after={len(aggregated_overlaps)}\n"
            )
            out.write("\n")

        out.write("=== total summary ===\n")
        out.write(
            f"visited={total_visited} accepted={total_accepted} "
            f"pruned={total_pruned} revisit_pruned={total_revisit_pruned} "
            f"unique_pruned_variants={len(pruned_variants)} active_variants={len(aggregated_overlaps)}\n"
        )

    print(output_path)


if __name__ == "__main__":
    main()
