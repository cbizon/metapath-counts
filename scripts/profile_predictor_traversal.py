#!/usr/bin/env python3
"""
Profile predictor-variant traversal on real 3-hop predictors.

This script profiles the branch-pruned traversal used during candidate-building
without running the full grouping worker. It uses real explicit predictors from
the ChemicalEntity/DiseaseOrPhenotypicFeature shard and a callback equivalent to
the candidate-building lower-bound prune logic.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from dataclasses import dataclass

from library.aggregation import traverse_metapath_variants_for_typepair_pruned


CHEM_DIS_TARGET_COUNTS = {
    "ChemicalEntity|related_to|A|DiseaseOrPhenotypicFeature": 9145079,
    "ChemicalEntity|related_to_at_instance_level|A|DiseaseOrPhenotypicFeature": 9095655,
    "ChemicalEntity|associated_with|A|DiseaseOrPhenotypicFeature": 6711945,
}


REAL_PREDICTORS = {
    "top1": (
        "Disease|contributes_to|R|Gene+Protein|affects_decreased_activity_or_abundance|R|"
        "Gene+Protein|directly_physically_interacts_with|A|SmallMolecule",
        1345540431,
    ),
    "top2": (
        "Disease|contributes_to|R|Gene+Protein|affects_increased_activity_or_abundance|R|"
        "Gene+Protein|directly_physically_interacts_with|A|SmallMolecule",
        1334563656,
    ),
    "top20": (
        "Disease|contributes_to|R|Drug+SmallMolecule|affects|F|"
        "Gene+Protein|directly_physically_interacts_with|A|SmallMolecule",
        406607235,
    ),
}


@dataclass
class TraversalStats:
    visited_variants: int = 0
    accepted_variants: int = 0
    pruned_variants: int = 0
    revisits_pruned: int = 0
    active_variants: int = 0


def profile_predictor(path: str, explicit_predictor_count: int, target_count: int, min_precision: float):
    max_predictor_count = None if min_precision <= 0 else target_count / min_precision
    lower_bound_counts = {}
    lower_bound_state_counts = {}
    aggregated_overlaps = {}
    pruned_variants = set()
    pruned_states = set()
    state_token_ids = {}
    next_state_token_id = 0
    stats = TraversalStats()

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

        proposed = lower_bound_state_counts.get(state_id, 0) + explicit_predictor_count
        if state_id in pruned_states:
            stats.revisits_pruned += 1
            return True
        if max_predictor_count is not None and proposed > max_predictor_count:
            pruned_states.add(state_id)
            lower_bound_state_counts.pop(state_id, None)
            stats.pruned_variants += 1
            return True
        lower_bound_state_counts[state_id] = proposed
        return False

    def visit_variant(variant: str) -> bool:
        stats.visited_variants += 1
        if variant in pruned_variants:
            stats.revisits_pruned += 1
            return True

        proposed = lower_bound_counts.get(variant, 0) + explicit_predictor_count
        if max_predictor_count is not None and proposed > max_predictor_count:
            pruned_variants.add(variant)
            lower_bound_counts.pop(variant, None)
            aggregated_overlaps.pop(variant, None)
            stats.pruned_variants += 1
            return True

        lower_bound_counts[variant] = proposed
        aggregated_overlaps[variant] = aggregated_overlaps.get(variant, 0) + 1
        stats.accepted_variants += 1
        stats.active_variants = len(aggregated_overlaps)
        return False

    profiler = cProfile.Profile()
    profiler.enable()
    traverse_metapath_variants_for_typepair_pruned(
        path,
        "ChemicalEntity",
        "DiseaseOrPhenotypicFeature",
        visit_variant,
        visit_state=visit_state,
    )
    profiler.disable()

    buffer = io.StringIO()
    stats_obj = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats_obj.print_stats(30)
    return stats, buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=sorted(REAL_PREDICTORS), default="top1")
    parser.add_argument(
        "--target",
        choices=sorted(CHEM_DIS_TARGET_COUNTS),
        default="ChemicalEntity|related_to|A|DiseaseOrPhenotypicFeature",
    )
    parser.add_argument("--min-precision", type=float, default=0.9)
    args = parser.parse_args()

    path, explicit_predictor_count = REAL_PREDICTORS[args.predictor]
    target_count = CHEM_DIS_TARGET_COUNTS[args.target]
    max_predictor_count = None if args.min_precision <= 0 else target_count / args.min_precision

    print("predictor_key:", args.predictor)
    print("predictor_count:", explicit_predictor_count)
    print("predictor_path:", path)
    print("target:", args.target)
    print("target_count:", target_count)
    print("min_precision:", args.min_precision)
    print("max_predictor_count:", max_predictor_count)

    stats, profile_text = profile_predictor(path, explicit_predictor_count, target_count, args.min_precision)
    print()
    print("traversal_stats:")
    print("  visited_variants:", stats.visited_variants)
    print("  accepted_variants:", stats.accepted_variants)
    print("  pruned_variants:", stats.pruned_variants)
    print("  revisits_pruned:", stats.revisits_pruned)
    print("  active_variants:", stats.active_variants)
    print()
    print("cProfile_top30:")
    print(profile_text)


if __name__ == "__main__":
    main()
