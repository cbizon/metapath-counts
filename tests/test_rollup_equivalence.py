from library.aggregation import (
    canonical_variant_metapath_from_state_ids,
    expand_metapath_to_typepair_variants,
    parse_metapath,
    promote_metapath_endpoints_to_typepair_starts,
    promote_metapath_endpoints_to_typepair_rollup_keys,
    traverse_canonical_variants_for_typepair_pruned,
)
from pipeline.workers.run_grouping import (
    build_global_variant_predictor_counts,
    group_predictor_items_by_promoted_start,
)


def _explicit_variant_counts(explicit_items, n_hops, type1, type2):
    counts = {}
    for path, count in explicit_items:
        if len(parse_metapath(path)[1]) != n_hops:
            continue
        if not promote_metapath_endpoints_to_typepair_rollup_keys(path, type1, type2):
            continue
        for variant in expand_metapath_to_typepair_variants(path, type1, type2):
            counts[variant] = counts.get(variant, 0) + count
    return counts


def _rolled_variant_counts(explicit_items, n_hops, type1, type2):
    rolled = group_predictor_items_by_promoted_start(explicit_items, n_hops, type1, type2, counters={})
    counts_by_state = build_global_variant_predictor_counts(explicit_items, n_hops, type1, type2, counters={})
    return {
        canonical_variant_metapath_from_state_ids(state_ids): count
        for state_ids, count in counts_by_state.items()
    }


def test_promoted_start_rollup_is_lossy_when_distinct_endpoints_collapse_to_same_endpoint():
    explicit_items = [
        ("Gene+Protein|affects|F|Disease|affects|R|Gene", 2),
        ("Protein|affects|F|Disease|affects|R|Gene", 2),
        ("Protein|associated_with|A|Disease|affects|R|Gene", 2),
    ]

    explicit_counts = _explicit_variant_counts(explicit_items, 2, "GeneOrGeneProduct", "GeneOrGeneProduct")
    promoted_rolls = group_predictor_items_by_promoted_start(
        explicit_items,
        2,
        "GeneOrGeneProduct",
        "GeneOrGeneProduct",
        counters={},
    )

    assert len(promoted_rolls) == 3
    assert explicit_counts[
        "GeneOrGeneProduct|related_to|A|ThingWithTaxon|related_to_at_instance_level|A|GeneOrGeneProduct"
    ] == 6


def test_rollup_preserves_same_endpoint_expansion_counts():
    explicit_items = [
        ("Gene|regulates|F|Gene", 2),
        ("Gene|interacts_with|F|Gene", 3),
    ]

    explicit_counts = _explicit_variant_counts(explicit_items, 1, "GeneOrGeneProduct", "GeneOrGeneProduct")
    rolled_counts = _rolled_variant_counts(explicit_items, 1, "GeneOrGeneProduct", "GeneOrGeneProduct")

    assert rolled_counts == explicit_counts


def test_rollup_keys_preserve_original_endpoint_distinctness():
    distinct_keys = promote_metapath_endpoints_to_typepair_rollup_keys(
        "Protein|affects|F|Disease|affects|R|Gene",
        "GeneOrGeneProduct",
        "GeneOrGeneProduct",
    )
    same_keys = promote_metapath_endpoints_to_typepair_rollup_keys(
        "Gene|regulates|F|Gene",
        "GeneOrGeneProduct",
        "GeneOrGeneProduct",
    )

    assert distinct_keys == [
        ("GeneOrGeneProduct|affects|F|Disease|affects|R|GeneOrGeneProduct", True)
    ]
    assert same_keys == [
        ("GeneOrGeneProduct|regulates|F|GeneOrGeneProduct", False)
    ]


def test_count_rollup_groups_by_full_variant_signature_without_changing_counts():
    explicit_items = [
        ("Protein|affects|F|Disease|affects|R|Gene", 2),
        ("Gene+Protein|affects|F|Disease|affects|R|Gene", 2),
        ("Protein|associated_with|A|Disease|affects|R|Gene", 2),
        ("Gene|regulates|F|Gene", 1),
    ]

    explicit_counts = _explicit_variant_counts(explicit_items, 2, "GeneOrGeneProduct", "GeneOrGeneProduct")
    rolled_counts = _rolled_variant_counts(explicit_items, 2, "GeneOrGeneProduct", "GeneOrGeneProduct")

    assert rolled_counts == explicit_counts


def test_promoted_start_traversal_mismatches_explicit_ancestor_set_for_failing_case():
    explicit_paths = [
        "Gene+Protein|affects|F|Disease|affects|R|Gene",
        "Protein|affects|F|Disease|affects|R|Gene",
        "Protein|associated_with|A|Disease|affects|R|Gene",
    ]
    type1 = "GeneOrGeneProduct"
    type2 = "GeneOrGeneProduct"

    explicit_variants = set()
    for path in explicit_paths:
        explicit_variants.update(expand_metapath_to_typepair_variants(path, type1, type2))

    rolled_starts = set()
    for path in explicit_paths:
        rolled_starts.update(
            promote_metapath_endpoints_to_typepair_starts(path, type1, type2)
        )

    traversed_variants = set()
    for start in rolled_starts:
        def visit(variant_state_ids):
            traversed_variants.add(canonical_variant_metapath_from_state_ids(variant_state_ids))
            return False

        traverse_canonical_variants_for_typepair_pruned(start, type1, type2, visit)

    assert traversed_variants != explicit_variants
    assert (
        "GeneOrGeneProduct|related_to_at_instance_level|A|Disease|associated_with|A|GeneOrGeneProduct"
        in traversed_variants
    )
    assert (
        "GeneOrGeneProduct|related_to_at_instance_level|A|Disease|associated_with|A|GeneOrGeneProduct"
        not in explicit_variants
    )
