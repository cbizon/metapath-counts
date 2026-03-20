from library.aggregation import (
    canonical_variant_metapath_from_state_ids,
    expand_metapath_to_typepair_variants,
    parse_metapath,
    promote_metapath_endpoints_to_typepair_starts,
    promote_metapath_endpoints_to_typepair_rollup_keys,
    traverse_canonical_variants_for_typepair_pruned,
)


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


def test_canonical_start_traversal_mismatches_explicit_expansion():
    """Traversal from CANONICAL promoted starts over-generates variants.

    This documents the known property that canonical reversal + hierarchy
    expansion != expansion + canonical reversal.  The worker avoids this
    by using non-canonical rollup keys (see next test).
    """
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


def test_noncanonical_rollup_key_traversal_matches_explicit_expansion():
    """Traversal from non-canonical rollup keys matches explicit expansion exactly.

    Non-canonical rollup keys keep predicates in their original positions,
    so hierarchy expansion produces the same cross-product as direct expansion.
    """
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

    rollup_keys = set()
    for path in explicit_paths:
        rollup_keys.update(
            promote_metapath_endpoints_to_typepair_rollup_keys(path, type1, type2)
        )

    traversed_variants = set()
    for start, force_same_endpoint_reverse in rollup_keys:
        def visit(variant_state_ids):
            traversed_variants.add(canonical_variant_metapath_from_state_ids(variant_state_ids))
            return False

        traverse_canonical_variants_for_typepair_pruned(
            start, type1, type2, visit,
            force_same_endpoint_reverse=force_same_endpoint_reverse,
        )

    assert traversed_variants == explicit_variants
