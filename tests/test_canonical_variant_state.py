#!/usr/bin/env python3

from library.aggregation import (
    canonical_variant_ancestor_closure_state_ids_for_typepair,
    canonical_variant_children_for_typepair,
    canonical_variant_metapath,
    canonical_variant_metapath_from_state_ids,
    canonical_variant_state_ids,
    traverse_canonical_variants_for_typepair_pruned,
    traverse_metapath_variants_for_typepair_pruned,
)


def test_canonical_variant_collapses_reverse_equivalent_nonsymmetric_path():
    left = canonical_variant_metapath("Disease|affects|R|Gene")
    right = canonical_variant_metapath("Gene|affects|F|Disease")

    assert left == right == "Disease|affects|R|Gene"


def test_canonical_variant_collapses_symmetric_direction_and_reverse():
    left = canonical_variant_metapath("Gene|interacts_with|F|Protein")
    right = canonical_variant_metapath("Protein|interacts_with|R|Gene")

    assert left == right == "Gene|interacts_with|A|Protein"


def test_canonical_variant_collapses_same_type_symmetric_directions():
    forward = canonical_variant_metapath("Gene|interacts_with|F|Gene")
    reverse = canonical_variant_metapath("Gene|interacts_with|R|Gene")

    assert forward == reverse == "Gene|interacts_with|A|Gene"


def test_canonical_variant_children_keep_exact_worker_endpoints():
    children = canonical_variant_children_for_typepair(
        "ChemicalEntity|affects|F|Gene|treats|F|Disease",
        type1="ChemicalEntity",
        type2="Disease",
    )

    assert children
    assert all(child.startswith("ChemicalEntity|") for child in children)
    assert all(child.endswith("|Disease") for child in children)
    assert any(
        child.startswith("ChemicalEntity|affects|F|")
        and child.endswith("|treats|F|Disease")
        and child != "ChemicalEntity|affects|F|Gene|treats|F|Disease"
        for child in children
    )


def test_canonical_variant_children_are_unique_under_canonicalization():
    children = canonical_variant_children_for_typepair(
        "Gene|interacts_with|A|Protein",
        type1="Gene",
        type2="Protein",
    )

    assert len(children) == len(set(children))
    assert "Gene|related_to_at_instance_level|A|Protein" in children


def test_canonical_variant_state_ids_round_trip_to_metapath():
    metapath = "Protein|interacts_with|R|Gene"
    state_ids = canonical_variant_state_ids(metapath)

    assert canonical_variant_metapath_from_state_ids(state_ids) == "Gene|interacts_with|A|Protein"


def test_canonical_integer_traversal_matches_bounded_traversal_variants():
    expected = []
    actual = []

    def visit_expected(variant):
        expected.append(variant)
        return False

    def visit_actual(state_ids):
        actual.append(canonical_variant_metapath_from_state_ids(state_ids))
        return False

    traverse_metapath_variants_for_typepair_pruned(
        "SmallMolecule|affects|F|Gene|treats|F|Disease",
        type1="ChemicalEntity",
        type2="Disease",
        visit_variant=visit_expected,
    )
    traverse_canonical_variants_for_typepair_pruned(
        "SmallMolecule|affects|F|Gene|treats|F|Disease",
        type1="ChemicalEntity",
        type2="Disease",
        visit_variant=visit_actual,
    )

    assert set(actual) == set(expected)


def test_canonical_integer_traversal_prunes_branches():
    visited = []

    def visit_variant(state_ids):
        variant = canonical_variant_metapath_from_state_ids(state_ids)
        visited.append(variant)
        return variant == "ChemicalEntity|treats|F|Disease"

    traverse_canonical_variants_for_typepair_pruned(
        "SmallMolecule|treats|F|Disease",
        type1="ChemicalEntity",
        type2="Disease",
        visit_variant=visit_variant,
    )

    assert "ChemicalEntity|treats|F|Disease" in visited
    assert "NamedThing|treats|F|Disease" not in visited


def test_canonical_variant_ancestor_closure_contains_self_and_broader_variants():
    state_ids = canonical_variant_state_ids("ChemicalEntity|affects|F|Gene|treats|F|Disease")
    closure = {
        canonical_variant_metapath_from_state_ids(closure_state_ids)
        for closure_state_ids in canonical_variant_ancestor_closure_state_ids_for_typepair(
            state_ids,
            type1="ChemicalEntity",
            type2="Disease",
        )
    }

    assert "ChemicalEntity|affects|F|Gene|treats|F|Disease" in closure
    assert "ChemicalEntity|related_to_at_instance_level|A|Gene|treats|F|Disease" in closure
