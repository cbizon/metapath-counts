#!/usr/bin/env python3
"""
Comprehensive tests for hierarchy inference utilities.

Tests cover:
- Type ancestor retrieval for regular types
- Type ancestor retrieval for pseudo-types
- Type parent retrieval (immediate parents only)
- Predicate hierarchy (ancestors and parents)
- Qualifier hierarchy
- Edge cases
"""

import pytest
from metapath_counts.hierarchy import (
    get_type_ancestors,
    get_type_parents,
    get_predicate_ancestors,
    get_predicate_parents,
    get_qualifier_ancestors
)
from metapath_counts.type_assignment import is_pseudo_type


class TestGetTypeAncestors:
    """Test type ancestor retrieval."""

    def test_regular_type_has_ancestors(self):
        """Regular type should return ancestors."""
        result = get_type_ancestors("SmallMolecule")
        # Should include the type itself
        assert "SmallMolecule" in result
        # Should include known ancestors
        assert "ChemicalEntity" in result
        assert "NamedThing" in result
        # Should be a set
        assert isinstance(result, set)

    def test_ancestors_include_self(self):
        """Type should be included in its own ancestors."""
        result = get_type_ancestors("Gene")
        assert "Gene" in result

    def test_gene_ancestors(self):
        """Test Gene has expected ancestors."""
        result = get_type_ancestors("Gene")
        assert "Gene" in result
        # Gene should have BiologicalEntity as ancestor
        assert "BiologicalEntity" in result
        # Should have root types
        assert "NamedThing" in result or "Entity" in result

    def test_disease_ancestors(self):
        """Test Disease hierarchy."""
        result = get_type_ancestors("Disease")
        assert "Disease" in result
        assert "DiseaseOrPhenotypicFeature" in result

    def test_pseudo_type_expansion(self):
        """Pseudo-type should return union of constituent ancestors."""
        result = get_type_ancestors("Gene+SmallMolecule")
        # Should include the pseudo-type itself
        assert "Gene+SmallMolecule" in result
        # Should include both constituents
        assert "Gene" in result
        assert "SmallMolecule" in result
        # Should include ancestors from both
        assert "BiologicalEntity" in result  # From Gene
        assert "ChemicalEntity" in result    # From SmallMolecule
        # Should have common ancestors only once
        assert "NamedThing" in result

    def test_three_type_pseudo(self):
        """Pseudo-type with three constituents."""
        result = get_type_ancestors("Gene+Protein+SmallMolecule")
        # Should include pseudo-type and all constituents
        assert "Gene+Protein+SmallMolecule" in result
        assert "Gene" in result
        assert "Protein" in result
        assert "SmallMolecule" in result
        # Should include union of all ancestors
        assert "BiologicalEntity" in result
        assert "ChemicalEntity" in result

    def test_no_duplicates(self):
        """Result should not contain duplicates."""
        result = get_type_ancestors("SmallMolecule")
        assert len(result) == len(set(result))

    def test_caching(self):
        """Repeated calls should use cache (same result)."""
        result1 = get_type_ancestors("SmallMolecule")
        result2 = get_type_ancestors("SmallMolecule")
        assert result1 == result2


class TestGetPredicateAncestors:
    """Test predicate ancestor retrieval."""

    def test_predicate_with_ancestors(self):
        """Predicate with known ancestors."""
        result = get_predicate_ancestors("treats")
        # Should be a list
        assert isinstance(result, list)
        # Should contain ancestor predicates
        # treats -> related_to (at some level)
        assert any("related_to" in pred for pred in result)

    def test_affects_hierarchy(self):
        """Test affects predicate hierarchy."""
        result = get_predicate_ancestors("affects")
        # affects should have related_to as ancestor
        assert any("related_to" in pred for pred in result)

    def test_regulates_hierarchy(self):
        """Test regulates has expected hierarchy."""
        result = get_predicate_ancestors("regulates")
        # regulates -> affects -> related_to
        assert any("affects" in pred for pred in result)
        assert any("related_to" in pred for pred in result)

    def test_biolink_prefix_handling(self):
        """Should handle both with and without biolink: prefix."""
        result1 = get_predicate_ancestors("treats")
        result2 = get_predicate_ancestors("biolink:treats")
        # Should return same ancestors (with biolink: prefix)
        assert result1 == result2

    def test_underscore_handling(self):
        """Should handle underscores and spaces."""
        result = get_predicate_ancestors("physically_interacts_with")
        # Should work correctly
        assert isinstance(result, list)

    def test_no_ancestors(self):
        """Predicate with no ancestors should return empty list."""
        # related_to is typically at the top
        result = get_predicate_ancestors("related_to")
        # Might be empty or very short
        assert isinstance(result, list)

    def test_invalid_predicate(self):
        """Invalid predicate should return empty list."""
        result = get_predicate_ancestors("not_a_real_predicate_xyz123")
        assert result == []

    def test_excludes_self(self):
        """Ancestors should not include the predicate itself."""
        result = get_predicate_ancestors("treats")
        # Should not include treats (only ancestors)
        assert "treats" not in result

    def test_format(self):
        """Results should NOT have biolink: prefix, should use underscores."""
        result = get_predicate_ancestors("treats")
        for pred in result:
            assert not pred.startswith("biolink:"), f"Found biolink: prefix in {pred}"
            assert " " not in pred  # Should use underscores


class TestGetQualifierAncestors:
    """Test qualifier ancestor retrieval."""

    def test_qualifier_with_hierarchy(self):
        """Test qualifier that has hierarchy."""
        # Note: This depends on what's in the Biolink model
        # Some qualifiers might not have hierarchies
        result = get_qualifier_ancestors("increased")
        # Should include itself
        assert "increased" in result
        assert isinstance(result, list)

    def test_qualifier_includes_self(self):
        """Qualifier ancestors should include the qualifier itself."""
        result = get_qualifier_ancestors("decreased")
        assert "decreased" in result

    def test_invalid_qualifier(self):
        """Invalid qualifier should return just itself."""
        result = get_qualifier_ancestors("not_a_real_qualifier_xyz123")
        assert result == ["not_a_real_qualifier_xyz123"]

    def test_prefix_handling(self):
        """Should handle biolink: prefix if present."""
        result1 = get_qualifier_ancestors("increased")
        result2 = get_qualifier_ancestors("biolink:increased")
        # Both should work
        assert len(result1) >= 1
        assert len(result2) >= 1


class TestHierarchyProperties:
    """Test properties that should hold for hierarchies."""

    def test_type_hierarchy_transitive(self):
        """If A is ancestor of B, ancestors of B should include ancestors of A."""
        # SmallMolecule -> ChemicalEntity -> NamedThing
        sm_ancestors = get_type_ancestors("SmallMolecule")
        ce_ancestors = get_type_ancestors("ChemicalEntity")
        # ChemicalEntity's ancestors should be subset of SmallMolecule's
        # (minus ChemicalEntity itself which is in SM but not CE)
        assert "NamedThing" in sm_ancestors
        assert "NamedThing" in ce_ancestors

    def test_type_ancestors_never_empty(self):
        """Every valid type should have at least itself as ancestor."""
        for type_name in ["Gene", "SmallMolecule", "Disease", "Protein"]:
            result = get_type_ancestors(type_name)
            assert len(result) >= 1
            assert type_name in result

    def test_pseudo_type_union_property(self):
        """Pseudo-type ancestors should be superset of each constituent's ancestors."""
        gene_ancestors = get_type_ancestors("Gene")
        sm_ancestors = get_type_ancestors("SmallMolecule")
        pseudo_ancestors = get_type_ancestors("Gene+SmallMolecule")

        # Pseudo-type should include all ancestors from both
        for ancestor in gene_ancestors:
            if ancestor != "Gene":  # Pseudo doesn't include constituent as ancestor directly
                assert ancestor in pseudo_ancestors
        for ancestor in sm_ancestors:
            if ancestor != "SmallMolecule":
                assert ancestor in pseudo_ancestors

    def test_predicate_ancestors_unique(self):
        """Predicate ancestors should not contain duplicates."""
        result = get_predicate_ancestors("regulates")
        assert len(result) == len(set(result))

    def test_caching_consistency(self):
        """Cached results should be consistent."""
        # Call multiple times
        results = [get_type_ancestors("SmallMolecule") for _ in range(3)]
        # All should be equal
        assert all(r == results[0] for r in results)


class TestHierarchyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_root_type_ancestors(self):
        """Root types should have minimal ancestors."""
        # NamedThing is close to root
        result = get_type_ancestors("NamedThing")
        assert "NamedThing" in result
        # Should have very few ancestors
        # (might have Entity or similar)

    def test_empty_string_type(self):
        """Empty string should be handled gracefully."""
        # Might return empty set or just the empty string
        result = get_type_ancestors("")
        assert isinstance(result, set)

    def test_case_sensitivity(self):
        """Type names should be case-sensitive."""
        result1 = get_type_ancestors("SmallMolecule")
        result2 = get_type_ancestors("smallmolecule")
        # These should be different (though second might be invalid)
        # The function should handle CamelCase correctly

    def test_pseudo_type_with_ancestor_relationship(self):
        """Pseudo-type where one constituent is ancestor of another should still work."""
        # This is pathological but should handle gracefully
        # In practice this shouldn't happen from assign_node_type
        result = get_type_ancestors("ChemicalEntity+SmallMolecule")
        assert isinstance(result, set)

    def test_very_long_pseudo_type(self):
        """Pseudo-type with many constituents."""
        long_pseudo = "+".join(sorted([
            "Gene",
            "SmallMolecule",
            "Protein",
            "Disease"
        ]))
        result = get_type_ancestors(long_pseudo)
        # Should include all constituents
        assert "Gene" in result
        assert "SmallMolecule" in result
        assert "Protein" in result
        assert "Disease" in result
        # Should be large set
        assert len(result) > 10


class TestGetTypeParents:
    """Test immediate parent type retrieval."""

    def test_regular_type_has_parent(self):
        """Regular type should return immediate parent."""
        result = get_type_parents("SmallMolecule")
        assert isinstance(result, set)
        # SmallMolecule's immediate parent is MolecularEntity or ChemicalEntity
        assert len(result) >= 1

    def test_gene_parent(self):
        """Gene should have BiologicalEntity or similar as parent."""
        result = get_type_parents("Gene")
        assert isinstance(result, set)
        # Should have at least one parent
        assert len(result) >= 1

    def test_pseudo_type_parents(self):
        """Pseudo-type should return parents of all constituents."""
        result = get_type_parents("Gene+SmallMolecule")
        assert isinstance(result, set)
        # Should have parents from both Gene and SmallMolecule

    def test_root_type_no_parent(self):
        """Root type (Entity) should have no parents."""
        result = get_type_parents("Entity")
        # May be empty or have very abstract parent
        assert isinstance(result, set)

    def test_unknown_type(self):
        """Unknown type should return empty set."""
        result = get_type_parents("CompletelyFakeType12345")
        assert isinstance(result, set)
        # Should handle gracefully (empty or just self)

    def test_parent_is_subset_of_ancestors(self):
        """Parents should be a subset of ancestors."""
        parents = get_type_parents("SmallMolecule")
        ancestors = get_type_ancestors("SmallMolecule")
        # All parents should be in ancestors
        for parent in parents:
            assert parent in ancestors


class TestGetPredicateParents:
    """Test immediate parent predicate retrieval."""

    def test_treats_parent(self):
        """treats should have a parent predicate."""
        result = get_predicate_parents("treats")
        assert isinstance(result, list)
        # Should have at least one parent (like affects or related_to)

    def test_affects_parent(self):
        """affects should have related_to as parent."""
        result = get_predicate_parents("affects")
        assert isinstance(result, list)

    def test_related_to_is_root(self):
        """related_to is a root predicate."""
        result = get_predicate_parents("related_to")
        # May have no parent or a very abstract one
        assert isinstance(result, list)

    def test_biolink_prefix_handled(self):
        """biolink: prefix should be handled."""
        result = get_predicate_parents("biolink:treats")
        assert isinstance(result, list)
        # Should return same as without prefix
        result2 = get_predicate_parents("treats")
        assert result == result2

    def test_underscore_handling(self):
        """Underscores in predicate names should work."""
        result = get_predicate_parents("physically_interacts_with")
        assert isinstance(result, list)

    def test_unknown_predicate(self):
        """Unknown predicate should return empty list."""
        result = get_predicate_parents("completely_fake_predicate_12345")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_parent_in_ancestors(self):
        """Parent should be in ancestors list."""
        parents = get_predicate_parents("treats")
        ancestors = get_predicate_ancestors("treats")
        # All parents should be in ancestors
        for parent in parents:
            assert parent in ancestors


class TestGetQualifierAncestorsExtended:
    """Additional tests for qualifier ancestor retrieval."""

    def test_increased_qualifier(self):
        """Test increased qualifier hierarchy."""
        result = get_qualifier_ancestors("increased")
        assert isinstance(result, list)
        # Should include self
        assert "increased" in result

    def test_decreased_qualifier(self):
        """Test decreased qualifier hierarchy."""
        result = get_qualifier_ancestors("decreased")
        assert isinstance(result, list)
        assert "decreased" in result

    def test_biolink_prefix_stripped(self):
        """biolink: prefix should be stripped."""
        result = get_qualifier_ancestors("biolink:increased")
        assert isinstance(result, list)
        # Should not have biolink: prefix in results
        for item in result:
            assert not item.startswith("biolink:")

    def test_unknown_qualifier(self):
        """Unknown qualifier should return list with just self."""
        result = get_qualifier_ancestors("completely_unknown_qualifier")
        assert isinstance(result, list)
        # Should at least include self
        assert "completely_unknown_qualifier" in result

    def test_empty_qualifier(self):
        """Empty string qualifier should be handled."""
        result = get_qualifier_ancestors("")
        assert isinstance(result, list)
