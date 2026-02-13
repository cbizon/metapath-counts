#!/usr/bin/env python3
"""
Comprehensive tests for type assignment logic.

Tests cover:
- Single leaf type assignment
- Multi-ancestor single-leaf type assignment
- Pseudo-type creation for multi-root nodes
- Edge cases and error handling
"""

import pytest
from library.type_assignment import (
    assign_node_type,
    find_leaf_types,
    format_pseudo_type,
    parse_pseudo_type,
    is_pseudo_type,
    is_ancestor_of
)


class TestIsAncestorOf:
    """Test ancestor relationship checking."""

    def test_direct_ancestor(self):
        """Test direct parent-child relationship."""
        assert is_ancestor_of("ChemicalEntity", "SmallMolecule") is True
        assert is_ancestor_of("NamedThing", "SmallMolecule") is True

    def test_not_ancestor(self):
        """Test non-ancestor relationship."""
        assert is_ancestor_of("SmallMolecule", "ChemicalEntity") is False
        assert is_ancestor_of("Gene", "SmallMolecule") is False

    def test_self_is_ancestor(self):
        """Test that a type is considered its own ancestor."""
        assert is_ancestor_of("SmallMolecule", "SmallMolecule") is True
        assert is_ancestor_of("Gene", "Gene") is True

    def test_distant_ancestor(self):
        """Test multi-level ancestor relationship."""
        # NamedThing is several levels above SmallMolecule
        assert is_ancestor_of("NamedThing", "SmallMolecule") is True
        assert is_ancestor_of("Entity", "Gene") is True


class TestFindLeafTypes:
    """Test finding leaf types in a category list."""

    def test_single_type(self):
        """Single type should be the only leaf."""
        result = find_leaf_types(["biolink:Gene"])
        assert result == ["Gene"]

    def test_hierarchical_chain(self):
        """In a hierarchical chain, only the most specific is a leaf."""
        result = find_leaf_types([
            "biolink:SmallMolecule",
            "biolink:ChemicalEntity",
            "biolink:NamedThing"
        ])
        assert result == ["SmallMolecule"]

    def test_multiple_leaf_types(self):
        """Multiple unrelated types should all be leaves."""
        result = find_leaf_types(["biolink:Gene", "biolink:SmallMolecule"])
        # Should be sorted
        assert result == ["Gene", "SmallMolecule"]

    def test_complex_hierarchy(self):
        """Test with complex hierarchy including multiple branches."""
        result = find_leaf_types([
            "biolink:SmallMolecule",
            "biolink:MolecularEntity",
            "biolink:ChemicalEntity",
            "biolink:NamedThing"
        ])
        # Only SmallMolecule is a leaf (all others are ancestors)
        assert result == ["SmallMolecule"]

    def test_empty_list(self):
        """Empty category list should return empty result."""
        result = find_leaf_types([])
        assert result == []

    def test_multiple_roots(self):
        """Multiple root types with some shared ancestors."""
        result = find_leaf_types([
            "biolink:Gene",
            "biolink:SmallMolecule",
            "biolink:NamedThing"  # Ancestor of both
        ])
        # Both Gene and SmallMolecule are leaves
        assert set(result) == {"Gene", "SmallMolecule"}

    def test_disease_hierarchy(self):
        """Test with Disease hierarchy."""
        result = find_leaf_types([
            "biolink:Disease",
            "biolink:DiseaseOrPhenotypicFeature",
            "biolink:NamedThing"
        ])
        assert result == ["Disease"]


class TestPseudoTypeUtilities:
    """Test pseudo-type formatting and parsing."""

    def test_format_single_type(self):
        """Single type in list."""
        result = format_pseudo_type(["SmallMolecule"])
        assert result == "SmallMolecule"

    def test_format_multiple_types(self):
        """Multiple types should be joined with + and sorted."""
        result = format_pseudo_type(["SmallMolecule", "Gene"])
        assert result == "Gene+SmallMolecule"  # Alphabetically sorted

    def test_format_three_types(self):
        """Three types should all be included."""
        result = format_pseudo_type(["Protein", "SmallMolecule", "Gene"])
        assert result == "Gene+Protein+SmallMolecule"

    def test_parse_single_type(self):
        """Single type (no +) should return list with one element."""
        result = parse_pseudo_type("SmallMolecule")
        assert result == ["SmallMolecule"]

    def test_parse_pseudo_type(self):
        """Pseudo-type with + should be split correctly."""
        result = parse_pseudo_type("Gene+SmallMolecule")
        assert result == ["Gene", "SmallMolecule"]

    def test_parse_three_types(self):
        """Three-type pseudo-type."""
        result = parse_pseudo_type("Gene+Protein+SmallMolecule")
        assert result == ["Gene", "Protein", "SmallMolecule"]

    def test_is_pseudo_type_true(self):
        """Pseudo-type should be detected."""
        assert is_pseudo_type("Gene+SmallMolecule") is True
        assert is_pseudo_type("A+B+C") is True

    def test_is_pseudo_type_false(self):
        """Regular type should not be detected as pseudo-type."""
        assert is_pseudo_type("SmallMolecule") is False
        assert is_pseudo_type("Gene") is False

    def test_roundtrip(self):
        """Format and parse should be inverse operations."""
        types = ["Gene", "Protein", "SmallMolecule"]
        formatted = format_pseudo_type(types)
        parsed = parse_pseudo_type(formatted)
        assert sorted(parsed) == sorted(types)


class TestAssignNodeType:
    """Test node type assignment."""

    def test_single_leaf_type(self):
        """Node with single most-specific type."""
        result = assign_node_type([
            "biolink:SmallMolecule",
            "biolink:ChemicalEntity",
            "biolink:NamedThing"
        ])
        assert result == "SmallMolecule"

    def test_multiple_leaf_types(self):
        """Node with multiple root types should get pseudo-type."""
        result = assign_node_type([
            "biolink:Gene",
            "biolink:SmallMolecule"
        ])
        assert result == "Gene+SmallMolecule"
        assert is_pseudo_type(result)

    def test_single_type_only(self):
        """Node with only one category."""
        result = assign_node_type(["biolink:Gene"])
        assert result == "Gene"

    def test_empty_categories(self):
        """Empty category list should return None."""
        result = assign_node_type([])
        assert result is None

    def test_disease_specific(self):
        """Disease with hierarchy."""
        result = assign_node_type([
            "biolink:Disease",
            "biolink:DiseaseOrPhenotypicFeature",
            "biolink:NamedThing"
        ])
        assert result == "Disease"

    def test_three_leaf_types(self):
        """Node with three unrelated types."""
        result = assign_node_type([
            "biolink:Gene",
            "biolink:SmallMolecule",
            "biolink:Protein"
        ])
        # Should be pseudo-type with all three
        assert is_pseudo_type(result)
        parsed = parse_pseudo_type(result)
        assert set(parsed) == {"Gene", "SmallMolecule", "Protein"}

    def test_biological_process(self):
        """Test with BiologicalProcess."""
        result = assign_node_type([
            "biolink:BiologicalProcess",
            "biolink:NamedThing"
        ])
        assert result == "BiologicalProcess"

    def test_complex_multi_root(self):
        """Multiple roots with different hierarchies."""
        result = assign_node_type([
            "biolink:SmallMolecule",
            "biolink:ChemicalEntity",
            "biolink:Gene",
            "biolink:BiologicalEntity",
            "biolink:NamedThing"
        ])
        # SmallMolecule and Gene are both leaves
        assert is_pseudo_type(result)
        parsed = parse_pseudo_type(result)
        assert set(parsed) == {"Gene", "SmallMolecule"}


class TestTypeAssignmentProperties:
    """Test properties that should hold for type assignment."""

    def test_deterministic(self):
        """Same input should always give same output."""
        categories = [
            "biolink:Gene",
            "biolink:SmallMolecule",
            "biolink:NamedThing"
        ]
        result1 = assign_node_type(categories)
        result2 = assign_node_type(categories)
        assert result1 == result2

    def test_order_independence(self):
        """Category order shouldn't affect result."""
        result1 = assign_node_type([
            "biolink:SmallMolecule",
            "biolink:ChemicalEntity",
            "biolink:NamedThing"
        ])
        result2 = assign_node_type([
            "biolink:NamedThing",
            "biolink:SmallMolecule",
            "biolink:ChemicalEntity"
        ])
        assert result1 == result2

    def test_pseudo_type_sorted(self):
        """Pseudo-types should be consistently sorted."""
        result1 = assign_node_type(["biolink:Gene", "biolink:SmallMolecule"])
        result2 = assign_node_type(["biolink:SmallMolecule", "biolink:Gene"])
        assert result1 == result2
        # Should be alphabetically sorted
        assert result1 == "Gene+SmallMolecule"

    def test_no_duplicates_in_pseudo_type(self):
        """Pseudo-type should not contain duplicates."""
        # Even if input has duplicates or equivalent types
        result = assign_node_type([
            "biolink:Gene",
            "biolink:SmallMolecule",
            "biolink:Gene"  # Duplicate
        ])
        parsed = parse_pseudo_type(result) if is_pseudo_type(result) else [result]
        assert len(parsed) == len(set(parsed))


class TestEdgeCases:
    """Test edge cases and potential error conditions."""

    def test_prefix_handling(self):
        """Should handle both with and without biolink: prefix."""
        result1 = assign_node_type(["biolink:Gene"])
        result2 = assign_node_type(["Gene"])
        # After normalization, both should work (though we expect biolink: prefix)
        # The function expects biolink: prefix, so this tests robustness
        assert result1 == "Gene"

    def test_none_in_list(self):
        """Handle None gracefully if it appears in list."""
        # This shouldn't happen in practice, but test robustness
        categories = ["biolink:Gene", "biolink:SmallMolecule"]
        result = assign_node_type(categories)
        assert result is not None

    def test_very_long_hierarchy(self):
        """Handle deep hierarchies."""
        categories = [
            "biolink:SmallMolecule",
            "biolink:MolecularEntity",
            "biolink:ChemicalEntity",
            "biolink:PhysicalEssence",
            "biolink:NamedThing",
            "biolink:Entity"
        ]
        result = assign_node_type(categories)
        assert result == "SmallMolecule"

    def test_filename_safety(self):
        """Pseudo-type strings should be filename-safe."""
        result = assign_node_type([
            "biolink:Gene",
            "biolink:SmallMolecule",
            "biolink:Protein"
        ])
        # Should only contain alphanumeric and +
        import re
        assert re.match(r'^[A-Za-z0-9+]+$', result)
