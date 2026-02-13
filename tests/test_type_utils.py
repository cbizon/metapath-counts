"""Tests for type_utils module."""

import pytest
from library import get_most_specific_type, get_symmetric_predicates


def test_get_most_specific_type_basic():
    """Test getting most specific type from a list."""
    categories = ["biolink:ChemicalEntity", "biolink:SmallMolecule"]
    result = get_most_specific_type(categories)
    assert result == "biolink:SmallMolecule"


def test_get_most_specific_type_single():
    """Test with single category."""
    categories = ["biolink:Disease"]
    result = get_most_specific_type(categories)
    assert result == "biolink:Disease"


def test_get_most_specific_type_empty():
    """Test with empty list."""
    categories = []
    result = get_most_specific_type(categories)
    assert result is None


def test_get_symmetric_predicates():
    """Test that symmetric predicates are returned."""
    predicates = get_symmetric_predicates()

    # Should be a set
    assert isinstance(predicates, set)

    # Should contain known symmetric predicates
    assert "directly_physically_interacts_with" in predicates
    assert "associated_with" in predicates

    # Should have multiple predicates
    assert len(predicates) > 10


def test_get_symmetric_predicates_cached():
    """Test that symmetric predicates are cached."""
    # Call twice - should return same object
    predicates1 = get_symmetric_predicates()
    predicates2 = get_symmetric_predicates()

    assert predicates1 is predicates2
