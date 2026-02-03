#!/usr/bin/env python3
"""
Utilities for working with biolink types.
"""

import bmt
from functools import lru_cache


# Global cache for ancestor counts
_ANCESTOR_CACHE = {}
_TOOLKIT = None


def _get_toolkit():
    """Get or create biolink toolkit singleton."""
    global _TOOLKIT
    if _TOOLKIT is None:
        _TOOLKIT = bmt.Toolkit()
    return _TOOLKIT


def _get_ancestor_count(biolink_type: str) -> int:
    """
    Get the number of ancestors for a biolink type (cached).

    Args:
        biolink_type: Type without biolink: prefix (e.g., "SmallMolecule")

    Returns:
        Number of ancestors
    """
    if biolink_type not in _ANCESTOR_CACHE:
        toolkit = _get_toolkit()
        if toolkit.is_category(biolink_type):
            ancestors = toolkit.get_ancestors(biolink_type)
            _ANCESTOR_CACHE[biolink_type] = len(ancestors)
        else:
            # Invalid type, give it -1 so it sorts last
            _ANCESTOR_CACHE[biolink_type] = -1

    return _ANCESTOR_CACHE[biolink_type]


@lru_cache(maxsize=1)
def get_symmetric_predicates() -> set:
    """
    Get all symmetric predicates from the Biolink Model.

    Returns:
        set: Set of symmetric predicate names (without biolink: prefix)
             e.g., {'directly_physically_interacts_with', 'associated_with', ...}
    """
    toolkit = _get_toolkit()
    symmetric_predicates = set()

    # Get all predicate slots
    all_predicates = toolkit.get_all_slots()

    for predicate_name in all_predicates:
        predicate = toolkit.get_element(predicate_name)

        # Check if predicate is marked as symmetric
        if predicate and predicate.symmetric:
            # Use slot_uri which has proper formatting: biolink:predicate_name
            formatted_name = predicate.slot_uri.replace('biolink:', '')
            symmetric_predicates.add(formatted_name)

    return symmetric_predicates


def get_most_specific_type(categories: list[str]) -> str:
    """
    Given a list of biolink categories, return the most specific one.

    Most specific = fewest ancestors in the biolink hierarchy.

    Args:
        categories: List of biolink types (e.g., ["biolink:SmallMolecule", "biolink:ChemicalEntity"])

    Returns:
        Most specific type (with biolink: prefix)

    Example:
        >>> get_most_specific_type([
        ...     "biolink:PhysicalEssenceOrOccurrent",
        ...     "biolink:BiologicalProcess"
        ... ])
        'biolink:BiologicalProcess'
    """
    if not categories:
        return None

    # For each category, count ancestors using cache
    type_depths = []
    for category in categories:
        # Remove biolink: prefix for bmt
        clean_type = category.replace('biolink:', '')

        # Get ancestor count (cached)
        ancestor_count = _get_ancestor_count(clean_type)

        # Skip invalid types
        if ancestor_count < 0:
            continue

        # More ancestors = more specific (further down the tree)
        # Fewer ancestors = more abstract (closer to root)
        type_depths.append((ancestor_count, category))

    if not type_depths:
        # If no valid categories found, just return first one
        return categories[0]

    # Sort by number of ancestors (descending = most specific first)
    # More ancestors = further down the tree = more specific
    type_depths.sort(reverse=True)

    return type_depths[0][1]


