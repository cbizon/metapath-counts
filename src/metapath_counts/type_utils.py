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


def filter_abstract_types(types: list[str], exclude_list: list[str] | set[str]) -> list[str]:
    """
    Filter out abstract types from a list of types.

    Args:
        types: List of type names (with or without biolink: prefix)
        exclude_list: List or set of type names to exclude (with or without biolink: prefix)

    Returns:
        List of types with excluded types removed, preserving order

    Example:
        >>> filter_abstract_types(
        ...     ['SmallMolecule', 'ChemicalEntity', 'NamedThing'],
        ...     ['NamedThing', 'PhysicalEssence']
        ... )
        ['SmallMolecule', 'ChemicalEntity']
    """
    if not types:
        return []

    if not exclude_list:
        return types.copy()

    # Normalize both lists to remove biolink: prefix for comparison
    normalized_exclude = {t.replace('biolink:', '').lower() for t in exclude_list}

    # Filter, case-insensitive matching
    filtered = []
    for type_name in types:
        clean_type = type_name.replace('biolink:', '')
        if clean_type.lower() not in normalized_exclude:
            filtered.append(type_name)

    return filtered


def get_all_types(
    categories: list[str],
    exclude_types: set[str] = None,
    max_depth: int = None,
    include_most_specific: bool = True
) -> list[str]:
    """
    Given a list of biolink categories, return all types (optionally filtered).

    This enables hierarchical type expansion where nodes can participate
    as multiple types in the hierarchy.

    Args:
        categories: List of biolink types (e.g., ["biolink:SmallMolecule", "biolink:ChemicalEntity"])
        exclude_types: Set of type names to exclude (without biolink: prefix)
                      e.g., {'NamedThing', 'PhysicalEssence'}
        max_depth: Maximum number of types to include, ordered by specificity
                  None = all types, 1 = most specific only, 2 = most specific + 1 parent, etc.
        include_most_specific: If True, always include most specific type even if it would be filtered

    Returns:
        List of type names (without biolink: prefix), ordered by specificity (most specific first)

    Example:
        >>> get_all_types(
        ...     ["biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:NamedThing"],
        ...     exclude_types={'NamedThing'}
        ... )
        ['SmallMolecule', 'ChemicalEntity']
    """
    if not categories:
        return []

    if exclude_types is None:
        exclude_types = set()

    # Normalize exclude_types to handle both formats
    normalized_exclude = {t.replace('biolink:', '') for t in exclude_types}

    # Build list of (ancestor_count, type_name_without_prefix)
    type_depths = []
    for category in categories:
        # Remove biolink: prefix
        clean_type = category.replace('biolink:', '')

        # Get ancestor count (cached)
        ancestor_count = _get_ancestor_count(clean_type)

        # Skip invalid types
        if ancestor_count < 0:
            continue

        type_depths.append((ancestor_count, clean_type))

    if not type_depths:
        return []

    # Sort by specificity (most ancestors = most specific)
    type_depths.sort(reverse=True)

    # Get most specific type before filtering
    most_specific = type_depths[0][1] if type_depths else None

    # Filter by exclude_types
    filtered = []
    for ancestor_count, type_name in type_depths:
        if type_name not in normalized_exclude:
            filtered.append(type_name)

    # Apply max_depth limit
    if max_depth is not None:
        if max_depth <= 0:
            # Special case: max_depth=0 means no types unless include_most_specific=True
            filtered = []
        else:
            filtered = filtered[:max_depth]

    # Ensure most specific is included if requested
    if include_most_specific and most_specific:
        if most_specific not in filtered:
            # Add most specific at the beginning
            filtered = [most_specific] + filtered
            # Remove duplicates while preserving order
            seen = set()
            deduped = []
            for t in filtered:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            filtered = deduped

    return filtered


if __name__ == "__main__":
    # Test get_most_specific_type
    print("=" * 60)
    print("Testing get_most_specific_type()")
    print("=" * 60)
    test_cases = [
        # PhysicalEssenceOrOccurrent is abstract, BiologicalProcess is more specific
        ["biolink:PhysicalEssenceOrOccurrent", "biolink:BiologicalProcess"],

        # SmallMolecule is more specific than ChemicalEntity
        ["biolink:ChemicalEntity", "biolink:SmallMolecule"],

        # Disease is more specific than DiseaseOrPhenotypicFeature
        ["biolink:DiseaseOrPhenotypicFeature", "biolink:Disease"],
    ]

    for categories in test_cases:
        most_specific = get_most_specific_type(categories)
        print(f"{categories}")
        print(f"  → Most specific: {most_specific}\n")

    # Test filter_abstract_types
    print("=" * 60)
    print("Testing filter_abstract_types()")
    print("=" * 60)

    # Test case 1: Basic filtering
    types = ['SmallMolecule', 'ChemicalEntity', 'NamedThing']
    exclude = ['NamedThing', 'PhysicalEssence']
    filtered = filter_abstract_types(types, exclude)
    print(f"Types: {types}")
    print(f"Exclude: {exclude}")
    print(f"  → {filtered}\n")

    # Test case 2: Case insensitive
    types = ['SmallMolecule', 'ChemicalEntity', 'NamedThing']
    exclude = ['namedthing', 'PHYSICALESSENCE']
    filtered = filter_abstract_types(types, exclude)
    print(f"Types: {types}")
    print(f"Exclude (case insensitive): {exclude}")
    print(f"  → {filtered}\n")

    # Test case 3: With biolink: prefix
    types = ['biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:NamedThing']
    exclude = ['NamedThing']
    filtered = filter_abstract_types(types, exclude)
    print(f"Types (with prefix): {types}")
    print(f"Exclude: {exclude}")
    print(f"  → {filtered}\n")

    # Test case 4: Mixed format
    types = ['biolink:SmallMolecule', 'ChemicalEntity', 'biolink:NamedThing']
    exclude = ['biolink:NamedThing']
    filtered = filter_abstract_types(types, exclude)
    print(f"Types (mixed): {types}")
    print(f"Exclude (with prefix): {exclude}")
    print(f"  → {filtered}\n")

    # Test get_all_types
    print("=" * 60)
    print("Testing get_all_types()")
    print("=" * 60)

    # Test case 1: Basic usage - return all types
    categories = ["biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:NamedThing"]
    all_types = get_all_types(categories)
    print(f"All types (no filtering): {categories}")
    print(f"  → {all_types}\n")

    # Test case 2: Exclude abstract types
    exclude = {'NamedThing', 'PhysicalEssence'}
    all_types = get_all_types(categories, exclude_types=exclude)
    print(f"With exclude_types={exclude}")
    print(f"  → {all_types}\n")

    # Test case 3: Max depth limiting
    all_types = get_all_types(categories, max_depth=2)
    print(f"With max_depth=2")
    print(f"  → {all_types}\n")

    # Test case 4: Exclude most specific but include_most_specific=True
    all_types = get_all_types(
        categories,
        exclude_types={'SmallMolecule'},
        include_most_specific=True
    )
    print(f"Exclude SmallMolecule but include_most_specific=True")
    print(f"  → {all_types}\n")

    # Test case 5: Complex hierarchy
    complex_cats = [
        "biolink:SmallMolecule",
        "biolink:MolecularEntity",
        "biolink:ChemicalEntity",
        "biolink:PhysicalEssence",
        "biolink:NamedThing"
    ]
    all_types = get_all_types(
        complex_cats,
        exclude_types={'PhysicalEssence', 'NamedThing'},
        max_depth=3
    )
    print(f"Complex hierarchy with filtering:")
    print(f"  Input: {complex_cats}")
    print(f"  Exclude: PhysicalEssence, NamedThing")
    print(f"  Max depth: 3")
    print(f"  → {all_types}\n")
