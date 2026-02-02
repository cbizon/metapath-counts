#!/usr/bin/env python3
"""
Type assignment logic for single-type-per-node approach.

Each node is assigned to exactly ONE type or pseudo-type:
- If node has single most-specific type: use that type
- If node has multiple leaf types: create pseudo-type from those types

This avoids matrix explosion while preserving ability to infer hierarchical paths.
"""

import bmt
from functools import lru_cache


# Global toolkit singleton
_TOOLKIT = None


def _get_toolkit():
    """Get or create biolink toolkit singleton."""
    global _TOOLKIT
    if _TOOLKIT is None:
        _TOOLKIT = bmt.Toolkit()
    return _TOOLKIT


def _normalize_type_for_bmt(type_name: str) -> str:
    """
    Normalize type name for BMT lookup.

    BMT uses lowercase with spaces (e.g., "small molecule").
    We use CamelCase (e.g., "SmallMolecule").

    Args:
        type_name: Type name in CamelCase format

    Returns:
        Type name in BMT format (lowercase with spaces)
    """
    # Convert CamelCase to space-separated lowercase
    # Insert space before capital letters, then lowercase
    import re
    # Insert space before uppercase letters that follow lowercase letters
    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', type_name)
    return spaced.lower()


def is_ancestor_of(potential_ancestor: str, potential_descendant: str) -> bool:
    """
    Check if one type is an ancestor of another in the Biolink hierarchy.

    Args:
        potential_ancestor: Type that might be ancestor (e.g., "ChemicalEntity")
        potential_descendant: Type that might be descendant (e.g., "SmallMolecule")

    Returns:
        True if potential_ancestor is an ancestor of potential_descendant

    Example:
        >>> is_ancestor_of("ChemicalEntity", "SmallMolecule")
        True
        >>> is_ancestor_of("SmallMolecule", "ChemicalEntity")
        False
    """
    if potential_ancestor == potential_descendant:
        return True

    toolkit = _get_toolkit()

    # Normalize type names for BMT lookup
    ancestor_normalized = _normalize_type_for_bmt(potential_ancestor)
    descendant_normalized = _normalize_type_for_bmt(potential_descendant)

    # Get ancestors of the potential descendant
    try:
        ancestors = toolkit.get_ancestors(descendant_normalized)
        # get_ancestors returns list including the type itself
        return ancestor_normalized in ancestors
    except Exception:
        # If type lookup fails, assume not ancestor
        return False


def find_leaf_types(categories: list[str]) -> list[str]:
    """
    Find leaf types (types with no descendants) in a list of categories.

    A leaf type is one where no other type in the list is a descendant of it.

    Args:
        categories: List of biolink types (with or without biolink: prefix)

    Returns:
        List of leaf types (without biolink: prefix), sorted for consistency

    Example:
        >>> find_leaf_types(["biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:NamedThing"])
        ['SmallMolecule']  # Most specific type

        >>> find_leaf_types(["biolink:SmallMolecule", "biolink:Gene"])
        ['Gene', 'SmallMolecule']  # Two unrelated leaf types
    """
    if not categories:
        return []

    # Normalize: remove biolink: prefix
    clean_types = [cat.replace('biolink:', '') for cat in categories]

    # For each type, check if any other type is its descendant
    leaf_types = []
    for type_i in clean_types:
        is_leaf = True
        for type_j in clean_types:
            if type_i != type_j:
                # If type_j is a descendant of type_i, then type_i is not a leaf
                if is_ancestor_of(type_i, type_j):
                    is_leaf = False
                    break

        if is_leaf:
            leaf_types.append(type_i)

    # Remove duplicates and sort for consistency
    leaf_types = sorted(set(leaf_types))

    return leaf_types


def format_pseudo_type(leaf_types: list[str]) -> str:
    """
    Format multiple leaf types as a pseudo-type string.

    Uses '+' separator and ensures filename-safe output.

    Args:
        leaf_types: List of leaf type names (without biolink: prefix)

    Returns:
        Pseudo-type string (e.g., "Gene+SmallMolecule")

    Example:
        >>> format_pseudo_type(["SmallMolecule", "Gene"])
        'Gene+SmallMolecule'  # Sorted alphabetically
    """
    # Sort for consistency
    sorted_types = sorted(leaf_types)
    return '+'.join(sorted_types)


def parse_pseudo_type(pseudo_type_str: str) -> list[str]:
    """
    Parse a pseudo-type string back into constituent types.

    Args:
        pseudo_type_str: Pseudo-type string (e.g., "Gene+SmallMolecule")

    Returns:
        List of constituent types

    Example:
        >>> parse_pseudo_type("Gene+SmallMolecule")
        ['Gene', 'SmallMolecule']
    """
    if '+' not in pseudo_type_str:
        return [pseudo_type_str]
    return pseudo_type_str.split('+')


def is_pseudo_type(type_str: str) -> bool:
    """
    Check if a type string represents a pseudo-type.

    Args:
        type_str: Type string to check

    Returns:
        True if this is a pseudo-type (contains '+')

    Example:
        >>> is_pseudo_type("SmallMolecule")
        False
        >>> is_pseudo_type("Gene+SmallMolecule")
        True
    """
    return '+' in type_str


def assign_node_type(categories: list[str]) -> str:
    """
    Assign a node to exactly ONE type or pseudo-type.

    Logic:
    - If single leaf type: return that type
    - If multiple leaf types: return pseudo-type (concatenated with '+')

    This ensures each node participates in exactly one type during matrix
    building, avoiding explosion while preserving ability to infer hierarchy.

    Args:
        categories: List of biolink types (e.g., ["biolink:SmallMolecule", "biolink:ChemicalEntity"])

    Returns:
        Single type name (without biolink: prefix) or pseudo-type string

    Examples:
        >>> assign_node_type(["biolink:SmallMolecule", "biolink:ChemicalEntity"])
        'SmallMolecule'  # Single leaf type

        >>> assign_node_type(["biolink:SmallMolecule", "biolink:Gene"])
        'Gene+SmallMolecule'  # Pseudo-type for multiple roots

        >>> assign_node_type([])
        None
    """
    if not categories:
        return None

    # Find leaf types (types with no descendants in the list)
    leaf_types = find_leaf_types(categories)

    if not leaf_types:
        # Shouldn't happen with valid Biolink types, but handle gracefully
        # Just use the first category
        return categories[0].replace('biolink:', '')

    if len(leaf_types) == 1:
        # Single most-specific type
        return leaf_types[0]
    else:
        # Multiple leaf types - create pseudo-type
        return format_pseudo_type(leaf_types)


if __name__ == "__main__":
    # Test cases
    print("=" * 60)
    print("Testing Type Assignment Logic")
    print("=" * 60)

    test_cases = [
        # Single leaf type
        (["biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:NamedThing"],
         "SmallMolecule"),

        # Multiple leaf types
        (["biolink:SmallMolecule", "biolink:Gene"],
         "Gene+SmallMolecule"),

        # Single type only
        (["biolink:Gene"],
         "Gene"),

        # Complex hierarchy
        (["biolink:Disease", "biolink:DiseaseOrPhenotypicFeature", "biolink:NamedThing"],
         "Disease"),
    ]

    print("\nTest: assign_node_type()")
    print("-" * 60)
    for categories, expected in test_cases:
        result = assign_node_type(categories)
        status = "✓" if result == expected else "✗"
        print(f"{status} {categories}")
        print(f"  → {result} (expected: {expected})")
        print()

    # Test pseudo-type utilities
    print("\nTest: Pseudo-type utilities")
    print("-" * 60)
    pseudo = format_pseudo_type(["SmallMolecule", "Gene", "Protein"])
    print(f"format_pseudo_type(['SmallMolecule', 'Gene', 'Protein']) = {pseudo}")

    parsed = parse_pseudo_type(pseudo)
    print(f"parse_pseudo_type('{pseudo}') = {parsed}")

    print(f"is_pseudo_type('SmallMolecule') = {is_pseudo_type('SmallMolecule')}")
    print(f"is_pseudo_type('{pseudo}') = {is_pseudo_type(pseudo)}")
