#!/usr/bin/env python3
"""
Hierarchy inference utilities for post-processing aggregation.

Provides functions to get ancestors in type, predicate, and qualifier hierarchies.
"""

import bmt
from functools import lru_cache
from .type_assignment import parse_pseudo_type, is_pseudo_type, _normalize_type_for_bmt


# Global toolkit singleton
_TOOLKIT = None


def _get_toolkit():
    """Get or create biolink toolkit singleton."""
    global _TOOLKIT
    if _TOOLKIT is None:
        _TOOLKIT = bmt.Toolkit()
    return _TOOLKIT


@lru_cache(maxsize=1024)
def get_type_ancestors(type_name: str) -> set[str]:
    """
    Get all ancestor types for a type or pseudo-type.

    For pseudo-types, returns the union of ancestors of all constituents.

    Args:
        type_name: Type name (e.g., "SmallMolecule" or "Gene+SmallMolecule")

    Returns:
        Set of ancestor type names (without biolink: prefix)

    Example:
        >>> get_type_ancestors("SmallMolecule")
        {'ChemicalEntity', 'MolecularEntity', 'NamedThing', ...}

        >>> get_type_ancestors("Gene+SmallMolecule")
        {'ChemicalEntity', 'GeneOrGeneProduct', 'NamedThing', ...}  # Union of both
    """
    toolkit = _get_toolkit()

    if is_pseudo_type(type_name):
        # Pseudo-type: get union of all constituent ancestors
        constituents = parse_pseudo_type(type_name)
        all_ancestors = set()

        # Add the pseudo-type itself
        all_ancestors.add(type_name)

        for constituent in constituents:
            # Add constituent itself
            all_ancestors.add(constituent)
            # Add its ancestors
            normalized = _normalize_type_for_bmt(constituent)
            try:
                ancestors = toolkit.get_ancestors(normalized)
                # Convert back to CamelCase
                for ancestor in ancestors:
                    # Convert "small molecule" -> "SmallMolecule"
                    camel = ''.join(word.capitalize() for word in ancestor.split())
                    all_ancestors.add(camel)
            except Exception:
                # Skip if type not found
                pass

        return all_ancestors
    else:
        # Regular type
        ancestors = {type_name}
        normalized = _normalize_type_for_bmt(type_name)
        try:
            bmt_ancestors = toolkit.get_ancestors(normalized)
            for ancestor in bmt_ancestors:
                # Convert back to CamelCase
                camel = ''.join(word.capitalize() for word in ancestor.split())
                ancestors.add(camel)
        except Exception:
            # If type not found, just return the type itself
            pass

        return ancestors


@lru_cache(maxsize=1024)
def get_type_parents(type_name: str) -> set[str]:
    """
    Get immediate parent types (one step up in hierarchy).

    For pseudo-types, returns parents of constituents.

    Args:
        type_name: Type name (e.g., "SmallMolecule" or "Gene+SmallMolecule")

    Returns:
        Set of immediate parent type names

    Example:
        >>> get_type_parents("SmallMolecule")
        {'ChemicalEntity'}
    """
    toolkit = _get_toolkit()

    if is_pseudo_type(type_name):
        # Pseudo-type: get parents of each constituent
        constituents = parse_pseudo_type(type_name)
        parents = set()

        for constituent in constituents:
            normalized = _normalize_type_for_bmt(constituent)
            try:
                element = toolkit.get_element(normalized)
                if element and hasattr(element, 'is_a') and element.is_a:
                    # Convert to CamelCase
                    parent_camel = ''.join(word.capitalize() for word in element.is_a.split())
                    parents.add(parent_camel)
            except Exception:
                pass

        return parents
    else:
        # Regular type
        parents = set()
        normalized = _normalize_type_for_bmt(type_name)
        try:
            element = toolkit.get_element(normalized)
            if element and hasattr(element, 'is_a') and element.is_a:
                # Convert to CamelCase
                parent_camel = ''.join(word.capitalize() for word in element.is_a.split())
                parents.add(parent_camel)
        except Exception:
            pass

        return parents


@lru_cache(maxsize=1024)
def get_predicate_parents(predicate: str) -> list[str]:
    """
    Get immediate parent predicates (one step up in hierarchy).

    Args:
        predicate: Predicate name (with or without biolink: prefix)

    Returns:
        List of immediate parent predicates (without biolink: prefix)

    Example:
        >>> get_predicate_parents("treats")
        ['affects']
    """
    toolkit = _get_toolkit()

    # Remove biolink: prefix if present for bmt lookup
    pred_name = predicate.replace("biolink:", "").replace("_", " ")

    parents = []
    try:
        element = toolkit.get_element(pred_name)
        if element and hasattr(element, 'is_a') and element.is_a:
            # Return without biolink: prefix, with underscores
            parent = element.is_a.replace(' ', '_')
            parents.append(parent)
    except Exception:
        pass

    return parents


@lru_cache(maxsize=1024)
def get_predicate_ancestors(predicate: str) -> list[str]:
    """
    Get ancestor predicates from biolink model.

    Adapted from biolink-rule-mining/src/prepare/redundant_edges.py

    Args:
        predicate: Predicate name (with or without biolink: prefix, underscores or spaces)

    Returns:
        List of ancestor predicates (without biolink: prefix, with underscores)
        Does NOT include the predicate itself

    Example:
        >>> get_predicate_ancestors("treats")
        ['related_to']
    """
    toolkit = _get_toolkit()

    # Remove biolink: prefix if present for bmt lookup
    pred_name = predicate.replace("biolink:", "").replace("_", " ")

    ancestors = []
    try:
        # Get ancestors includes the element itself, so skip the first one
        all_ancestors = toolkit.get_ancestors(pred_name)
        if all_ancestors and len(all_ancestors) > 1:
            # Skip first element (the predicate itself)
            for ancestor in all_ancestors[1:]:
                # Return without biolink: prefix, with underscores
                formatted = ancestor.replace(' ', '_')
                ancestors.append(formatted)
    except Exception:
        # If predicate not found, return empty list
        pass

    return ancestors


def get_qualifier_ancestors(qualifier_value: str) -> list[str]:
    """
    Get ancestor values for a qualifier from biolink model.

    Handles both regular elements and enum values (e.g., direction/aspect qualifiers).
    Adapted from biolink-rule-mining/src/prepare/redundant_edges.py

    Args:
        qualifier_value: Qualifier value (e.g., "increased" or "biolink:increased")

    Returns:
        List of ancestor qualifier values (most specific first)
        Includes the qualifier value itself

    Example:
        >>> get_qualifier_ancestors("increased")
        ['increased', 'regulated']  # Example hierarchy
    """
    toolkit = _get_toolkit()

    # Remove any prefix for lookup
    qual_name = qualifier_value.split(":")[-1] if ":" in qualifier_value else qualifier_value

    ancestors = [qual_name]  # Start with original

    # Check qualifier enums for is_a relationships
    for enum_name in ['DirectionQualifierEnum', 'GeneOrGeneProductOrChemicalEntityAspectEnum']:
        try:
            enum_def = toolkit.get_element(enum_name)
            if enum_def and hasattr(enum_def, 'permissible_values'):
                if qual_name in enum_def.permissible_values:
                    # Walk up the is_a chain for this enum value
                    current = enum_def.permissible_values[qual_name]
                    while current and hasattr(current, 'is_a') and current.is_a:
                        parent = current.is_a
                        if parent not in ancestors:
                            ancestors.append(parent)
                        # Get the parent's definition to continue walking
                        if parent in enum_def.permissible_values:
                            current = enum_def.permissible_values[parent]
                        else:
                            break
                    return ancestors
        except Exception:
            pass

    # If not found in enums, try regular element hierarchy
    try:
        # Convert underscores to spaces for element lookup
        element_name = qual_name.replace("_", " ")
        all_ancestors = toolkit.get_ancestors(element_name)
        if all_ancestors:
            ancestors = [a.replace(" ", "_") for a in all_ancestors]
    except Exception:
        pass

    return ancestors


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hierarchy Inference Utilities")
    print("=" * 60)

    # Test type ancestors
    print("\nTest: get_type_ancestors()")
    print("-" * 60)

    test_type = "SmallMolecule"
    ancestors = get_type_ancestors(test_type)
    print(f"{test_type} ancestors: {sorted(ancestors)[:5]}... ({len(ancestors)} total)")

    test_pseudo = "Gene+SmallMolecule"
    ancestors_pseudo = get_type_ancestors(test_pseudo)
    print(f"{test_pseudo} ancestors: {sorted(ancestors_pseudo)[:5]}... ({len(ancestors_pseudo)} total)")

    # Test predicate ancestors (should return without biolink: prefix)
    print("\nTest: get_predicate_ancestors()")
    print("-" * 60)

    test_preds = ["treats", "affects", "regulates", "biolink:treats"]
    for pred in test_preds:
        ancestors = get_predicate_ancestors(pred)
        print(f"{pred} → {ancestors}")
        # Verify no biolink: prefix in output
        for anc in ancestors:
            assert not anc.startswith("biolink:"), f"Found biolink: prefix in ancestor: {anc}"
    print("✓ All ancestors returned without biolink: prefix")

    # Test qualifier ancestors
    print("\nTest: get_qualifier_ancestors()")
    print("-" * 60)

    # Note: This may not return anything if no qualifier hierarchies exist
    test_quals = ["increased", "decreased"]
    for qual in test_quals:
        ancestors = get_qualifier_ancestors(qual)
        print(f"{qual} → {ancestors}")
