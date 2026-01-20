#!/usr/bin/env python3
"""
Extract symmetric predicates from the Biolink Model.

Uses the biolink-model-toolkit to identify all predicates that are marked
as symmetric in the official Biolink Model specification.
"""

from bmt import Toolkit

def get_symmetric_predicates():
    """
    Get all symmetric predicates from the Biolink Model.

    Returns:
        set: Set of symmetric predicate names (with biolink: prefix)
    """
    tk = Toolkit()
    symmetric_predicates = set()

    # Get all predicate slots
    all_predicates = tk.get_all_slots()

    for predicate_name in all_predicates:
        predicate = tk.get_element(predicate_name)

        # Check if predicate is marked as symmetric
        if predicate and predicate.symmetric:
            # Add with biolink: prefix
            symmetric_predicates.add(f"biolink:{predicate_name}")

    return symmetric_predicates


def main():
    print("Extracting symmetric predicates from Biolink Model...")
    print()

    symmetric_predicates = get_symmetric_predicates()

    print(f"Found {len(symmetric_predicates)} symmetric predicates:")
    print()

    # Print as Python set for easy copy-paste
    print("SYMMETRIC_PREDICATES = {")
    for pred in sorted(symmetric_predicates):
        print(f"    '{pred}',")
    print("}")
    print()

    # Also print without biolink: prefix (for use in metapath matching)
    print("# Without biolink: prefix (for metapath matching):")
    print("SYMMETRIC_PREDICATES = {")
    for pred in sorted(symmetric_predicates):
        pred_without_prefix = pred.replace('biolink:', '')
        print(f"    '{pred_without_prefix}',")
    print("}")


if __name__ == '__main__':
    main()
