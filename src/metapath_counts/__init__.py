"""
Metapath Counts - Parallel 3-hop metapath analysis for biolink knowledge graphs.

This package provides tools for calculating metapath overlaps in large knowledge graphs
using GraphBLAS sparse matrices and SLURM-based parallel processing.
"""

__version__ = "0.1.0"

from .type_utils import (
    get_most_specific_type,
    get_symmetric_predicates,
    get_all_types,
    filter_abstract_types
)

from .type_assignment import (
    assign_node_type,
    is_pseudo_type,
    parse_pseudo_type,
    format_pseudo_type,
    find_leaf_types
)

from .hierarchy import (
    get_type_ancestors,
    get_predicate_ancestors,
    get_qualifier_ancestors
)

__all__ = [
    "get_most_specific_type",
    "get_symmetric_predicates",
    "get_all_types",
    "filter_abstract_types",
    "assign_node_type",
    "is_pseudo_type",
    "parse_pseudo_type",
    "format_pseudo_type",
    "find_leaf_types",
    "get_type_ancestors",
    "get_predicate_ancestors",
    "get_qualifier_ancestors"
]
