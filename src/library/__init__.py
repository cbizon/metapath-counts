"""
Metapath Counts - Parallel 3-hop metapath analysis for biolink knowledge graphs.

This package provides tools for calculating metapath overlaps in large knowledge graphs
using GraphBLAS sparse matrices and SLURM-based parallel processing.
"""

__version__ = "0.1.0"

from .type_utils import (
    get_most_specific_type,
    get_symmetric_predicates
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

from .aggregation import (
    parse_metapath,
    build_metapath,
    get_type_variants,
    get_predicate_variants,
    generate_metapath_variants,
    expand_metapath_to_variants,
    calculate_metrics,
    build_compound_predicate,
    parse_compound_predicate,
    is_compound_predicate
)

__all__ = [
    "get_most_specific_type",
    "get_symmetric_predicates",
    "assign_node_type",
    "is_pseudo_type",
    "parse_pseudo_type",
    "format_pseudo_type",
    "find_leaf_types",
    "get_type_ancestors",
    "get_predicate_ancestors",
    "get_qualifier_ancestors",
    "parse_metapath",
    "build_metapath",
    "get_type_variants",
    "get_predicate_variants",
    "generate_metapath_variants",
    "expand_metapath_to_variants",
    "calculate_metrics",
    "build_compound_predicate",
    "parse_compound_predicate",
    "is_compound_predicate"
]
