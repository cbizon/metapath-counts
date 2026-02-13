#!/usr/bin/env python3
"""
Hierarchical aggregation utilities for metapath analysis.

This module provides functions for:
- Parsing and building metapath strings
- Expanding metapaths to hierarchical variants (types and predicates)
- Generating all implied variants through the biolink hierarchy

These functions are used by multiple scripts (analyze_hop_overlap.py,
group_by_onehop.py, group_single_onehop_worker.py) and are centralized
here to avoid duplication.
"""

import itertools
from typing import Iterator, List, Tuple

from .hierarchy import get_type_ancestors, get_predicate_ancestors, get_qualifier_ancestors
from .type_assignment import is_pseudo_type, parse_pseudo_type
from .type_utils import get_symmetric_predicates

# Cache symmetric predicates at module load time
_SYMMETRIC_PREDICATES = None


def _get_symmetric_predicates():
    """Get cached symmetric predicates (lazy initialization)."""
    global _SYMMETRIC_PREDICATES
    if _SYMMETRIC_PREDICATES is None:
        _SYMMETRIC_PREDICATES = get_symmetric_predicates()
    return _SYMMETRIC_PREDICATES


def build_compound_predicate(predicate: str, direction: str = None, aspect: str = None) -> str:
    """
    Build a compound predicate string encoding qualifier information.

    Format: ``base--direction--aspect`` when qualifiers are present,
    or plain ``base`` when neither qualifier is given.

    Args:
        predicate: Base predicate name (e.g., "causes")
        direction: Direction qualifier (e.g., "decreased") or None
        aspect: Aspect qualifier (e.g., "activity_or_abundance") or None

    Returns:
        Compound predicate string, or plain predicate if no qualifiers

    Examples:
        >>> build_compound_predicate("causes", "decreased", "activity_or_abundance")
        'causes--decreased--activity_or_abundance'
        >>> build_compound_predicate("causes", "decreased", None)
        'causes--decreased--'
        >>> build_compound_predicate("treats")
        'treats'
    """
    if not direction and not aspect:
        return predicate
    return f"{predicate}--{direction or ''}--{aspect or ''}"


def parse_compound_predicate(compound: str) -> Tuple[str, str, str]:
    """
    Parse a compound predicate into its components.

    Args:
        compound: Predicate string (plain or compound)

    Returns:
        Tuple of (base_predicate, direction_qualifier, aspect_qualifier)
        where direction and aspect are None when absent

    Examples:
        >>> parse_compound_predicate("causes--decreased--activity_or_abundance")
        ('causes', 'decreased', 'activity_or_abundance')
        >>> parse_compound_predicate("causes--decreased--")
        ('causes', 'decreased', None)
        >>> parse_compound_predicate("treats")
        ('treats', None, None)
    """
    if '--' not in compound:
        return (compound, None, None)
    parts = compound.split('--', 2)
    base = parts[0]
    direction = parts[1] if parts[1] else None
    aspect = parts[2] if len(parts) > 2 and parts[2] else None
    return (base, direction, aspect)


def is_compound_predicate(predicate: str) -> bool:
    """
    Check if a predicate string encodes qualifier information.

    Args:
        predicate: Predicate string

    Returns:
        True if the predicate contains '--' separator (is compound)
    """
    return '--' in predicate


def parse_metapath(metapath: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse a metapath into node types, predicates, and directions.

    Metapath format: Type1|pred1|dir1|Type2|pred2|dir2|...|TypeN
    - N-hop path has N+1 node types, N predicates, N directions
    - Number of parts = 1 + 3*N (for N hops)

    Examples:
        1-hop: "A|pred1|F|B" -> nodes=['A', 'B'], predicates=['pred1'], directions=['F']
        2-hop: "A|p1|F|B|p2|R|C" -> nodes=['A', 'B', 'C'], predicates=['p1', 'p2'], directions=['F', 'R']
        3-hop: "A|p1|F|B|p2|R|C|p3|F|D" -> nodes=['A', 'B', 'C', 'D'], ...

    Args:
        metapath: Pipe-separated metapath string

    Returns:
        Tuple of (nodes, predicates, directions)

    Raises:
        ValueError: If metapath format is invalid
    """
    parts = metapath.split('|')
    num_parts = len(parts)

    # Formula: num_parts = 1 + 3*n_hops
    # So: n_hops = (num_parts - 1) / 3
    if (num_parts - 1) % 3 != 0 or num_parts < 4:
        raise ValueError(f"Invalid metapath format: {metapath} (parts: {num_parts}, expected 1+3*N)")

    n_hops = (num_parts - 1) // 3

    # Extract nodes: positions 0, 3, 6, 9, ... (every 3rd starting at 0)
    nodes = [parts[i * 3] for i in range(n_hops + 1)]

    # Extract predicates: positions 1, 4, 7, 10, ... (every 3rd starting at 1)
    predicates = [parts[i * 3 + 1] for i in range(n_hops)]

    # Extract directions: positions 2, 5, 8, 11, ... (every 3rd starting at 2)
    directions = [parts[i * 3 + 2] for i in range(n_hops)]

    return nodes, predicates, directions


def build_metapath(nodes: List[str], predicates: List[str], directions: List[str]) -> str:
    """
    Build metapath string from components.

    Args:
        nodes: List of node types
        predicates: List of predicate names
        directions: List of directions (F, R, or A)

    Returns:
        Pipe-separated metapath string
    """
    result = []
    for i in range(len(predicates)):
        result.append(nodes[i])
        result.append(predicates[i])
        result.append(directions[i])
    result.append(nodes[-1])
    return '|'.join(result)


def canonicalize_metapath(nodes: List[str], predicates: List[str], directions: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Ensure metapath is in canonical form based on alphabetical ordering of endpoint types.

    Canonical form: first_type <= last_type (alphabetically).
    If not in canonical form, reverses the path and flips directions.

    Args:
        nodes: List of node types
        predicates: List of predicate names
        directions: List of directions (F, R, or A)

    Returns:
        Tuple of (nodes, predicates, directions) in canonical form
    """
    first_type = nodes[0]
    last_type = nodes[-1]

    if first_type <= last_type:
        # Already canonical
        return nodes, predicates, directions

    # Not canonical - reverse the path
    reversed_nodes = list(reversed(nodes))
    reversed_predicates = list(reversed(predicates))

    # Flip directions: F <-> R, A stays A
    flipped_directions = []
    for d in reversed(directions):
        if d == 'F':
            flipped_directions.append('R')
        elif d == 'R':
            flipped_directions.append('F')
        else:  # 'A' for symmetric predicates
            flipped_directions.append('A')

    return reversed_nodes, reversed_predicates, flipped_directions


def get_type_variants(type_name: str, include_self: bool = True) -> List[str]:
    """
    Get all type variants for hierarchical aggregation.

    For pseudo-types: expands to constituents
    For all types: includes ancestors

    Args:
        type_name: Type name (e.g., "SmallMolecule" or "Gene+SmallMolecule")
        include_self: Whether to include the type itself in the results

    Returns:
        List of type variants (includes pseudo-type itself, constituents, and all ancestors)
    """
    variants = []

    # Always include the original type if requested
    if include_self:
        variants.append(type_name)

    if is_pseudo_type(type_name):
        # Pseudo-type: add each constituent
        constituents = parse_pseudo_type(type_name)
        variants.extend(constituents)

    # Get ancestors (works for both regular types and pseudo-types)
    ancestors = get_type_ancestors(type_name)

    # Add ancestors (excluding the type itself if it's in there)
    for ancestor in ancestors:
        if ancestor != type_name and ancestor not in variants:
            variants.append(ancestor)

    return variants


def get_predicate_variants(predicate: str, include_self: bool = True) -> List[str]:
    """
    Get all predicate variants for hierarchical aggregation.

    For plain predicates: returns self + ancestor predicates.

    For compound predicates (``base--direction--aspect``): returns a cross-product
    of base predicate ancestors × direction qualifier ancestors (+ None) × aspect
    qualifier ancestors (+ None).  Each combination is built with
    ``build_compound_predicate``; when both qualifiers are None the result
    collapses to a plain predicate.

    Args:
        predicate: Predicate name (plain or compound, with or without biolink: prefix)
        include_self: Whether to include the predicate itself

    Returns:
        List of predicate variants (all without biolink: prefix)
    """
    if not is_compound_predicate(predicate):
        # Plain predicate path (original logic)
        variants = []
        clean_predicate = predicate.replace('biolink:', '')
        if include_self:
            variants.append(clean_predicate)
        ancestors = get_predicate_ancestors(predicate)
        for ancestor in ancestors:
            if ancestor not in variants:
                variants.append(ancestor)
        return variants

    # Compound predicate path
    base, direction, aspect = parse_compound_predicate(predicate)

    # Base predicate variants: self + ancestors
    base_variants = [base] + [a for a in get_predicate_ancestors(base) if a != base]

    # Direction qualifier variants: self + ancestors + None (drop qualifier)
    if direction:
        dir_ancestors = get_qualifier_ancestors(direction)
        dir_variants = list(dict.fromkeys(dir_ancestors))  # preserve order, deduplicate
        dir_variants.append(None)
    else:
        dir_variants = [None]

    # Aspect qualifier variants: self + ancestors + None (drop qualifier)
    if aspect:
        asp_ancestors = get_qualifier_ancestors(aspect)
        asp_variants = list(dict.fromkeys(asp_ancestors))
        asp_variants.append(None)
    else:
        asp_variants = [None]

    seen = set()
    variants = []
    for b in base_variants:
        for d in dir_variants:
            for a in asp_variants:
                v = build_compound_predicate(b, d, a)
                if include_self or v != predicate:
                    if v not in seen:
                        seen.add(v)
                        variants.append(v)

    return variants


def generate_metapath_variants(metapath: str) -> Iterator[str]:
    """
    Generate all implied variants of a metapath through hierarchical aggregation.

    This expands pseudo-types to constituents and propagates to ancestor
    types and predicates.

    IMPORTANT: When a directional predicate (F/R) is expanded to a symmetric
    ancestor predicate (like related_to), the direction is changed to 'A'.

    Args:
        metapath: Original metapath (may contain pseudo-types)

    Yields:
        All implied metapath variants

    Example:
        Input: "Gene+Protein|affects|F|SmallMolecule"
        Yields:
            - "Gene+Protein|affects|F|SmallMolecule" (original)
            - "Gene|affects|F|SmallMolecule" (expand pseudo-type)
            - "Protein|affects|F|SmallMolecule" (expand pseudo-type)
            - "GeneOrGeneProduct|affects|F|SmallMolecule" (ancestor)
            - "Gene|affects|F|ChemicalEntity" (ancestor)
            - "Gene|related_to|A|SmallMolecule" (symmetric ancestor - direction changed to A!)
            - ... (all combinations)
    """
    nodes, predicates, directions = parse_metapath(metapath)

    # Get all variants for each node type
    node_variants = [get_type_variants(node) for node in nodes]

    # Get all variants for each predicate
    predicate_variants = [get_predicate_variants(pred) for pred in predicates]

    # Get symmetric predicates for direction adjustment
    symmetric_preds = _get_symmetric_predicates()

    # Generate all combinations
    for node_combo in itertools.product(*node_variants):
        for pred_combo in itertools.product(*predicate_variants):
            # Adjust directions: if predicate is symmetric, direction must be 'A'
            adjusted_directions = []
            skip_variant = False

            for i, pred in enumerate(pred_combo):
                # For compound predicates, symmetry is determined by the base predicate
                base_pred = parse_compound_predicate(pred)[0]
                if base_pred in symmetric_preds:
                    adjusted_directions.append('A')

                    # BUGFIX: For same-type ORIGINAL paths, avoid double-counting when expanding
                    # non-symmetric predicates (F/R) to symmetric ancestors (A).
                    # Only generate variants from the canonical direction (F, not R).
                    # Check if ORIGINAL path (nodes, not node_combo) has same src/tgt.
                    if nodes[0] == nodes[-1]:  # Original path has same src and tgt type
                        if directions[i] == 'R':  # Original was reverse direction
                            # Skip this variant - the F version will generate it
                            skip_variant = True
                            break
                else:
                    adjusted_directions.append(directions[i])

            if skip_variant:
                continue

            # Canonicalize: ensure first_type <= last_type alphabetically
            canon_nodes, canon_preds, canon_dirs = canonicalize_metapath(
                list(node_combo), list(pred_combo), adjusted_directions
            )

            # Build the variant metapath
            variant = build_metapath(canon_nodes, canon_preds, canon_dirs)
            yield variant

            # Fix: When different-type endpoints expand to same-type endpoints,
            # the direction was locked by the original alphabetical ordering.
            # We need to also yield the reversed direction variant.
            if (nodes[0] != nodes[-1]                      # original had different types
                    and canon_nodes[0] == canon_nodes[-1]   # variant has same types
                    and any(d != 'A' for d in canon_dirs)): # not all-symmetric (would be identical)
                rev_nodes = list(reversed(canon_nodes))
                rev_preds = list(reversed(canon_preds))
                rev_dirs = ['R' if d == 'F' else 'F' if d == 'R' else 'A'
                            for d in reversed(canon_dirs)]
                yield build_metapath(rev_nodes, rev_preds, rev_dirs)


def expand_metapath_to_variants(metapath: str) -> set:
    """
    Expand a metapath to all hierarchical variants.

    This is a convenience wrapper around generate_metapath_variants that
    returns a set instead of an iterator.

    Args:
        metapath: Pipe-separated metapath like "Gene|affects|F|Disease"

    Returns:
        Set of expanded metapaths including hierarchical ancestors
    """
    return set(generate_metapath_variants(metapath))


def calculate_metrics(
    nhop_count: int,
    onehop_count: int,
    overlap: int,
    total_possible: int,
    full_metrics: bool = False
) -> dict:
    """
    Calculate prediction metrics from N-hop and 1-hop metapath overlap.

    The N-hop path is treated as a predictor for the 1-hop path.

    Args:
        nhop_count: Number of N-hop paths (predictor)
        onehop_count: Number of 1-hop paths (target)
        overlap: Number of node pairs with both N-hop and 1-hop paths
        total_possible: Total possible node pairs
        full_metrics: If True, return all 18 metrics; if False, return basic 6

    Confusion matrix:
    - TP (True Positives): overlap
    - FP (False Positives): nhop_count - overlap
    - FN (False Negatives): onehop_count - overlap
    - TN (True Negatives): total_possible - nhop_count - onehop_count + overlap

    Note: TN can be negative when aggregated counts exceed total_possible.
    This indicates aggregation error (see CLAUDE.md "Approximate Metrics").

    Returns:
        dict with calculated metrics (6 basic or 18 full)
    """
    import math

    # Confusion matrix
    tp = overlap
    fp = nhop_count - overlap
    fn = onehop_count - overlap
    tn = total_possible - nhop_count - onehop_count + overlap
    total = tp + fp + fn + tn

    # Basic metrics with division by zero protection
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Specificity: TN / (TN + FP) - handle negative TN from aggregation errors
    specificity = tn / (tn + fp) if (tn + fp) > 0 and tn >= 0 else 0.0

    # NPV (Negative Predictive Value): TN / (TN + FN)
    npv = tn / (tn + fn) if (tn + fn) > 0 and tn >= 0 else 0.0

    # Matthews Correlation Coefficient (MCC)
    denom_product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_product > 0:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denom_product)
    else:
        mcc = 0.0

    if not full_metrics:
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'specificity': specificity,
            'npv': npv
        }

    # Full metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = recall  # True Positive Rate = Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    balanced_accuracy = (tpr + specificity) / 2.0

    # Likelihood Ratios
    plr = tpr / fpr if fpr > 0 else float('inf')  # Positive Likelihood Ratio
    nlr = fnr / specificity if specificity > 0 else float('inf')  # Negative Likelihood Ratio

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Total': total,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'NPV': npv,
        'Accuracy': accuracy,
        'Balanced_Accuracy': balanced_accuracy,
        'F1': f1,
        'MCC': mcc,
        'TPR': tpr,
        'FPR': fpr,
        'FNR': fnr,
        'PLR': plr,
        'NLR': nlr
    }
