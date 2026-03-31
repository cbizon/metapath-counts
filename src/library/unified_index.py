"""
Unified index utilities for exact pair-set tracking.

Provides functions to remap per-type GraphBLAS matrices into a unified
ancestor-type coordinate space so that pair sets from different explicit
types can be unioned or intersected without node ID lookups.

Key insight: each node has exactly one explicit type (single-type assignment),
and per-type indices are dense and 0-based.  A unified index for an ancestor
type is built by concatenating subtypes' index ranges using only the counts
from the matrices manifest and the type hierarchy.
"""

import json
from functools import lru_cache
from pathlib import Path

import graphblas as gb
import numpy as np

from .aggregation import (
    get_predicate_variants,
    normalize_predicate,
    parse_compound_predicate,
    parse_metapath,
)
from .hierarchy import get_type_ancestors
from .type_utils import get_symmetric_predicates


def _collect_type_counts(manifest_matrices):
    """Extract per-type node counts from the matrices manifest.

    Each explicit type appears as src or tgt in various matrices.
    nrows corresponds to src_type count, ncols to tgt_type count.
    """
    type_counts = {}
    for m in manifest_matrices:
        src = m["src_type"]
        tgt = m["tgt_type"]
        type_counts[src] = max(type_counts.get(src, 0), m["nrows"])
        type_counts[tgt] = max(type_counts.get(tgt, 0), m["ncols"])
    return type_counts


def build_unified_type_offsets(ancestor_type, manifest_matrices):
    """Compute offset table for remapping per-type indices to unified ancestor-type space.

    Args:
        ancestor_type: The ancestor type to build unified index for (e.g. "ChemicalEntity")
        manifest_matrices: List of matrix metadata dicts from manifest.json,
            each with keys: src_type, tgt_type, nrows, ncols, predicate, filename

    Returns:
        (offsets, total_size) where offsets is {explicit_type: offset} and
        total_size is the total number of nodes in the unified index.
    """
    type_counts = _collect_type_counts(manifest_matrices)

    # Find all explicit types that are descendants of (or equal to) ancestor_type.
    # get_type_ancestors(t) includes t itself, so ancestor_type in get_type_ancestors(t)
    # means t == ancestor_type or t is a descendant.
    subtypes = sorted(
        t for t in type_counts
        if ancestor_type in get_type_ancestors(t)
    )

    offsets = {}
    running_offset = 0
    for t in subtypes:
        offsets[t] = running_offset
        running_offset += type_counts[t]

    return offsets, running_offset


def remap_matrix_to_unified(matrix, src_type, tgt_type,
                            src_offsets, tgt_offsets,
                            unified_nrows, unified_ncols):
    """Remap a matrix from per-type coordinates to unified ancestor-type coordinates.

    Args:
        matrix: GraphBLAS boolean matrix in per-type coordinates
        src_type: Explicit source type name
        tgt_type: Explicit target type name
        src_offsets: {type: offset} for the source ancestor type
        tgt_offsets: {type: offset} for the target ancestor type
        unified_nrows: Total unified dimension for source ancestor type
        unified_ncols: Total unified dimension for target ancestor type

    Returns:
        GraphBLAS boolean matrix in unified coordinates, or None if
        src_type or tgt_type is not in the offsets.
    """
    if src_type not in src_offsets or tgt_type not in tgt_offsets:
        return None

    if matrix.nvals == 0:
        return gb.Matrix(gb.dtypes.BOOL, nrows=unified_nrows, ncols=unified_ncols)

    rows, cols, vals = matrix.to_coo()
    rows = rows + src_offsets[src_type]
    cols = cols + tgt_offsets[tgt_type]

    return gb.Matrix.from_coo(
        rows, cols, vals,
        nrows=unified_nrows,
        ncols=unified_ncols,
        dtype=gb.dtypes.BOOL,
        dup_op=gb.binary.any,
    )


def load_base_matrices(matrices_dir):
    """Load the manifest and all base matrices from the matrices directory.

    Args:
        matrices_dir: Path to the matrices directory (containing manifest.json
            and .npz files).

    Returns:
        (manifest_matrices, base_matrices) where manifest_matrices is the list
        of matrix metadata dicts and base_matrices is
        {(src_type, predicate, tgt_type): GraphBLAS matrix}.
    """
    matrices_path = Path(matrices_dir)
    manifest_path = matrices_path / "manifest.json"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    manifest_matrices = manifest["matrices"]
    base_matrices = {}

    for mat_info in manifest_matrices:
        src_type = mat_info["src_type"]
        pred = mat_info["predicate"]
        tgt_type = mat_info["tgt_type"]
        filename = mat_info["filename"]

        npz_path = matrices_path / filename
        data = np.load(npz_path)

        matrix = gb.Matrix.from_coo(
            data["rows"],
            data["cols"],
            data["vals"],
            nrows=int(data["nrows"]),
            ncols=int(data["ncols"]),
            dtype=gb.dtypes.BOOL,
            dup_op=gb.binary.any,
        )
        base_matrices[(src_type, normalize_predicate(pred), tgt_type)] = matrix

    return manifest_matrices, base_matrices


@lru_cache(maxsize=None)
def _predicate_variant_set(predicate):
    """Return the set of all predicate variants (self + ancestors) for caching."""
    return set(get_predicate_variants(predicate))


def build_target_pair_set(target_variant, base_matrices, manifest_matrices,
                          src_offsets, tgt_offsets,
                          unified_nrows, unified_ncols):
    """Union all base matrices that roll up to a target variant in unified space.

    For a target variant like "ChemicalEntity|related_to|A|Disease", finds all
    base matrices whose (src_type, predicate, tgt_type) is a specialization of
    the target variant's types and predicate, then unions them in unified
    coordinate space.

    Args:
        target_variant: 1-hop metapath like "ChemicalEntity|related_to|A|Disease"
        base_matrices: {(src_type, predicate, tgt_type): GraphBLAS matrix}
        manifest_matrices: List of matrix metadata dicts from manifest.json
        src_offsets: {explicit_type: offset} for the source ancestor type
        tgt_offsets: {explicit_type: offset} for the target ancestor type
        unified_nrows: Total unified rows
        unified_ncols: Total unified cols

    Returns:
        GraphBLAS boolean matrix representing the union of all matching
        base matrices in unified coordinates.
    """
    nodes, predicates, directions = parse_metapath(target_variant)
    assert len(predicates) == 1, f"target_variant must be 1-hop, got: {target_variant}"

    ancestor_src = nodes[0]
    ancestor_tgt = nodes[1]
    target_pred = normalize_predicate(predicates[0])
    direction = directions[0]

    union_matrix = gb.Matrix(gb.dtypes.BOOL, nrows=unified_nrows, ncols=unified_ncols)

    for (bm_src, bm_pred, bm_tgt), matrix in base_matrices.items():
        # Check predicate match: target_pred must be in the predicate variants
        # of bm_pred (meaning bm_pred is equal to or more specific than target_pred).
        if target_pred not in _predicate_variant_set(bm_pred):
            continue

        bm_src_ancestors = get_type_ancestors(bm_src)
        bm_tgt_ancestors = get_type_ancestors(bm_tgt)

        # Forward match: bm_src under ancestor_src, bm_tgt under ancestor_tgt.
        # Contributes pairs (a, b) where a is bm_src-type, b is bm_tgt-type.
        if direction in ('F', 'A'):
            if ancestor_src in bm_src_ancestors and ancestor_tgt in bm_tgt_ancestors:
                remapped = remap_matrix_to_unified(
                    matrix, bm_src, bm_tgt,
                    src_offsets, tgt_offsets,
                    unified_nrows, unified_ncols,
                )
                if remapped is not None and remapped.nvals > 0:
                    union_matrix << union_matrix.ewise_add(remapped, gb.binary.any)

        # Reverse match: bm_tgt under ancestor_src, bm_src under ancestor_tgt.
        # The forward edge goes from bm_src to bm_tgt, but viewed in reverse
        # the pair is (bm_tgt, bm_src) in (ancestor_src, ancestor_tgt) space.
        # We transpose the matrix to get (bm_tgt, bm_src) shape, then remap
        # with bm_tgt in src_offsets and bm_src in tgt_offsets.
        if direction in ('R', 'A'):
            if ancestor_src in bm_tgt_ancestors and ancestor_tgt in bm_src_ancestors:
                rows, cols, vals = matrix.to_coo()
                if len(rows) > 0:
                    # Transpose: swap rows and cols, then remap
                    transposed_rows = cols + src_offsets.get(bm_tgt, 0)
                    transposed_cols = rows + tgt_offsets.get(bm_src, 0)
                    if bm_tgt in src_offsets and bm_src in tgt_offsets:
                        remapped = gb.Matrix.from_coo(
                            transposed_rows, transposed_cols, vals,
                            nrows=unified_nrows,
                            ncols=unified_ncols,
                            dtype=gb.dtypes.BOOL,
                            dup_op=gb.binary.any,
                        )
                        union_matrix << union_matrix.ewise_add(remapped, gb.binary.any)

    return union_matrix


def _lookup_hop_matrix(src, pred, tgt, direction, base_matrices):
    """Look up the base matrix for a single hop, respecting direction.

    For direction F or A, the edge goes src→tgt, so we prefer (src, pred, tgt).
    For direction R, the edge goes tgt→src (reverse traversal), so we prefer
    (tgt, pred, src).T to get a matrix with rows=src, cols=tgt.

    Falls back to the other key for symmetric predicates where only one
    direction may be stored.

    Returns:
        GraphBLAS matrix with rows=src, cols=tgt, or None if not found.
    """
    pred = normalize_predicate(pred)
    if direction == 'R':
        # Reverse: real edge is tgt→src, need (tgt, pred, src).T
        reverse_key = (tgt, pred, src)
        if reverse_key in base_matrices:
            return base_matrices[reverse_key].T
        # Fallback for symmetric predicates stored in canonical direction only
        key = (src, pred, tgt)
        if key in base_matrices:
            return base_matrices[key]
        return None
    else:
        # Forward (F) or Any (A): real edge is src→tgt
        key = (src, pred, tgt)
        if key in base_matrices:
            return base_matrices[key]
        # Fallback for symmetric predicates stored in canonical direction only
        reverse_key = (tgt, pred, src)
        if reverse_key in base_matrices:
            return base_matrices[reverse_key].T
        return None


def reconstruct_nhop_matrix(nhop_metapath, base_matrices):
    """Multiply base matrices along an N-hop path to reconstruct the N-hop pair set.

    For each hop in the metapath, looks up the corresponding base matrix and
    chains them via boolean matrix multiplication (any_pair semiring).

    Args:
        nhop_metapath: N-hop metapath string like
            "SmallMolecule|treats|F|Gene|has_phenotype|F|Disease"
        base_matrices: {(src_type, predicate, tgt_type): GraphBLAS matrix}

    Returns:
        GraphBLAS boolean matrix representing the N-hop pair set,
        or None if any hop's base matrix is missing.
    """
    nodes, predicates, directions = parse_metapath(nhop_metapath)

    accumulated = None

    for i in range(len(predicates)):
        hop_matrix = _lookup_hop_matrix(
            nodes[i], predicates[i], nodes[i + 1], directions[i], base_matrices,
        )
        if hop_matrix is None:
            return None

        if accumulated is None:
            accumulated = hop_matrix.dup(dtype=gb.dtypes.BOOL)
        else:
            accumulated = accumulated.mxm(hop_matrix, gb.semiring.any_pair).new()

    return accumulated


def reconstruct_prefix_matrix(nhop_metapath, n_prefix_hops, base_matrices):
    """Reconstruct only the first n_prefix_hops of an N-hop metapath.

    For a 3-hop path like "A|p1|F|B|p2|F|C|p3|F|D" with n_prefix_hops=2,
    this reconstructs the matrix for "A|p1|F|B|p2|F|C".

    Args:
        nhop_metapath: Full N-hop metapath string.
        n_prefix_hops: Number of hops to include in the prefix.
        base_matrices: {(src_type, predicate, tgt_type): GraphBLAS matrix}

    Returns:
        GraphBLAS boolean matrix for the prefix, or None if any hop's
        base matrix is missing.  Returns None if n_prefix_hops < 1.
    """
    if n_prefix_hops < 1:
        return None

    nodes, predicates, directions = parse_metapath(nhop_metapath)

    if n_prefix_hops > len(predicates):
        return None

    accumulated = None

    for i in range(n_prefix_hops):
        hop_matrix = _lookup_hop_matrix(
            nodes[i], predicates[i], nodes[i + 1], directions[i], base_matrices,
        )
        if hop_matrix is None:
            return None

        if accumulated is None:
            accumulated = hop_matrix.dup(dtype=gb.dtypes.BOOL)
        else:
            accumulated = accumulated.mxm(hop_matrix, gb.semiring.any_pair).new()

    return accumulated


def remap_nhop_to_unified(nhop_matrix, nhop_metapath,
                          src_offsets, tgt_offsets,
                          unified_nrows, unified_ncols):
    """Remap an N-hop result matrix to unified ancestor-type coordinates.

    Args:
        nhop_matrix: GraphBLAS boolean matrix in per-type coordinates
        nhop_metapath: The N-hop metapath string (to extract endpoint types)
        src_offsets: {explicit_type: offset} for the source ancestor type
        tgt_offsets: {explicit_type: offset} for the target ancestor type
        unified_nrows: Total unified rows
        unified_ncols: Total unified cols

    Returns:
        GraphBLAS boolean matrix in unified coordinates, or None if
        endpoints are not in the offsets.
    """
    nodes, _, _ = parse_metapath(nhop_metapath)
    src_type = nodes[0]
    tgt_type = nodes[-1]

    return remap_matrix_to_unified(
        nhop_matrix, src_type, tgt_type,
        src_offsets, tgt_offsets,
        unified_nrows, unified_ncols,
    )
