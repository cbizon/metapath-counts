#!/usr/bin/env python3
"""Tests for the unified index module."""

import json
import os
import tempfile

import graphblas as gb
import numpy as np

from library.unified_index import (
    build_target_pair_set,
    build_unified_type_offsets,
    load_base_matrices,
    reconstruct_nhop_matrix,
    reconstruct_prefix_matrix,
    remap_matrix_to_unified,
    remap_nhop_to_unified,
)


def _make_bool_matrix(rows, cols, nrows, ncols):
    """Helper to build a small GraphBLAS boolean matrix from COO lists."""
    if len(rows) == 0:
        return gb.Matrix(gb.dtypes.BOOL, nrows=nrows, ncols=ncols)
    return gb.Matrix.from_coo(
        np.array(rows, dtype=np.uint64),
        np.array(cols, dtype=np.uint64),
        np.ones(len(rows), dtype=bool),
        nrows=nrows,
        ncols=ncols,
        dtype=gb.dtypes.BOOL,
        dup_op=gb.binary.any,
    )


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _small_manifest():
    """A small manifest with a few types and matrices."""
    return [
        {"src_type": "SmallMolecule", "predicate": "treats", "tgt_type": "Disease",
         "nrows": 10, "ncols": 5, "filename": "SmallMolecule__treats__Disease.npz"},
        {"src_type": "Drug", "predicate": "treats", "tgt_type": "Disease",
         "nrows": 3, "ncols": 5, "filename": "Drug__treats__Disease.npz"},
        {"src_type": "SmallMolecule", "predicate": "affects", "tgt_type": "Gene",
         "nrows": 10, "ncols": 8, "filename": "SmallMolecule__affects__Gene.npz"},
        {"src_type": "Gene", "predicate": "has_phenotype", "tgt_type": "Disease",
         "nrows": 8, "ncols": 5, "filename": "Gene__has_phenotype__Disease.npz"},
    ]


def _small_base_matrices():
    """Build base matrices matching _small_manifest."""
    return {
        ("SmallMolecule", "treats", "Disease"): _make_bool_matrix(
            [0, 1, 2], [0, 1, 2], nrows=10, ncols=5,
        ),
        ("Drug", "treats", "Disease"): _make_bool_matrix(
            [0, 1], [0, 1], nrows=3, ncols=5,
        ),
        ("SmallMolecule", "affects", "Gene"): _make_bool_matrix(
            [0, 1, 2, 3], [0, 1, 2, 3], nrows=10, ncols=8,
        ),
        ("Gene", "has_phenotype", "Disease"): _make_bool_matrix(
            [0, 1, 2], [0, 1, 2], nrows=8, ncols=5,
        ),
    }


# ---------------------------------------------------------------------------
# Tests: build_unified_type_offsets
# ---------------------------------------------------------------------------

class TestBuildUnifiedTypeOffsets:
    def test_single_type_is_its_own_ancestor(self):
        manifest = [
            {"src_type": "Gene", "predicate": "affects", "tgt_type": "Disease",
             "nrows": 100, "ncols": 50, "filename": "f.npz"},
        ]
        offsets, total = build_unified_type_offsets("Gene", manifest)
        assert "Gene" in offsets
        assert offsets["Gene"] == 0
        assert total == 100

    def test_ancestor_gathers_multiple_subtypes(self):
        manifest = _small_manifest()
        # ChemicalEntity is an ancestor of both SmallMolecule and Drug
        offsets, total = build_unified_type_offsets("ChemicalEntity", manifest)
        assert "SmallMolecule" in offsets
        assert "Drug" in offsets
        # Subtypes are sorted alphabetically, Drug < SmallMolecule
        assert offsets["Drug"] == 0
        assert offsets["SmallMolecule"] == 3  # Drug has 3 nodes
        assert total == 13  # 3 (Drug) + 10 (SmallMolecule)

    def test_exact_type_includes_self(self):
        manifest = _small_manifest()
        offsets, total = build_unified_type_offsets("Disease", manifest)
        assert "Disease" in offsets
        assert offsets["Disease"] == 0
        assert total == 5

    def test_type_not_in_manifest_returns_empty(self):
        manifest = _small_manifest()
        offsets, total = build_unified_type_offsets("PhenotypicFeature", manifest)
        # PhenotypicFeature is not present in any matrix in the manifest
        assert offsets == {}
        assert total == 0


# ---------------------------------------------------------------------------
# Tests: remap_matrix_to_unified
# ---------------------------------------------------------------------------

class TestRemapMatrixToUnified:
    def test_remap_shifts_indices(self):
        matrix = _make_bool_matrix([0, 1], [0, 1], nrows=3, ncols=5)
        src_offsets = {"Drug": 0, "SmallMolecule": 3}
        tgt_offsets = {"Disease": 0}
        result = remap_matrix_to_unified(
            matrix, "SmallMolecule", "Disease",
            src_offsets, tgt_offsets,
            unified_nrows=13, unified_ncols=5,
        )
        assert result is not None
        assert result.nrows == 13
        assert result.ncols == 5
        assert result.nvals == 2
        rows, cols, _ = result.to_coo()
        # SmallMolecule offset is 3, so rows shift by 3
        assert list(rows) == [3, 4]
        assert list(cols) == [0, 1]

    def test_remap_unknown_type_returns_none(self):
        matrix = _make_bool_matrix([0], [0], nrows=2, ncols=2)
        result = remap_matrix_to_unified(
            matrix, "UnknownType", "Disease",
            {"SmallMolecule": 0}, {"Disease": 0},
            unified_nrows=10, unified_ncols=5,
        )
        assert result is None

    def test_remap_empty_matrix(self):
        matrix = _make_bool_matrix([], [], nrows=3, ncols=5)
        src_offsets = {"Drug": 0}
        tgt_offsets = {"Disease": 0}
        result = remap_matrix_to_unified(
            matrix, "Drug", "Disease",
            src_offsets, tgt_offsets,
            unified_nrows=10, unified_ncols=5,
        )
        assert result is not None
        assert result.nvals == 0


# ---------------------------------------------------------------------------
# Tests: build_target_pair_set
# ---------------------------------------------------------------------------

class TestBuildTargetPairSet:
    def test_exact_type_match(self):
        """Target variant with exact types should find the matching base matrix."""
        manifest = _small_manifest()
        base = _small_base_matrices()
        src_offsets = {"SmallMolecule": 0}
        tgt_offsets = {"Disease": 0}
        result = build_target_pair_set(
            "SmallMolecule|treats|F|Disease",
            base, manifest,
            src_offsets, tgt_offsets,
            unified_nrows=10, unified_ncols=5,
        )
        # Should find the SmallMolecule|treats|Disease matrix with 3 pairs
        assert result.nvals == 3

    def test_ancestor_type_unions_subtypes(self):
        """Ancestor-typed target should union all matching subtype matrices."""
        manifest = _small_manifest()
        base = _small_base_matrices()
        # ChemicalEntity is ancestor of both SmallMolecule and Drug
        src_offsets, src_total = build_unified_type_offsets("ChemicalEntity", manifest)
        tgt_offsets, tgt_total = build_unified_type_offsets("Disease", manifest)
        result = build_target_pair_set(
            "ChemicalEntity|treats|F|Disease",
            base, manifest,
            src_offsets, tgt_offsets,
            unified_nrows=src_total, unified_ncols=tgt_total,
        )
        # SmallMolecule has 3 treats pairs + Drug has 2 treats pairs = 5
        assert result.nvals == 5

    def test_ancestor_predicate_matches(self):
        """Target with ancestor predicate should match base matrices with descendant predicates."""
        manifest = _small_manifest()
        base = _small_base_matrices()
        src_offsets, src_total = build_unified_type_offsets("ChemicalEntity", manifest)
        tgt_offsets, tgt_total = build_unified_type_offsets("Disease", manifest)
        # related_to is an ancestor of treats
        result = build_target_pair_set(
            "ChemicalEntity|related_to|A|Disease",
            base, manifest,
            src_offsets, tgt_offsets,
            unified_nrows=src_total, unified_ncols=tgt_total,
        )
        # Should match treats (descendant of related_to) and has_phenotype if applicable
        # treats: SmallMolecule(3) + Drug(2) = 5 pairs
        # has_phenotype is NOT a descendant of related_to in biolink typically
        # But has_phenotype IS a descendant of related_to in Biolink Model
        # Gene has_phenotype Disease: Gene is not under ChemicalEntity, so not matched
        assert result.nvals >= 5

    def test_no_matching_matrices(self):
        """Target with no matching base matrices should return empty matrix."""
        manifest = _small_manifest()
        base = _small_base_matrices()
        src_offsets = {"PhenotypicFeature": 0}
        tgt_offsets = {"Disease": 0}
        result = build_target_pair_set(
            "PhenotypicFeature|treats|F|Disease",
            base, manifest,
            src_offsets, tgt_offsets,
            unified_nrows=10, unified_ncols=5,
        )
        assert result.nvals == 0

    def test_exact_count_less_than_sum(self):
        """When base matrices share pairs, exact count should be less than sum."""
        # Create two base matrices that share some pairs
        manifest = [
            {"src_type": "SmallMolecule", "predicate": "treats", "tgt_type": "Disease",
             "nrows": 10, "ncols": 5, "filename": "f1.npz"},
            {"src_type": "SmallMolecule", "predicate": "ameliorates", "tgt_type": "Disease",
             "nrows": 10, "ncols": 5, "filename": "f2.npz"},
        ]
        # Both matrices share pairs (0,0) and (1,1)
        base = {
            ("SmallMolecule", "treats", "Disease"): _make_bool_matrix(
                [0, 1, 2], [0, 1, 2], nrows=10, ncols=5,
            ),
            ("SmallMolecule", "ameliorates", "Disease"): _make_bool_matrix(
                [0, 1, 3], [0, 1, 3], nrows=10, ncols=5,
            ),
        }
        src_offsets = {"SmallMolecule": 0}
        tgt_offsets = {"Disease": 0}
        # related_to is ancestor of both treats and ameliorates
        result = build_target_pair_set(
            "SmallMolecule|related_to|A|Disease",
            base, manifest,
            src_offsets, tgt_offsets,
            unified_nrows=10, unified_ncols=5,
        )
        # Sum would be 3 + 3 = 6, but union is 4 (shared pairs 0,0 and 1,1)
        assert result.nvals == 4
        sum_count = 3 + 3
        assert result.nvals < sum_count


# ---------------------------------------------------------------------------
# Tests: reconstruct_nhop_matrix
# ---------------------------------------------------------------------------

class TestReconstructNhopMatrix:
    def test_1hop_forward(self):
        base = _small_base_matrices()
        result = reconstruct_nhop_matrix(
            "SmallMolecule|treats|F|Disease", base,
        )
        assert result is not None
        assert result.nvals == 3

    def test_2hop_forward(self):
        base = _small_base_matrices()
        result = reconstruct_nhop_matrix(
            "SmallMolecule|affects|F|Gene|has_phenotype|F|Disease", base,
        )
        assert result is not None
        # 4 SmallMolecule→Gene pairs, 3 Gene→Disease pairs
        # Pairs: SM0→G0→D0, SM1→G1→D1, SM2→G2→D2 (SM3→G3 but G3 has no phenotype)
        assert result.nvals == 3
        assert result.nrows == 10  # SmallMolecule dimension
        assert result.ncols == 5   # Disease dimension

    def test_missing_matrix_returns_none(self):
        base = _small_base_matrices()
        result = reconstruct_nhop_matrix(
            "SmallMolecule|treats|F|Gene|treats|F|Disease", base,
        )
        # Gene|treats|Disease doesn't exist as a base matrix
        # (Gene, treats, Disease) not in base_matrices
        # So should return None
        assert result is None

    def test_reverse_direction(self):
        base = _small_base_matrices()
        # Disease|treats|R|SmallMolecule means: from Disease, follow treats in reverse
        # The forward edge is SmallMolecule→Disease, stored as (SmallMolecule, treats, Disease)
        # So we look for (Disease, treats, SmallMolecule) which doesn't exist,
        # then try (SmallMolecule, treats, Disease).T
        result = reconstruct_nhop_matrix(
            "Disease|treats|R|SmallMolecule", base,
        )
        assert result is not None
        assert result.nvals == 3
        assert result.nrows == 5   # Disease dimension
        assert result.ncols == 10  # SmallMolecule dimension


# ---------------------------------------------------------------------------
# Tests: remap_nhop_to_unified
# ---------------------------------------------------------------------------

class TestRemapNhopToUnified:
    def test_remap_preserves_pair_count(self):
        base = _small_base_matrices()
        nhop = reconstruct_nhop_matrix(
            "SmallMolecule|affects|F|Gene|has_phenotype|F|Disease", base,
        )
        src_offsets = {"SmallMolecule": 0, "Drug": 10}
        tgt_offsets = {"Disease": 0}
        result = remap_nhop_to_unified(
            nhop, "SmallMolecule|affects|F|Gene|has_phenotype|F|Disease",
            src_offsets, tgt_offsets,
            unified_nrows=13, unified_ncols=5,
        )
        assert result is not None
        assert result.nvals == nhop.nvals


# ---------------------------------------------------------------------------
# Tests: load_base_matrices (with temp dir)
# ---------------------------------------------------------------------------

class TestReconstructPrefixMatrix:
    def test_prefix_of_2hop_is_1hop(self):
        """Prefix of a 2-hop path should be the 1-hop matrix."""
        base = _small_base_matrices()
        prefix = reconstruct_prefix_matrix(
            "SmallMolecule|affects|F|Gene|has_phenotype|F|Disease",
            n_prefix_hops=1,
            base_matrices=base,
        )
        assert prefix is not None
        # Should be the SmallMolecule|affects|Gene matrix (4 pairs)
        assert prefix.nvals == 4
        assert prefix.nrows == 10  # SmallMolecule dimension
        assert prefix.ncols == 8   # Gene dimension

    def test_prefix_equals_full_when_n_prefix_equals_n_hops(self):
        """Prefix with all hops should equal the full reconstruction."""
        base = _small_base_matrices()
        prefix = reconstruct_prefix_matrix(
            "SmallMolecule|affects|F|Gene|has_phenotype|F|Disease",
            n_prefix_hops=2,
            base_matrices=base,
        )
        full = reconstruct_nhop_matrix(
            "SmallMolecule|affects|F|Gene|has_phenotype|F|Disease",
            base,
        )
        assert prefix is not None
        assert full is not None
        assert prefix.nvals == full.nvals
        assert prefix.nrows == full.nrows
        assert prefix.ncols == full.ncols

    def test_prefix_0_returns_none(self):
        """Zero prefix hops should return None."""
        base = _small_base_matrices()
        result = reconstruct_prefix_matrix(
            "SmallMolecule|treats|F|Disease",
            n_prefix_hops=0,
            base_matrices=base,
        )
        assert result is None

    def test_prefix_exceeding_hops_returns_none(self):
        """Prefix hops exceeding path length should return None."""
        base = _small_base_matrices()
        result = reconstruct_prefix_matrix(
            "SmallMolecule|treats|F|Disease",
            n_prefix_hops=5,
            base_matrices=base,
        )
        assert result is None

    def test_missing_matrix_returns_none(self):
        """Missing base matrix in prefix should return None."""
        base = _small_base_matrices()
        result = reconstruct_prefix_matrix(
            "SmallMolecule|treats|F|Gene|has_phenotype|F|Disease",
            n_prefix_hops=1,
            base_matrices=base,
        )
        # SmallMolecule|treats|Gene doesn't exist
        assert result is None


class TestLoadBaseMatrices:
    def test_round_trip(self):
        """Serialize a few matrices and load them back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "num_matrices": 1,
                "matrices": [
                    {
                        "src_type": "Gene",
                        "predicate": "affects",
                        "tgt_type": "Disease",
                        "nrows": 4,
                        "ncols": 3,
                        "nvals": 2,
                        "filename": "Gene__affects__Disease.npz",
                    }
                ],
            }
            with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
                json.dump(manifest, f)

            np.savez_compressed(
                os.path.join(tmpdir, "Gene__affects__Disease.npz"),
                rows=np.array([0, 1], dtype=np.uint64),
                cols=np.array([0, 1], dtype=np.uint64),
                vals=np.array([True, True]),
                nrows=4,
                ncols=3,
                nvals=2,
            )

            loaded_manifest, loaded_matrices = load_base_matrices(tmpdir)
            assert len(loaded_manifest) == 1
            assert ("Gene", "affects", "Disease") in loaded_matrices
            m = loaded_matrices[("Gene", "affects", "Disease")]
            assert m.nrows == 4
            assert m.ncols == 3
            assert m.nvals == 2
