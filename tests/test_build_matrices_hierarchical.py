"""
Unit tests for build_matrices() with hierarchical types.

Tests that matrices are correctly built with multiple type combinations per edge.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from analyze_hop_overlap import build_matrices


class TestBuildMatrices:
    """Test build_matrices() function with hierarchical types."""

    def create_test_edges_file(self, edges_data):
        """Helper to create a temporary edges file."""
        tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        for edge in edges_data:
            tmpfile.write(json.dumps(edge) + '\n')
        tmpfile.flush()
        tmpfile.close()
        return tmpfile.name

    def test_basic_matrix_creation(self):
        """Test basic matrix creation with single type per node."""
        node_types = {
            "NODE:1": ["Gene"],
            "NODE:2": ["Gene"],
            "NODE:3": ["Disease"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:associated_with", "object": "NODE:3"},
            {"subject": "NODE:2", "predicate": "biolink:associated_with", "object": "NODE:3"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Should create one matrix for Gene-associated_with-Disease
            assert len(matrices) >= 1
            assert ("Gene", "associated_with", "Disease") in matrices

            # Check matrix dimensions
            matrix = matrices[("Gene", "associated_with", "Disease")]
            assert matrix.nrows == 2  # Two genes (NODE:1, NODE:2)
            assert matrix.ncols == 1  # One disease (NODE:3)
            assert matrix.nvals == 2  # Two edges
        finally:
            Path(edges_file).unlink()

    def test_multiple_types_per_node(self):
        """Test that edges appear in multiple matrices when nodes have multiple types."""
        node_types = {
            "NODE:1": ["SmallMolecule", "ChemicalEntity"],
            "NODE:2": ["Gene", "BiologicalEntity"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:affects", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Should create 2 × 2 = 4 matrices (Cartesian product of types)
            expected_keys = [
                ("SmallMolecule", "affects", "Gene"),
                ("SmallMolecule", "affects", "BiologicalEntity"),
                ("ChemicalEntity", "affects", "Gene"),
                ("ChemicalEntity", "affects", "BiologicalEntity")
            ]

            for key in expected_keys:
                assert key in matrices, f"Missing matrix for {key}"

            # All matrices should have the same edge
            for key in expected_keys:
                assert matrices[key].nvals == 1
        finally:
            Path(edges_file).unlink()

    def test_expansion_factor_calculation(self, capsys):
        """Test that type expansion factor is calculated and printed."""
        node_types = {
            "NODE:1": ["SmallMolecule", "ChemicalEntity"],
            "NODE:2": ["Gene"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:affects", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Capture output
            captured = capsys.readouterr()

            # Should print expansion factor
            assert "Type expansion factor" in captured.out
            # 1 edge × 2 src types × 1 tgt type = 2x expansion
            assert "2.00x" in captured.out or "2.0x" in captured.out
        finally:
            Path(edges_file).unlink()

    def test_symmetric_predicates(self):
        """Test that symmetric predicates create bidirectional edges."""
        node_types = {
            "NODE:1": ["Gene"],
            "NODE:2": ["Gene"]
        }

        # directly_physically_interacts_with is symmetric
        edges = [
            {"subject": "NODE:1", "predicate": "biolink:directly_physically_interacts_with", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Should create one matrix (Gene-directly_physically_interacts_with-Gene)
            key = ("Gene", "directly_physically_interacts_with", "Gene")
            assert key in matrices

            # Should have 2 edges (forward and reverse)
            matrix = matrices[key]
            assert matrix.nvals == 2  # Both directions
        finally:
            Path(edges_file).unlink()

    def test_symmetric_with_multiple_types(self):
        """Test symmetric predicates with multiple types per node."""
        node_types = {
            "NODE:1": ["Gene", "BiologicalEntity"],
            "NODE:2": ["Gene", "BiologicalEntity"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:directly_physically_interacts_with", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Should create matrices for all type combinations
            # 2 types × 2 types = 4 combinations, each with forward + reverse
            expected_keys = [
                ("Gene", "directly_physically_interacts_with", "Gene"),
                ("Gene", "directly_physically_interacts_with", "BiologicalEntity"),
                ("BiologicalEntity", "directly_physically_interacts_with", "Gene"),
                ("BiologicalEntity", "directly_physically_interacts_with", "BiologicalEntity")
            ]

            for key in expected_keys:
                assert key in matrices, f"Missing matrix for {key}"
                # Each should have 2 edges (forward and reverse)
                assert matrices[key].nvals == 2
        finally:
            Path(edges_file).unlink()

    def test_subclass_edges_skipped(self):
        """Test that subclass_of edges are skipped."""
        node_types = {
            "NODE:1": ["Gene"],
            "NODE:2": ["Gene"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:subclass_of", "object": "NODE:2"},
            {"subject": "NODE:1", "predicate": "biolink:related_to", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Should not create matrix for subclass_of
            assert ("Gene", "subclass_of", "Gene") not in matrices

            # Should create matrix for related_to
            assert ("Gene", "related_to", "Gene") in matrices
        finally:
            Path(edges_file).unlink()

    def test_nodes_without_types_skipped(self):
        """Test that edges with nodes lacking types are skipped."""
        node_types = {
            "NODE:1": ["Gene"],
            # NODE:2 not in node_types
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:affects", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Should not create any matrices since NODE:2 has no types
            assert len(matrices) == 0
        finally:
            Path(edges_file).unlink()

    def test_separate_index_spaces_per_type(self):
        """Test that each type has its own index space."""
        node_types = {
            "NODE:1": ["SmallMolecule", "ChemicalEntity"],
            "NODE:2": ["SmallMolecule"],
            "NODE:3": ["Gene"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:affects", "object": "NODE:3"},
            {"subject": "NODE:2", "predicate": "biolink:affects", "object": "NODE:3"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # SmallMolecule matrix should have 2 rows (NODE:1, NODE:2)
            sm_gene_matrix = matrices[("SmallMolecule", "affects", "Gene")]
            assert sm_gene_matrix.nrows == 2

            # ChemicalEntity matrix should have 1 row (only NODE:1)
            ce_gene_matrix = matrices[("ChemicalEntity", "affects", "Gene")]
            assert ce_gene_matrix.nrows == 1

            # Both should have 1 column (NODE:3 as Gene)
            assert sm_gene_matrix.ncols == 1
            assert ce_gene_matrix.ncols == 1
        finally:
            Path(edges_file).unlink()

    def test_matrix_size_statistics(self, capsys):
        """Test that matrix size statistics are printed."""
        node_types = {
            "NODE:1": ["Gene"],
            "NODE:2": ["Gene"],
            "NODE:3": ["Disease"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:associated_with", "object": "NODE:3"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            captured = capsys.readouterr()

            # Should print statistics
            assert "Edge statistics" in captured.out
            assert "Unique edge type triples" in captured.out
            assert "Average matrix dimensions" in captured.out
            assert "Average entries per matrix" in captured.out
        finally:
            Path(edges_file).unlink()

    def test_multiple_edges_same_matrix(self):
        """Test multiple edges in the same matrix."""
        node_types = {
            "NODE:1": ["Gene"],
            "NODE:2": ["Gene"],
            "NODE:3": ["Disease"],
            "NODE:4": ["Disease"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:associated_with", "object": "NODE:3"},
            {"subject": "NODE:2", "predicate": "biolink:associated_with", "object": "NODE:4"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            matrix = matrices[("Gene", "associated_with", "Disease")]
            assert matrix.nrows == 2  # Two genes
            assert matrix.ncols == 2  # Two diseases
            assert matrix.nvals == 2  # Two edges
        finally:
            Path(edges_file).unlink()

    def test_duplicate_edges_handled(self):
        """Test that duplicate edges are handled correctly."""
        node_types = {
            "NODE:1": ["Gene"],
            "NODE:2": ["Disease"]
        }

        # Same edge appears twice
        edges = [
            {"subject": "NODE:1", "predicate": "biolink:associated_with", "object": "NODE:2"},
            {"subject": "NODE:1", "predicate": "biolink:associated_with", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Should still only have one entry (dup_op=any)
            matrix = matrices[("Gene", "associated_with", "Disease")]
            assert matrix.nvals == 1
        finally:
            Path(edges_file).unlink()

    def test_three_types_per_node(self):
        """Test nodes with three types create 3x3=9 matrices."""
        node_types = {
            "NODE:1": ["SmallMolecule", "ChemicalEntity", "MolecularEntity"],
            "NODE:2": ["Gene", "BiologicalEntity", "GenomicEntity"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:affects", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            # Should create 3 × 3 = 9 matrices
            assert len(matrices) == 9

            # All should have same edge count
            for matrix in matrices.values():
                assert matrix.nvals == 1
        finally:
            Path(edges_file).unlink()

    def test_empty_edges_file(self):
        """Test with empty edges file."""
        node_types = {
            "NODE:1": ["Gene"]
        }

        tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        tmpfile.close()

        try:
            matrices = build_matrices(tmpfile.name, node_types)
            assert len(matrices) == 0
        finally:
            Path(tmpfile.name).unlink()

    def test_matrix_boolean_dtype(self):
        """Test that matrices use boolean dtype."""
        node_types = {
            "NODE:1": ["Gene"],
            "NODE:2": ["Disease"]
        }

        edges = [
            {"subject": "NODE:1", "predicate": "biolink:associated_with", "object": "NODE:2"}
        ]

        edges_file = self.create_test_edges_file(edges)

        try:
            matrices = build_matrices(edges_file, node_types)

            matrix = matrices[("Gene", "associated_with", "Disease")]
            # GraphBLAS matrices should be boolean
            assert matrix.dtype.name == 'BOOL'
        finally:
            Path(edges_file).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
