"""Tests for analyze_3hop_overlap.py functions."""

import pytest
import tempfile
import json
from pathlib import Path
import sys

# Add scripts to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from analyze_hop_overlap import (
    format_metapath,
    build_matrix_list,
    is_canonical_direction,
    should_process_path,
)
from prebuild_matrices import (
    load_node_types,
    build_matrices,
)


class TestIsCanonicalDirection:
    """Tests for is_canonical_direction function."""

    def test_alphabetically_first_is_canonical(self):
        """Test that alphabetically first source type is canonical."""
        assert is_canonical_direction('Disease', 'Drug') == True
        assert is_canonical_direction('Drug', 'Gene') == True
        assert is_canonical_direction('Gene', 'Protein') == True

    def test_alphabetically_second_is_not_canonical(self):
        """Test that alphabetically second source type is not canonical."""
        assert is_canonical_direction('Drug', 'Disease') == False
        assert is_canonical_direction('Gene', 'Drug') == False
        assert is_canonical_direction('Protein', 'Gene') == False

    def test_same_type_is_canonical(self):
        """Test that same type is always canonical."""
        assert is_canonical_direction('Gene', 'Gene') == True
        assert is_canonical_direction('Disease', 'Disease') == True


class TestShouldProcessPath:
    """Tests for should_process_path function."""

    def test_1hop_uses_canonical_direction(self):
        """Test that 1-hop uses canonical direction, not size."""
        # For 1-hop, size doesn't matter - only alphabetical order
        # Disease < Drug alphabetically, so Disease->Drug is canonical
        assert should_process_path(
            n_hops=1,
            first_matrix_nvals=1000,  # Doesn't matter for 1-hop
            last_matrix_nvals=1000,
            src_type='Disease',
            tgt_type='Drug'
        ) == True

        # Drug > Disease, so Drug->Disease is NOT canonical
        assert should_process_path(
            n_hops=1,
            first_matrix_nvals=1000,
            last_matrix_nvals=1000,
            src_type='Drug',
            tgt_type='Disease'
        ) == False

    def test_2hop_uses_size(self):
        """Test that 2-hop uses size-based elimination."""
        # For N>1, use size: last >= first
        assert should_process_path(
            n_hops=2,
            first_matrix_nvals=100,
            last_matrix_nvals=200,
            src_type='Drug',
            tgt_type='Disease'  # Would be non-canonical, but size wins
        ) == True

        assert should_process_path(
            n_hops=2,
            first_matrix_nvals=200,
            last_matrix_nvals=100,
            src_type='Disease',
            tgt_type='Drug'  # Would be canonical, but size loses
        ) == False

    def test_3hop_uses_size(self):
        """Test that 3-hop uses size-based elimination."""
        assert should_process_path(
            n_hops=3,
            first_matrix_nvals=500,
            last_matrix_nvals=500,  # Equal is OK
            src_type='Protein',
            tgt_type='Gene'
        ) == True

        assert should_process_path(
            n_hops=3,
            first_matrix_nvals=1000,
            last_matrix_nvals=999,  # Just below
            src_type='Gene',
            tgt_type='Protein'
        ) == False

    def test_1hop_same_type(self):
        """Test 1-hop with same source and target type."""
        # Gene -> Gene: same type, so canonical (Gene <= Gene)
        assert should_process_path(
            n_hops=1,
            first_matrix_nvals=100,
            last_matrix_nvals=100,
            src_type='Gene',
            tgt_type='Gene'
        ) == True


class TestFormatMetapath:
    """Tests for format_metapath function."""

    def test_format_1hop_forward(self):
        """Test formatting a 1-hop forward metapath."""
        result = format_metapath(
            ['Drug', 'Disease'],
            ['treats'],
            ['F']
        )
        assert result == 'Drug|treats|F|Disease'

    def test_format_1hop_reverse(self):
        """Test formatting a 1-hop reverse metapath."""
        result = format_metapath(
            ['Disease', 'Drug'],
            ['treats'],
            ['R']
        )
        assert result == 'Disease|treats|R|Drug'

    def test_format_2hop(self):
        """Test formatting a 2-hop metapath."""
        result = format_metapath(
            ['Drug', 'Gene', 'Disease'],
            ['affects', 'associated_with'],
            ['F', 'F']
        )
        # associated_with is symmetric, so should be 'A'
        assert result == 'Drug|affects|F|Gene|associated_with|A|Disease'

    def test_format_3hop(self):
        """Test formatting a 3-hop metapath."""
        result = format_metapath(
            ['Drug', 'Gene', 'Protein', 'Disease'],
            ['affects', 'regulates', 'causes'],
            ['F', 'R', 'F']
        )
        assert result == 'Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease'

    def test_format_symmetric_predicate(self):
        """Test that symmetric predicates get direction 'A'."""
        # directly_physically_interacts_with is symmetric
        result = format_metapath(
            ['Gene', 'Gene'],
            ['directly_physically_interacts_with'],
            ['F']  # Should become 'A'
        )
        assert result == 'Gene|directly_physically_interacts_with|A|Gene'

    def test_format_mixed_symmetric_nonsymmetric(self):
        """Test metapath with both symmetric and non-symmetric predicates."""
        result = format_metapath(
            ['Gene', 'Gene', 'Disease'],
            ['directly_physically_interacts_with', 'causes'],
            ['F', 'F']
        )
        # First predicate is symmetric (A), second is not (F)
        assert result == 'Gene|directly_physically_interacts_with|A|Gene|causes|F|Disease'


class TestLoadNodeTypes:
    """Tests for load_node_types function."""

    def test_load_basic_nodes(self):
        """Test loading nodes from a basic KGX file (single type assignment)."""
        nodes_data = [
            {"id": "PUBCHEM:123", "category": ["biolink:SmallMolecule", "biolink:ChemicalEntity"], "name": "Aspirin"},
            {"id": "MONDO:456", "category": ["biolink:Disease"], "name": "Headache"},
            {"id": "HGNC:789", "category": ["biolink:Gene"], "name": "TP53"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for node in nodes_data:
                f.write(json.dumps(node) + '\n')
            temp_path = f.name

        try:
            node_types = load_node_types(temp_path)

            assert len(node_types) == 3
            # With single type assignment, returns most specific type
            assert node_types['PUBCHEM:123'] == 'SmallMolecule'
            assert node_types['MONDO:456'] == 'Disease'
            assert node_types['HGNC:789'] == 'Gene'
        finally:
            Path(temp_path).unlink()

    def test_load_node_without_category(self):
        """Test that nodes without category are skipped."""
        nodes_data = [
            {"id": "NODE:1", "category": ["biolink:Gene"], "name": "Gene1"},
            {"id": "NODE:2", "name": "NoCategory"},  # No category
            {"id": "NODE:3", "category": [], "name": "EmptyCategory"},  # Empty category
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for node in nodes_data:
                f.write(json.dumps(node) + '\n')
            temp_path = f.name

        try:
            node_types = load_node_types(temp_path)

            # Only NODE:1 should be loaded
            assert len(node_types) == 1
            assert 'NODE:1' in node_types
            assert 'NODE:2' not in node_types
            assert 'NODE:3' not in node_types
        finally:
            Path(temp_path).unlink()


class TestBuildMatrices:
    """Tests for build_matrices function."""

    def test_build_simple_matrices(self):
        """Test building matrices from simple edge data (single type assignment)."""
        # Create test node types (single type per node)
        node_types = {
            'DRUG:1': 'Drug',
            'DRUG:2': 'Drug',
            'GENE:1': 'Gene',
            'GENE:2': 'Gene',
            'DISEASE:1': 'Disease',
        }

        # Create test edges
        edges_data = [
            {"subject": "DRUG:1", "predicate": "biolink:affects", "object": "GENE:1"},
            {"subject": "DRUG:2", "predicate": "biolink:affects", "object": "GENE:2"},
            {"subject": "GENE:1", "predicate": "biolink:causes", "object": "DISEASE:1"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for edge in edges_data:
                f.write(json.dumps(edge) + '\n')
            temp_path = f.name

        try:
            matrices, node_to_idx = build_matrices(temp_path, node_types)

            # Should have matrices for each (src_type, pred, tgt_type) triple
            assert ('Drug', 'affects', 'Gene') in matrices
            assert ('Gene', 'causes', 'Disease') in matrices

            # Check matrix dimensions
            drug_gene_matrix = matrices[('Drug', 'affects', 'Gene')]
            assert drug_gene_matrix.nrows == 2  # 2 drugs
            assert drug_gene_matrix.ncols == 2  # 2 genes
            assert drug_gene_matrix.nvals == 2  # 2 edges

            gene_disease_matrix = matrices[('Gene', 'causes', 'Disease')]
            assert gene_disease_matrix.nrows == 2  # 2 genes
            assert gene_disease_matrix.ncols == 1  # 1 disease
            assert gene_disease_matrix.nvals == 1  # 1 edge
        finally:
            Path(temp_path).unlink()

    def test_build_matrices_skips_subclass(self):
        """Test that subclass_of edges are skipped."""
        node_types = {
            'NODE:1': 'Gene',
            'NODE:2': 'Gene',
        }

        edges_data = [
            {"subject": "NODE:1", "predicate": "biolink:subclass_of", "object": "NODE:2"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for edge in edges_data:
                f.write(json.dumps(edge) + '\n')
            temp_path = f.name

        try:
            matrices, node_to_idx = build_matrices(temp_path, node_types)
            # Should have no matrices since subclass_of is skipped
            assert len(matrices) == 0
        finally:
            Path(temp_path).unlink()

    def test_build_matrices_symmetric_predicate(self):
        """Test that symmetric predicates create edges in both directions (single type assignment)."""
        node_types = {
            'GENE:1': 'Gene',
            'GENE:2': 'Gene',
        }

        edges_data = [
            {"subject": "GENE:1", "predicate": "biolink:directly_physically_interacts_with", "object": "GENE:2"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for edge in edges_data:
                f.write(json.dumps(edge) + '\n')
            temp_path = f.name

        try:
            matrices, node_to_idx = build_matrices(temp_path, node_types)

            # Should have matrix for Gene -> Gene
            assert ('Gene', 'directly_physically_interacts_with', 'Gene') in matrices

            # The symmetric edge should be added in both directions
            matrix = matrices[('Gene', 'directly_physically_interacts_with', 'Gene')]
            # 2 edges: (GENE:1, GENE:2) and (GENE:2, GENE:1) from symmetric handling
            assert matrix.nvals == 2
        finally:
            Path(temp_path).unlink()


class TestBuildMatrixList:
    """Tests for build_matrix_list function."""

    def test_build_matrix_list_forward_reverse(self):
        """Test that matrix list includes forward and reverse directions (single type assignment)."""
        node_types = {
            'DRUG:1': 'Drug',
            'GENE:1': 'Gene',
        }

        edges_data = [
            {"subject": "DRUG:1", "predicate": "biolink:affects", "object": "GENE:1"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for edge in edges_data:
                f.write(json.dumps(edge) + '\n')
            temp_path = f.name

        try:
            matrices, node_to_idx = build_matrices(temp_path, node_types)
            all_matrices, matrix_metadata = build_matrix_list(matrices)

            # Should have both forward and reverse
            forward_found = False
            reverse_found = False

            for src_type, pred, tgt_type, matrix, direction in all_matrices:
                if src_type == 'Drug' and pred == 'affects' and tgt_type == 'Gene' and direction == 'F':
                    forward_found = True
                if src_type == 'Gene' and pred == 'affects' and tgt_type == 'Drug' and direction == 'R':
                    reverse_found = True

            assert forward_found, "Forward direction not found"
            assert reverse_found, "Reverse direction not found"
        finally:
            Path(temp_path).unlink()

    def test_build_matrix_list_symmetric_no_reverse(self):
        """Test that symmetric predicates don't create separate reverse entries (single type assignment)."""
        node_types = {
            'GENE:1': 'Gene',
            'GENE:2': 'Gene',
        }

        edges_data = [
            {"subject": "GENE:1", "predicate": "biolink:directly_physically_interacts_with", "object": "GENE:2"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for edge in edges_data:
                f.write(json.dumps(edge) + '\n')
            temp_path = f.name

        try:
            matrices, node_to_idx = build_matrices(temp_path, node_types)
            all_matrices, matrix_metadata = build_matrix_list(matrices)

            # Count entries for symmetric predicate
            sym_entries = [
                (src, pred, tgt, d)
                for src, pred, tgt, _, d in all_matrices
                if pred == 'directly_physically_interacts_with'
            ]

            # Should only have 1 entry (forward), not 2 (no separate reverse for symmetric)
            assert len(sym_entries) == 1
            assert sym_entries[0][3] == 'F'  # Direction should be forward
        finally:
            Path(temp_path).unlink()

    def test_build_matrix_list_bidirectional_edges(self):
        """Test that bidirectional physical edges don't overwrite each other in matrix_metadata.

        This tests the bug where if both ChemicalEntity→treats→Disease and
        Disease→treats→ChemicalEntity exist as physical matrices, the forward
        direction gets overwritten by the transpose of the reverse matrix.
        """
        import graphblas as gb

        # Create two physical matrices: forward and reverse
        # ChemicalEntity→treats→Disease: 3 edges
        forward_matrix = gb.Matrix.from_coo(
            [0, 1, 2],  # rows (chemicals)
            [0, 0, 1],  # cols (diseases)
            [True, True, True],
            nrows=3,
            ncols=2,
            dtype=gb.dtypes.BOOL
        )

        # Disease→treats→ChemicalEntity: 1 edge (unusual but exists in real data)
        reverse_matrix = gb.Matrix.from_coo(
            [0],  # rows (diseases)
            [1],  # cols (chemicals)
            [True],
            nrows=2,
            ncols=3,
            dtype=gb.dtypes.BOOL
        )

        # Simulate the input matrices dict as it comes from load_prebuilt_matrices
        matrices = {
            ('ChemicalEntity', 'treats', 'Disease'): forward_matrix,
            ('Disease', 'treats', 'ChemicalEntity'): reverse_matrix,
        }

        all_matrices, matrix_metadata = build_matrix_list(matrices)

        # With the fix, matrix_metadata keys now include direction:
        # (src_type, pred, tgt_type, direction) -> matrix

        # Check that we have BOTH the forward ChemicalEntity→Disease entry
        # AND the reverse Disease→ChemicalEntity entry (as transpose)
        assert ('ChemicalEntity', 'treats', 'Disease', 'F') in matrix_metadata
        assert ('Disease', 'treats', 'ChemicalEntity', 'F') in matrix_metadata

        # Check forward ChemicalEntity→Disease (should have 3 edges from physical matrix)
        forward_metadata_matrix = matrix_metadata[('ChemicalEntity', 'treats', 'Disease', 'F')]
        assert forward_metadata_matrix.nvals == 3, \
            f"Expected 3 edges from forward ChemicalEntity→Disease matrix, got {forward_metadata_matrix.nvals}"

        # Check forward Disease→ChemicalEntity (should have 1 edge from physical matrix)
        reverse_metadata_matrix = matrix_metadata[('Disease', 'treats', 'ChemicalEntity', 'F')]
        assert reverse_metadata_matrix.nvals == 1, \
            f"Expected 1 edge from forward Disease→ChemicalEntity matrix, got {reverse_metadata_matrix.nvals}"

        # Also check that we have the synthetic reverse entries (matrix.T)
        assert ('Disease', 'treats', 'ChemicalEntity', 'R') in matrix_metadata
        assert ('ChemicalEntity', 'treats', 'Disease', 'R') in matrix_metadata

        # ChemicalEntity→Disease as reverse (transpose of Disease→ChemicalEntity)
        # Should have same edges as forward Disease→ChemicalEntity (1 edge)
        ce_disease_reverse = matrix_metadata[('ChemicalEntity', 'treats', 'Disease', 'R')]
        assert ce_disease_reverse.nvals == 1

        # Disease→ChemicalEntity as reverse (transpose of ChemicalEntity→Disease)
        # Should have same edges as forward ChemicalEntity→Disease (3 edges)
        disease_ce_reverse = matrix_metadata[('Disease', 'treats', 'ChemicalEntity', 'R')]
        assert disease_ce_reverse.nvals == 3
