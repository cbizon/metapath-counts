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
    is_palindromic_path,
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
        """Test that 3-hop uses size-based elimination with alphabetical tiebreaker."""
        # last > first: process regardless of direction
        assert should_process_path(
            n_hops=3,
            first_matrix_nvals=499,
            last_matrix_nvals=500,
            src_type='Protein',
            tgt_type='Gene'
        ) == True

        # Equal nvals: fall back to alphabetical (Gene < Protein, so Gene→Protein is canonical)
        assert should_process_path(
            n_hops=3,
            first_matrix_nvals=500,
            last_matrix_nvals=500,
            src_type='Gene',
            tgt_type='Protein'  # canonical direction
        ) == True

        assert should_process_path(
            n_hops=3,
            first_matrix_nvals=500,
            last_matrix_nvals=500,
            src_type='Protein',
            tgt_type='Gene'  # non-canonical direction
        ) == False

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


class TestIsPalindromicPath:
    """Tests for is_palindromic_path function.

    A palindromic path reads the same forward and backward, producing a
    symmetric accumulated matrix. Uses same reversal logic as
    canonicalize_metapath: reverse nodes, reverse predicates, flip F<->R.
    """

    @pytest.fixture
    def symmetric_predicates(self):
        """A small set of symmetric predicates for testing."""
        return {'interacts_with', 'directly_physically_interacts_with',
                'associated_with', 'correlated_with'}

    def test_2hop_palindrome_affects_forward_reverse(self, symmetric_predicates):
        """Gene|affects|F|Disease|affects|R|Gene is palindromic."""
        assert is_palindromic_path(
            ['Gene', 'Disease', 'Gene'],
            ['affects', 'affects'],
            ['F', 'R'],
            symmetric_predicates
        ) is True

    def test_2hop_not_palindrome_different_predicates(self, symmetric_predicates):
        """Gene|regulates|F|Gene|affects|F|Disease is not palindromic (different types + preds)."""
        assert is_palindromic_path(
            ['Gene', 'Gene', 'Disease'],
            ['regulates', 'affects'],
            ['F', 'F'],
            symmetric_predicates
        ) is False

    def test_2hop_not_palindrome_same_type_different_predicates(self, symmetric_predicates):
        """Gene|regulates|F|Gene|interacts_with|A|Gene — same endpoints but not palindromic."""
        assert is_palindromic_path(
            ['Gene', 'Gene', 'Gene'],
            ['regulates', 'interacts_with'],
            ['F', 'F'],
            symmetric_predicates
        ) is False

    def test_2hop_palindrome_symmetric_predicate(self, symmetric_predicates):
        """Gene|interacts_with|A|Protein|interacts_with|A|Gene — palindromic via symmetric pred."""
        # Raw directions are F/R but both become 'A' since interacts_with is symmetric
        assert is_palindromic_path(
            ['Gene', 'Protein', 'Gene'],
            ['interacts_with', 'interacts_with'],
            ['F', 'R'],
            symmetric_predicates
        ) is True

    def test_1hop_palindrome_symmetric_predicate(self, symmetric_predicates):
        """Gene|interacts_with|A|Gene — 1-hop palindromic (symmetric, same types)."""
        assert is_palindromic_path(
            ['Gene', 'Gene'],
            ['interacts_with'],
            ['F'],
            symmetric_predicates
        ) is True

    def test_1hop_not_palindrome_nonsymmetric(self, symmetric_predicates):
        """Gene|regulates|F|Gene — not palindromic (F != R)."""
        assert is_palindromic_path(
            ['Gene', 'Gene'],
            ['regulates'],
            ['F'],
            symmetric_predicates
        ) is False

    def test_1hop_different_types_not_palindromic(self, symmetric_predicates):
        """Gene|affects|F|Disease — different endpoint types, never palindromic."""
        assert is_palindromic_path(
            ['Gene', 'Disease'],
            ['affects'],
            ['F'],
            symmetric_predicates
        ) is False

    def test_3hop_palindrome(self, symmetric_predicates):
        """3-hop palindrome: middle predicate must be symmetric (eff_dir == opposite(eff_dir))."""
        # Gene|affects|F|Disease|associated_with|A|Disease|affects|R|Gene
        # associated_with is symmetric, so the middle direction is 'A' which self-mirrors
        assert is_palindromic_path(
            ['Gene', 'Disease', 'Disease', 'Gene'],
            ['affects', 'associated_with', 'affects'],
            ['F', 'F', 'R'],
            symmetric_predicates
        ) is True

    def test_3hop_not_palindrome(self, symmetric_predicates):
        """3-hop with non-mirrored structure."""
        assert is_palindromic_path(
            ['Gene', 'Disease', 'Protein', 'Gene'],
            ['affects', 'treats', 'affects'],
            ['F', 'F', 'R'],
            symmetric_predicates
        ) is False

    def test_3hop_palindrome_all_symmetric(self, symmetric_predicates):
        """3-hop with all symmetric predicates and mirrored types."""
        assert is_palindromic_path(
            ['Gene', 'Protein', 'Protein', 'Gene'],
            ['interacts_with', 'associated_with', 'interacts_with'],
            ['F', 'F', 'R'],
            symmetric_predicates
        ) is True

    def test_3hop_not_palindrome_asymmetric_middle(self, symmetric_predicates):
        """3-hop: non-symmetric middle predicate prevents palindrome even with mirrored outers.

        For odd-length paths, the middle predicate's effective direction must satisfy
        eff_dir == opposite(eff_dir), which is only true for 'A' (symmetric predicates).
        """
        assert is_palindromic_path(
            ['Gene', 'Disease', 'Disease', 'Gene'],
            ['affects', 'regulates', 'affects'],
            ['F', 'F', 'R'],
            symmetric_predicates
        ) is False


class TestSameTypeDeduplication:
    """Tests for the same-type endpoint deduplication logic in process_path.

    When src_type == tgt_type, the accumulated matrix may contain:
    - Self-pairs (diagonal): a node reaching itself — always meaningless
    - Symmetric duplicates: for palindromic paths, (i,j) and (j,i) both exist

    process_path applies:
    - offdiag for non-palindromic same-type paths (removes diagonal only)
    - triu(k=1) for palindromic same-type paths (removes diagonal + lower triangle)
    """

    def test_offdiag_removes_only_diagonal(self):
        """Non-palindromic same-type: offdiag removes diagonal, keeps both triangles."""
        import graphblas as gb

        # Simulate accumulated matrix with diagonal and both triangles
        # e.g., Gene|regulates|F|Gene|interacts_with|A|Gene (non-palindromic)
        matrix = gb.Matrix.from_coo(
            [0, 0, 1, 1, 2],
            [0, 1, 0, 1, 1],
            [True, True, True, True, True],
            nrows=3, ncols=3, dtype=gb.dtypes.BOOL
        )
        assert matrix.nvals == 5  # (0,0) diag, (0,1) upper, (1,0) lower, (1,1) diag, (2,1) lower

        work_matrix = matrix.select(gb.select.offdiag).new()

        # Diagonal (0,0) and (1,1) removed; off-diagonal (0,1), (1,0), (2,1) preserved
        assert work_matrix.nvals == 3
        assert work_matrix[0, 1].new().value == True
        assert work_matrix[1, 0].new().value == True
        assert work_matrix[2, 1].new().value == True

    def test_triu_removes_diagonal_and_lower_triangle(self):
        """Palindromic same-type: triu(k=1) removes diagonal AND lower triangle."""
        import graphblas as gb

        # Same matrix as above
        matrix = gb.Matrix.from_coo(
            [0, 0, 1, 1, 2],
            [0, 1, 0, 1, 1],
            [True, True, True, True, True],
            nrows=3, ncols=3, dtype=gb.dtypes.BOOL
        )
        assert matrix.nvals == 5

        work_matrix = matrix.select(gb.select.triu, 1).new()

        # Only strict upper triangle: (0,1) survives
        # (0,0) diagonal removed, (1,0) lower removed, (1,1) diagonal removed, (2,1) lower removed
        assert work_matrix.nvals == 1
        assert work_matrix[0, 1].new().value == True

    def test_offdiag_preserves_asymmetric_entries(self):
        """Non-palindromic: lower-triangle-only entries are preserved (not lost like with triu)."""
        import graphblas as gb

        # Matrix where some entries exist only in lower triangle (asymmetric matrix)
        # This is the key difference: offdiag keeps these, triu would discard them
        matrix = gb.Matrix.from_coo(
            [0, 1, 2, 2],
            [0, 0, 0, 1],
            [True, True, True, True],
            nrows=3, ncols=3, dtype=gb.dtypes.BOOL
        )
        # (0,0) diagonal, (1,0) lower only, (2,0) lower only, (2,1) lower only

        work_offdiag = matrix.select(gb.select.offdiag).new()
        work_triu = matrix.select(gb.select.triu, 1).new()

        # offdiag keeps all 3 lower-triangle entries
        assert work_offdiag.nvals == 3
        # triu discards them all — wrong for non-palindromic paths!
        assert work_triu.nvals == 0


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


class TestAggregateExplicitResults:
    """Tests for aggregate_explicit_results function."""

    def test_aggregate_single_result(self):
        """Test aggregating a single explicit result expands to variants."""
        from analyze_hop_overlap import aggregate_explicit_results

        # Create a temporary explicit results file
        with tempfile.TemporaryDirectory() as tmpdir:
            explicit_file = Path(tmpdir) / "explicit.tsv"
            aggregated_file = Path(tmpdir) / "aggregated.tsv"

            # Write explicit results (SmallMolecule should expand to ChemicalEntity)
            with open(explicit_file, 'w') as f:
                f.write("predictor_metapath\tpredictor_count\tpredicted_metapath\tpredicted_count\toverlap\ttotal_possible\n")
                f.write("SmallMolecule|affects|F|Gene\t100\tSmallMolecule|affects|F|Gene\t50\t30\t1000\n")

            # Run aggregation
            aggregate_explicit_results(str(explicit_file), str(aggregated_file))

            # Read aggregated results
            with open(aggregated_file) as f:
                lines = f.readlines()[1:]  # Skip header

            # Should have multiple lines due to hierarchical expansion
            assert len(lines) > 1

            # Parse results
            results = {}
            for line in lines:
                parts = line.strip().split('\t')
                nhop, nhop_count, onehop, onehop_count, overlap, total = parts
                results[(nhop, onehop)] = (int(nhop_count), int(onehop_count), int(overlap), int(total))

            # Canonical form should be present (Gene < SmallMolecule alphabetically)
            assert ("Gene|affects|R|SmallMolecule", "Gene|affects|R|SmallMolecule") in results

            # ChemicalEntity (ancestor of SmallMolecule) should appear
            chem_variants = [k for k in results.keys() if "ChemicalEntity" in k[0] or "ChemicalEntity" in k[1]]
            assert len(chem_variants) > 0

    def test_aggregate_preserves_counts(self):
        """Test that aggregation preserves counts correctly."""
        from analyze_hop_overlap import aggregate_explicit_results

        with tempfile.TemporaryDirectory() as tmpdir:
            explicit_file = Path(tmpdir) / "explicit.tsv"
            aggregated_file = Path(tmpdir) / "aggregated.tsv"

            # Write explicit results
            with open(explicit_file, 'w') as f:
                f.write("predictor_metapath\tpredictor_count\tpredicted_metapath\tpredicted_count\toverlap\ttotal_possible\n")
                f.write("Gene|affects|F|Disease\t200\tGene|affects|F|Disease\t100\t50\t5000\n")

            aggregate_explicit_results(str(explicit_file), str(aggregated_file))

            with open(aggregated_file) as f:
                lines = f.readlines()[1:]

            # Find the canonical form row (Disease < Gene alphabetically)
            canonical_nhop = "Disease|affects|R|Gene"
            canonical_onehop = "Disease|affects|R|Gene"
            for line in lines:
                parts = line.strip().split('\t')
                if parts[0] == canonical_nhop and parts[2] == canonical_onehop:
                    assert int(parts[1]) == 200  # nhop_count preserved
                    assert int(parts[3]) == 100  # onehop_count preserved
                    assert int(parts[4]) == 50   # overlap preserved
                    assert int(parts[5]) == 5000 # total_possible preserved
                    break
            else:
                pytest.fail(f"Canonical result {canonical_nhop} not found in aggregated output")

    def test_aggregate_multiple_results_sum_correctly(self):
        """Test that multiple explicit results sum correctly when aggregated."""
        from analyze_hop_overlap import aggregate_explicit_results

        with tempfile.TemporaryDirectory() as tmpdir:
            explicit_file = Path(tmpdir) / "explicit.tsv"
            aggregated_file = Path(tmpdir) / "aggregated.tsv"

            # Two different explicit paths that both aggregate to BiologicalEntity
            with open(explicit_file, 'w') as f:
                f.write("predictor_metapath\tpredictor_count\tpredicted_metapath\tpredicted_count\toverlap\ttotal_possible\n")
                f.write("Gene|affects|F|Disease\t100\tGene|affects|F|Disease\t50\t25\t1000\n")
                f.write("Protein|affects|F|Disease\t80\tProtein|affects|F|Disease\t40\t20\t800\n")

            aggregate_explicit_results(str(explicit_file), str(aggregated_file))

            with open(aggregated_file) as f:
                lines = f.readlines()[1:]

            results = {}
            for line in lines:
                parts = line.strip().split('\t')
                key = (parts[0], parts[2])
                results[key] = (int(parts[1]), int(parts[3]), int(parts[4]), int(parts[5]))

            # BiologicalEntity (ancestor of both Gene and Protein) should have summed counts
            bio_variants = [(k, v) for k, v in results.items()
                           if "BiologicalEntity" in k[0] and "BiologicalEntity" in k[1]]

            # Should have at least one BiologicalEntity variant
            assert len(bio_variants) > 0

            # The counts should be sums (100+80=180 for nhop, 50+40=90 for onehop, etc.)
            for key, (nhop, onehop, overlap, total) in bio_variants:
                if key == ("BiologicalEntity|affects|F|Disease", "BiologicalEntity|affects|F|Disease"):
                    assert nhop == 180  # 100 + 80
                    assert onehop == 90  # 50 + 40
                    assert overlap == 45  # 25 + 20
                    assert total == 1800  # 1000 + 800


class TestDetermineNeededMatrices:
    """Tests for determine_needed_matrices function."""

    def test_determine_needed_for_single_matrix1(self):
        """Test determining needed matrices for a single starting matrix."""
        from analyze_hop_overlap import determine_needed_matrices

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            # Create a simple manifest
            manifest = {
                "num_matrices": 3,
                "matrices": [
                    {"src_type": "Drug", "predicate": "treats", "tgt_type": "Disease", "filename": "m1.npz"},
                    {"src_type": "Drug", "predicate": "affects", "tgt_type": "Gene", "filename": "m2.npz"},
                    {"src_type": "Gene", "predicate": "causes", "tgt_type": "Disease", "filename": "m3.npz"},
                ]
            }

            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)

            # Determine needed matrices for 2-hop starting from Drug|affects|F|Gene
            matrix1_spec = ("Drug", "affects", "F", "Gene")
            needed = determine_needed_matrices(manifest_path, n_hops=2, matrix1_spec=matrix1_spec)

            # Should need the starting matrix
            assert ("Drug", "affects", "Gene") in needed

            # Should need matrices that continue from Gene
            assert ("Gene", "causes", "Disease") in needed

            # Should need 1-hop comparison matrices for Drug->Disease
            # (final endpoints of Drug->Gene->Disease)
            assert ("Drug", "treats", "Disease") in needed

    def test_determine_needed_excludes_unrelated(self):
        """Test that unrelated matrices are excluded."""
        from analyze_hop_overlap import determine_needed_matrices

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            manifest = {
                "num_matrices": 4,
                "matrices": [
                    {"src_type": "Drug", "predicate": "affects", "tgt_type": "Gene", "filename": "m1.npz"},
                    {"src_type": "Gene", "predicate": "causes", "tgt_type": "Disease", "filename": "m2.npz"},
                    {"src_type": "Protein", "predicate": "binds", "tgt_type": "Pathway", "filename": "m3.npz"},
                    {"src_type": "Drug", "predicate": "treats", "tgt_type": "Disease", "filename": "m4.npz"},
                ]
            }

            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)

            matrix1_spec = ("Drug", "affects", "F", "Gene")
            needed = determine_needed_matrices(manifest_path, n_hops=2, matrix1_spec=matrix1_spec)

            # Protein->Pathway is unrelated to Drug->Gene->Disease paths
            assert ("Protein", "binds", "Pathway") not in needed

    def test_determine_needed_for_1hop(self):
        """Test determining needed matrices for 1-hop analysis."""
        from analyze_hop_overlap import determine_needed_matrices

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            manifest = {
                "num_matrices": 2,
                "matrices": [
                    {"src_type": "Drug", "predicate": "treats", "tgt_type": "Disease", "filename": "m1.npz"},
                    {"src_type": "Gene", "predicate": "causes", "tgt_type": "Disease", "filename": "m2.npz"},
                ]
            }

            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)

            # For 1-hop Drug->Disease, we only need Drug->Disease matrices
            matrix1_spec = ("Drug", "treats", "F", "Disease")
            needed = determine_needed_matrices(manifest_path, n_hops=1, matrix1_spec=matrix1_spec)

            assert ("Drug", "treats", "Disease") in needed
            # Gene->Disease should NOT be needed (different source type)
            # But it might be included as a 1-hop comparison matrix if types match
            # For 1-hop analysis, we compare Drug->Disease with all Drug->Disease matrices


class TestLoadPrebuiltMatrices:
    """Tests for load_prebuilt_matrices function."""

    def test_load_with_filtering(self):
        """Test loading only needed matrices."""
        from analyze_hop_overlap import load_prebuilt_matrices
        import numpy as np
        import graphblas as gb

        with tempfile.TemporaryDirectory() as tmpdir:
            matrices_dir = Path(tmpdir)

            # Create mock manifest
            manifest = {
                "num_matrices": 2,
                "total_size_bytes": 1000,
                "matrices": [
                    {"src_type": "Drug", "predicate": "treats", "tgt_type": "Disease",
                     "filename": "drug_treats_disease.npz", "nvals": 10},
                    {"src_type": "Gene", "predicate": "causes", "tgt_type": "Disease",
                     "filename": "gene_causes_disease.npz", "nvals": 5},
                ]
            }

            with open(matrices_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f)

            # Create mock matrix files
            for mat_info in manifest["matrices"]:
                np.savez(
                    matrices_dir / mat_info["filename"],
                    rows=np.array([0, 1]),
                    cols=np.array([0, 0]),
                    vals=np.array([True, True]),
                    nrows=2,
                    ncols=1
                )

            # Load only Drug->Disease
            needed = {("Drug", "treats", "Disease")}
            matrices = load_prebuilt_matrices(str(matrices_dir), needed_triples=needed)

            assert len(matrices) == 1
            assert ("Drug", "treats", "Disease") in matrices
            assert ("Gene", "causes", "Disease") not in matrices

    def test_load_all_matrices(self):
        """Test loading all matrices when no filter specified."""
        from analyze_hop_overlap import load_prebuilt_matrices
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            matrices_dir = Path(tmpdir)

            manifest = {
                "num_matrices": 2,
                "total_size_bytes": 1000,
                "matrices": [
                    {"src_type": "Drug", "predicate": "treats", "tgt_type": "Disease",
                     "filename": "m1.npz", "nvals": 10},
                    {"src_type": "Gene", "predicate": "causes", "tgt_type": "Disease",
                     "filename": "m2.npz", "nvals": 5},
                ]
            }

            with open(matrices_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f)

            for mat_info in manifest["matrices"]:
                np.savez(
                    matrices_dir / mat_info["filename"],
                    rows=np.array([0]),
                    cols=np.array([0]),
                    vals=np.array([True]),
                    nrows=1,
                    ncols=1
                )

            # Load all (no filter)
            matrices = load_prebuilt_matrices(str(matrices_dir), needed_triples=None)

            assert len(matrices) == 2
            assert ("Drug", "treats", "Disease") in matrices
            assert ("Gene", "causes", "Disease") in matrices
