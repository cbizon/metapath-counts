"""Integration tests for the metapath analysis pipeline.

These tests use small synthetic KGX files to test end-to-end functionality
with actual GraphBLAS matrices. Tests are kept minimal to avoid OOM issues.
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys

# Add scripts to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from analyze_hop_overlap import (
    build_matrix_list,
    format_metapath,
    analyze_nhop_overlap,
)
from prebuild_matrices import (
    load_node_types,
    build_matrices,
)


@pytest.fixture(scope="module")
def tiny_graph():
    """
    Create a tiny knowledge graph for testing.

    Using module scope to avoid repeated setup/teardown.

    Graph structure:
        Drug1 --affects--> Gene1 --regulates--> Gene2 --causes--> Disease1
        Drug1 --treats--> Disease1 (direct 1-hop for comparison)
    """
    nodes = [
        {"id": "DRUG:1", "category": ["biolink:Drug"], "name": "Drug1"},
        {"id": "GENE:1", "category": ["biolink:Gene"], "name": "Gene1"},
        {"id": "GENE:2", "category": ["biolink:Gene"], "name": "Gene2"},
        {"id": "DISEASE:1", "category": ["biolink:Disease"], "name": "Disease1"},
    ]

    edges = [
        {"subject": "DRUG:1", "predicate": "biolink:affects", "object": "GENE:1"},
        {"subject": "GENE:1", "predicate": "biolink:regulates", "object": "GENE:2"},
        {"subject": "GENE:2", "predicate": "biolink:causes", "object": "DISEASE:1"},
        {"subject": "DRUG:1", "predicate": "biolink:treats", "object": "DISEASE:1"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        nodes_file = Path(tmpdir) / "nodes.jsonl"
        edges_file = Path(tmpdir) / "edges.jsonl"

        with open(nodes_file, 'w') as f:
            for node in nodes:
                f.write(json.dumps(node) + '\n')

        with open(edges_file, 'w') as f:
            for edge in edges:
                f.write(json.dumps(edge) + '\n')

        # Pre-build matrices once for all tests
        node_types = load_node_types(str(nodes_file))
        matrices, node_to_idx = build_matrices(str(edges_file), node_types)

        yield {
            'nodes_file': str(nodes_file),
            'edges_file': str(edges_file),
            'tmpdir': tmpdir,
            'node_types': node_types,
            'matrices': matrices,
            'node_to_idx': node_to_idx,
        }


class TestMatrixBuilding:
    """Test matrix building from KGX files."""

    def test_load_node_types(self, tiny_graph):
        """Test loading nodes returns correct types."""
        node_types = tiny_graph['node_types']
        assert len(node_types) == 4
        assert 'Drug' in node_types['DRUG:1']
        assert 'Gene' in node_types['GENE:1']
        assert 'Disease' in node_types['DISEASE:1']

    def test_build_matrices_creates_expected_keys(self, tiny_graph):
        """Test that matrices are created for each predicate/type combo."""
        matrices = tiny_graph['matrices']

        assert ('Drug', 'affects', 'Gene') in matrices
        assert ('Gene', 'regulates', 'Gene') in matrices
        assert ('Gene', 'causes', 'Disease') in matrices
        assert ('Drug', 'treats', 'Disease') in matrices

    def test_matrix_edge_counts(self, tiny_graph):
        """Test that matrices have correct edge counts."""
        matrices = tiny_graph['matrices']

        # Drug->affects->Gene has 1 edge
        assert matrices[('Drug', 'affects', 'Gene')].nvals == 1

        # Gene->regulates->Gene has 1 edge
        assert matrices[('Gene', 'regulates', 'Gene')].nvals == 1

    def test_matrix_list_has_both_directions(self, tiny_graph):
        """Test that matrix list includes forward and reverse."""
        matrices = tiny_graph['matrices']
        all_matrices, _ = build_matrix_list(matrices)

        affects_entries = [
            (src, tgt, d)
            for src, pred, tgt, _, d in all_matrices
            if pred == 'affects'
        ]

        forward = any(e[0] == 'Drug' and e[1] == 'Gene' and e[2] == 'F' for e in affects_entries)
        reverse = any(e[0] == 'Gene' and e[1] == 'Drug' and e[2] == 'R' for e in affects_entries)

        assert forward, "Forward direction not found"
        assert reverse, "Reverse direction not found"


class TestNHopAnalysis:
    """Test N-hop overlap analysis."""

    def test_1hop_analysis_output_format(self, tiny_graph):
        """Test 1-hop analysis produces valid output."""
        matrices = tiny_graph['matrices']
        output_file = Path(tiny_graph['tmpdir']) / "output_1hop.tsv"

        analyze_nhop_overlap(matrices, str(output_file), n_hops=1)

        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        # Should have header
        assert len(lines) >= 1
        header = lines[0].strip()
        assert '1hop_metapath' in header
        assert '1hop_count' in header

    def test_2hop_analysis_output_format(self, tiny_graph):
        """Test 2-hop analysis produces valid output."""
        matrices = tiny_graph['matrices']
        output_file = Path(tiny_graph['tmpdir']) / "output_2hop.tsv"

        analyze_nhop_overlap(matrices, str(output_file), n_hops=2)

        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) >= 1
        header = lines[0].strip()
        assert '2hop_metapath' in header

    def test_3hop_analysis_finds_path(self, tiny_graph):
        """Test 3-hop analysis finds the expected path."""
        matrices = tiny_graph['matrices']
        output_file = Path(tiny_graph['tmpdir']) / "output_3hop.tsv"

        analyze_nhop_overlap(matrices, str(output_file), n_hops=3)

        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()[1:]  # Skip header

        # Should find some 3-hop paths
        assert len(lines) > 0

        # Parse and check for Drug->Disease path
        found_drug_disease = False
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                path = parts[0]
                if 'Drug' in path and 'Disease' in path:
                    found_drug_disease = True
                    break

        assert found_drug_disease, "Should find Drug->Disease 3-hop path"


class TestMetapathFormatting:
    """Test metapath format/parse roundtrip."""

    def test_format_1hop(self):
        """Test formatting a 1-hop metapath."""
        result = format_metapath(['Drug', 'Disease'], ['treats'], ['F'])
        assert result == 'Drug|treats|F|Disease'

    def test_format_3hop(self):
        """Test formatting a 3-hop metapath."""
        result = format_metapath(
            ['Drug', 'Gene', 'Protein', 'Disease'],
            ['affects', 'regulates', 'causes'],
            ['F', 'R', 'F']
        )
        assert result == 'Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease'

    def test_symmetric_predicate_gets_direction_a(self):
        """Test that symmetric predicates get direction 'A'."""
        # interacts_with is symmetric in biolink
        result = format_metapath(['Gene', 'Gene'], ['interacts_with'], ['F'])
        # Should have 'A' direction, not 'F'
        assert '|A|' in result


class TestPseudoTypeInMatrices:
    """Test pseudo-type handling in matrix building."""

    def test_multi_root_node_creates_pseudo_type_matrix(self):
        """Test that multi-root nodes create pseudo-type matrices."""
        nodes = [
            {"id": "MULTI:1", "category": ["biolink:Gene", "biolink:SmallMolecule"]},
            {"id": "DISEASE:1", "category": ["biolink:Disease"]}
        ]
        edges = [
            {"subject": "MULTI:1", "predicate": "biolink:affects", "object": "DISEASE:1"}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            nodes_file = Path(tmpdir) / "nodes.jsonl"
            edges_file = Path(tmpdir) / "edges.jsonl"

            with open(nodes_file, 'w') as f:
                for node in nodes:
                    f.write(json.dumps(node) + '\n')

            with open(edges_file, 'w') as f:
                for edge in edges:
                    f.write(json.dumps(edge) + '\n')

            node_types = load_node_types(str(nodes_file))
            matrices, _ = build_matrices(str(edges_file), node_types)

            # Should have a matrix with pseudo-type (contains '+')
            pseudo_matrices = [
                key for key in matrices.keys()
                if '+' in key[0] or '+' in key[2]
            ]
            assert len(pseudo_matrices) > 0
