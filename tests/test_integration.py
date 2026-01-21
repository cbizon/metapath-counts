"""Integration tests for the metapath analysis pipeline.

These tests use small synthetic KGX files to test end-to-end functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys

# Add scripts to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from analyze_3hop_overlap import (
    load_node_types,
    build_matrices,
    build_matrix_list,
    format_metapath,
    analyze_3hop_overlap,
    analyze_nhop_overlap,
)
from group_by_onehop import (
    parse_metapath,
    calculate_metrics,
)


@pytest.fixture
def tiny_graph():
    """
    Create a tiny knowledge graph for testing.

    Graph structure:
        Drug1 --affects--> Gene1 --regulates--> Gene2 --causes--> Disease1
        Drug1 --treats--> Disease1 (direct 1-hop for comparison)
        Drug2 --affects--> Gene1
        Gene1 --interacts_with--> Gene2 (symmetric)
    """
    nodes = [
        {"id": "DRUG:1", "category": ["biolink:Drug", "biolink:ChemicalEntity"], "name": "Drug1"},
        {"id": "DRUG:2", "category": ["biolink:Drug"], "name": "Drug2"},
        {"id": "GENE:1", "category": ["biolink:Gene"], "name": "Gene1"},
        {"id": "GENE:2", "category": ["biolink:Gene"], "name": "Gene2"},
        {"id": "DISEASE:1", "category": ["biolink:Disease"], "name": "Disease1"},
    ]

    edges = [
        {"subject": "DRUG:1", "predicate": "biolink:affects", "object": "GENE:1"},
        {"subject": "DRUG:2", "predicate": "biolink:affects", "object": "GENE:1"},
        {"subject": "GENE:1", "predicate": "biolink:regulates", "object": "GENE:2"},
        {"subject": "GENE:2", "predicate": "biolink:causes", "object": "DISEASE:1"},
        {"subject": "DRUG:1", "predicate": "biolink:treats", "object": "DISEASE:1"},
        {"subject": "GENE:1", "predicate": "biolink:interacts_with", "object": "GENE:2"},
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

        yield {
            'nodes_file': str(nodes_file),
            'edges_file': str(edges_file),
            'tmpdir': tmpdir,
        }


@pytest.fixture
def linear_chain_graph():
    """
    Create a simple linear chain graph for testing.

    Graph structure:
        A --p1--> B --p2--> C --p3--> D
        A --direct--> D (1-hop for comparison)

    This creates exactly one 3-hop path that should predict the 1-hop.
    """
    nodes = [
        {"id": "A:1", "category": ["biolink:Gene"], "name": "A"},
        {"id": "B:1", "category": ["biolink:Protein"], "name": "B"},
        {"id": "C:1", "category": ["biolink:Pathway"], "name": "C"},
        {"id": "D:1", "category": ["biolink:Disease"], "name": "D"},
    ]

    edges = [
        {"subject": "A:1", "predicate": "biolink:affects", "object": "B:1"},
        {"subject": "B:1", "predicate": "biolink:regulates", "object": "C:1"},
        {"subject": "C:1", "predicate": "biolink:causes", "object": "D:1"},
        {"subject": "A:1", "predicate": "biolink:associated_with", "object": "D:1"},
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

        yield {
            'nodes_file': str(nodes_file),
            'edges_file': str(edges_file),
            'tmpdir': tmpdir,
        }


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_load_tiny_graph(self, tiny_graph):
        """Test loading nodes and edges from tiny graph."""
        node_types = load_node_types(tiny_graph['nodes_file'])

        assert len(node_types) == 5
        assert node_types['DRUG:1'] == 'Drug'
        assert node_types['GENE:1'] == 'Gene'
        assert node_types['DISEASE:1'] == 'Disease'

    def test_build_matrices_tiny_graph(self, tiny_graph):
        """Test building matrices from tiny graph."""
        node_types = load_node_types(tiny_graph['nodes_file'])
        matrices = build_matrices(tiny_graph['edges_file'], node_types)

        # Should have matrices for each predicate/type combination
        assert ('Drug', 'affects', 'Gene') in matrices
        assert ('Gene', 'regulates', 'Gene') in matrices
        assert ('Gene', 'causes', 'Disease') in matrices
        assert ('Drug', 'treats', 'Disease') in matrices

        # Check Drug->affects->Gene matrix
        drug_gene = matrices[('Drug', 'affects', 'Gene')]
        assert drug_gene.nvals == 2  # Two drugs affect Gene1

    def test_matrix_list_has_both_directions(self, tiny_graph):
        """Test that matrix list includes forward and reverse."""
        node_types = load_node_types(tiny_graph['nodes_file'])
        matrices = build_matrices(tiny_graph['edges_file'], node_types)
        all_matrices, _ = build_matrix_list(matrices)

        # Find affects matrices
        affects_entries = [
            (src, tgt, d)
            for src, pred, tgt, _, d in all_matrices
            if pred == 'affects'
        ]

        # Should have both forward (Drug->Gene) and reverse (Gene->Drug)
        forward = any(e[0] == 'Drug' and e[1] == 'Gene' and e[2] == 'F' for e in affects_entries)
        reverse = any(e[0] == 'Gene' and e[1] == 'Drug' and e[2] == 'R' for e in affects_entries)

        assert forward, "Forward direction not found"
        assert reverse, "Reverse direction not found"

    def test_full_analysis_linear_chain(self, linear_chain_graph):
        """Test full analysis on linear chain produces expected output."""
        node_types = load_node_types(linear_chain_graph['nodes_file'])
        matrices = build_matrices(linear_chain_graph['edges_file'], node_types)

        output_file = Path(linear_chain_graph['tmpdir']) / "output.tsv"
        analyze_3hop_overlap(matrices, str(output_file))

        # Check output file exists and has content
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        # Should have header + at least one data line
        assert len(lines) >= 2

        # Check header
        header = lines[0].strip().split('\t')
        assert header[0] == '3hop_metapath'
        assert header[1] == '3hop_count'
        assert header[2] == '1hop_metapath'

    def test_3hop_path_counts(self, linear_chain_graph):
        """Test that 3-hop path counts are correct."""
        node_types = load_node_types(linear_chain_graph['nodes_file'])
        matrices = build_matrices(linear_chain_graph['edges_file'], node_types)

        output_file = Path(linear_chain_graph['tmpdir']) / "output.tsv"
        analyze_3hop_overlap(matrices, str(output_file))

        with open(output_file) as f:
            lines = f.readlines()[1:]  # Skip header

        # Parse results
        results = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                results.append({
                    '3hop': parts[0],
                    '3hop_count': int(parts[1]),
                    '1hop': parts[2],
                    '1hop_count': int(parts[3]),
                    'overlap': int(parts[4]),
                })

        # Should find the path Gene|affects|F|Protein|regulates|F|Pathway|causes|F|Disease
        # with count of 1 (single path A->B->C->D)
        gene_to_disease = [
            r for r in results
            if 'Gene' in r['3hop'] and 'Disease' in r['3hop']
            and r['3hop_count'] == 1
        ]
        assert len(gene_to_disease) > 0, "Expected to find Gene->Disease 3-hop path"


class TestMetricCalculations:
    """Test metric calculations with known values."""

    def test_metrics_perfect_classifier(self):
        """Test metrics for a perfect classifier."""
        metrics = calculate_metrics(
            threehop_count=50,
            onehop_count=50,
            overlap=50,
            total_possible=100
        )

        assert metrics['Precision'] == 1.0
        assert metrics['Recall'] == 1.0
        assert metrics['F1'] == 1.0
        assert metrics['MCC'] == 1.0
        assert metrics['Accuracy'] == 1.0

    def test_metrics_random_classifier(self):
        """Test metrics for a random classifier (50% precision)."""
        # 50 positives, 50 negatives, predict 50 as positive
        # Overlap 25 (half correct)
        metrics = calculate_metrics(
            threehop_count=50,
            onehop_count=50,
            overlap=25,
            total_possible=100
        )

        # TP=25, FP=25, FN=25, TN=25
        assert metrics['TP'] == 25
        assert metrics['FP'] == 25
        assert metrics['FN'] == 25
        assert metrics['TN'] == 25

        # Precision = 25/50 = 0.5
        assert abs(metrics['Precision'] - 0.5) < 0.001

        # MCC should be 0 for random classifier
        assert abs(metrics['MCC']) < 0.001


class TestNHopAnalysis:
    """Test N-hop analysis generalization."""

    def test_1hop_analysis(self, linear_chain_graph):
        """Test 1-hop analysis (comparing 1-hop with 1-hop)."""
        node_types = load_node_types(linear_chain_graph['nodes_file'])
        matrices = build_matrices(linear_chain_graph['edges_file'], node_types)

        output_file = Path(linear_chain_graph['tmpdir']) / "output_1hop.tsv"
        analyze_nhop_overlap(matrices, str(output_file), n_hops=1)

        # Check output exists
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        # Should have header + data
        assert len(lines) >= 2

        # Check header
        header = lines[0].strip()
        assert '1hop_metapath' in header
        assert '1hop_count' in header

    def test_2hop_analysis(self, linear_chain_graph):
        """Test 2-hop analysis (comparing 2-hop with 1-hop)."""
        node_types = load_node_types(linear_chain_graph['nodes_file'])
        matrices = build_matrices(linear_chain_graph['edges_file'], node_types)

        output_file = Path(linear_chain_graph['tmpdir']) / "output_2hop.tsv"
        analyze_nhop_overlap(matrices, str(output_file), n_hops=2)

        # Check output exists
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        # Should have at least header
        assert len(lines) >= 1

        # Check header contains 2hop
        header = lines[0].strip()
        assert '2hop_metapath' in header
        assert '2hop_count' in header

    def test_3hop_backwards_compat(self, linear_chain_graph):
        """Test that 3-hop analysis still works (backwards compatibility)."""
        node_types = load_node_types(linear_chain_graph['nodes_file'])
        matrices = build_matrices(linear_chain_graph['edges_file'], node_types)

        output_file = Path(linear_chain_graph['tmpdir']) / "output_3hop_compat.tsv"
        analyze_3hop_overlap(matrices, str(output_file))

        # Check output exists
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        # Should have header + data
        assert len(lines) >= 2


class TestMetapathFormats:
    """Test metapath format handling across the pipeline."""

    def test_format_parse_roundtrip_1hop(self):
        """Test that format and parse are inverses for 1-hop."""
        # Create a 1-hop metapath
        original = format_metapath(
            ['Drug', 'Disease'],
            ['treats'],
            ['F']
        )

        # Parse it back
        nodes, predicates, directions = parse_metapath(original)

        assert nodes == ['Drug', 'Disease']
        assert predicates == ['treats']
        assert directions == ['F']

    def test_format_parse_roundtrip_3hop(self):
        """Test that format and parse are inverses for 3-hop."""
        # Create a 3-hop metapath
        original = format_metapath(
            ['Drug', 'Gene', 'Protein', 'Disease'],
            ['affects', 'regulates', 'causes'],
            ['F', 'R', 'F']
        )

        # Parse it back
        nodes, predicates, directions = parse_metapath(original)

        assert nodes == ['Drug', 'Gene', 'Protein', 'Disease']
        assert predicates == ['affects', 'regulates', 'causes']
        assert directions == ['F', 'R', 'F']

    def test_symmetric_predicate_handling(self):
        """Test that symmetric predicates get direction 'A'."""
        # interacts_with is symmetric in biolink
        metapath = format_metapath(
            ['Gene', 'Gene'],
            ['interacts_with'],
            ['F']
        )

        # Should have 'A' direction, not 'F'
        nodes, predicates, directions = parse_metapath(metapath)
        assert directions[0] == 'A'
