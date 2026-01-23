"""
Unit tests for load_node_types() with hierarchical types.

Tests that nodes are correctly loaded with multiple types per node.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from analyze_hop_overlap import load_node_types


class TestLoadNodeTypes:
    """Test load_node_types() function with hierarchical types."""

    def create_test_nodes_file(self, nodes_data):
        """Helper to create a temporary nodes file."""
        tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        for node in nodes_data:
            tmpfile.write(json.dumps(node) + '\n')
        tmpfile.flush()
        tmpfile.close()
        return tmpfile.name

    def test_basic_loading(self):
        """Test basic node loading with multiple types."""
        nodes = [
            {
                "id": "NODE:1",
                "category": ["biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:NamedThing"]
            },
            {
                "id": "NODE:2",
                "category": ["biolink:Gene", "biolink:BiologicalEntity"]
            }
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        try:
            result = load_node_types(nodes_file)

            # Check that both nodes are loaded
            assert len(result) == 2
            assert "NODE:1" in result
            assert "NODE:2" in result

            # Check that each node has multiple types
            assert isinstance(result["NODE:1"], list)
            assert isinstance(result["NODE:2"], list)

            # Check that types are present (without biolink: prefix)
            assert 'SmallMolecule' in result["NODE:1"]
            assert 'Gene' in result["NODE:2"]
        finally:
            Path(nodes_file).unlink()

    def test_exclude_types_filtering(self):
        """Test that excluded types are filtered out."""
        nodes = [
            {
                "id": "NODE:1",
                "category": [
                    "biolink:SmallMolecule",
                    "biolink:ChemicalEntity",
                    "biolink:NamedThing",
                    "biolink:PhysicalEssence"
                ]
            }
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        config = {
            'exclude_types': {'NamedThing', 'PhysicalEssence'},
            'max_depth': None,
            'include_most_specific': True
        }

        try:
            result = load_node_types(nodes_file, config=config)

            # Check that excluded types are not present
            assert 'NamedThing' not in result["NODE:1"]
            assert 'PhysicalEssence' not in result["NODE:1"]

            # Check that other types are present
            assert 'SmallMolecule' in result["NODE:1"]
            assert 'ChemicalEntity' in result["NODE:1"]
        finally:
            Path(nodes_file).unlink()

    def test_max_depth_limiting(self):
        """Test that max_depth limits number of types."""
        nodes = [
            {
                "id": "NODE:1",
                "category": [
                    "biolink:SmallMolecule",
                    "biolink:MolecularEntity",
                    "biolink:ChemicalEntity",
                    "biolink:NamedThing"
                ]
            }
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        config = {
            'exclude_types': set(),
            'max_depth': 2,
            'include_most_specific': True
        }

        try:
            result = load_node_types(nodes_file, config=config)

            # Should have at most 2 types
            assert len(result["NODE:1"]) <= 2
            # Most specific should be included
            assert 'SmallMolecule' in result["NODE:1"]
        finally:
            Path(nodes_file).unlink()

    def test_default_config(self):
        """Test loading with default configuration."""
        nodes = [
            {
                "id": "NODE:1",
                "category": [
                    "biolink:SmallMolecule",
                    "biolink:ChemicalEntity",
                    "biolink:ThingWithTaxon",  # Default excluded
                    "biolink:NamedThing"
                ]
            }
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        try:
            # Call without config (should use defaults)
            result = load_node_types(nodes_file)

            # Default config should exclude ThingWithTaxon
            assert 'ThingWithTaxon' not in result["NODE:1"]

            # Other types should be present
            assert 'SmallMolecule' in result["NODE:1"]
        finally:
            Path(nodes_file).unlink()

    def test_node_without_categories(self):
        """Test handling of nodes without category field."""
        nodes = [
            {
                "id": "NODE:1",
                "category": ["biolink:Gene"]
            },
            {
                "id": "NODE:2",
                # No category field
            },
            {
                "id": "NODE:3",
                "category": []
            }
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        try:
            result = load_node_types(nodes_file)

            # Only NODE:1 should be loaded
            assert len(result) == 1
            assert "NODE:1" in result
            assert "NODE:2" not in result
            assert "NODE:3" not in result
        finally:
            Path(nodes_file).unlink()

    def test_type_count_distribution(self, capsys):
        """Test that type distribution statistics are printed."""
        nodes = [
            {"id": "NODE:1", "category": ["biolink:Gene"]},  # 1 type
            {"id": "NODE:2", "category": ["biolink:Disease", "biolink:NamedThing"]},  # 2 types
            {"id": "NODE:3", "category": ["biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:NamedThing"]},  # 3 types
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        config = {
            'exclude_types': set(),  # Don't exclude anything
            'max_depth': None,
            'include_most_specific': True
        }

        try:
            result = load_node_types(nodes_file, config=config)

            # Capture output
            captured = capsys.readouterr()

            # Should print distribution statistics
            assert "Type count distribution" in captured.out
            assert "Average types per node" in captured.out
        finally:
            Path(nodes_file).unlink()

    def test_include_most_specific_always(self):
        """Test that most specific type is always included."""
        nodes = [
            {
                "id": "NODE:1",
                "category": [
                    "biolink:SmallMolecule",
                    "biolink:ChemicalEntity",
                    "biolink:NamedThing"
                ]
            }
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        config = {
            'exclude_types': {'SmallMolecule'},  # Try to exclude most specific
            'max_depth': None,
            'include_most_specific': True
        }

        try:
            result = load_node_types(nodes_file, config=config)

            # Most specific should still be included
            assert 'SmallMolecule' in result["NODE:1"]
        finally:
            Path(nodes_file).unlink()

    def test_ordering_by_specificity(self):
        """Test that types are ordered by specificity (most specific first)."""
        nodes = [
            {
                "id": "NODE:1",
                "category": [
                    "biolink:NamedThing",  # Most abstract
                    "biolink:ChemicalEntity",
                    "biolink:SmallMolecule"  # Most specific
                ]
            }
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        config = {
            'exclude_types': set(),
            'max_depth': None,
            'include_most_specific': True
        }

        try:
            result = load_node_types(nodes_file, config=config)

            # First type should be most specific
            assert result["NODE:1"][0] == 'SmallMolecule'
        finally:
            Path(nodes_file).unlink()

    def test_multiple_nodes_with_same_types(self):
        """Test loading multiple nodes with overlapping types."""
        nodes = [
            {
                "id": "NODE:1",
                "category": ["biolink:Gene", "biolink:BiologicalEntity"]
            },
            {
                "id": "NODE:2",
                "category": ["biolink:Gene", "biolink:BiologicalEntity"]
            },
            {
                "id": "NODE:3",
                "category": ["biolink:Gene", "biolink:BiologicalEntity", "biolink:NamedThing"]
            }
        ]

        nodes_file = self.create_test_nodes_file(nodes)

        try:
            result = load_node_types(nodes_file)

            assert len(result) == 3
            # All should have Gene
            for node_id in ["NODE:1", "NODE:2", "NODE:3"]:
                assert 'Gene' in result[node_id]
        finally:
            Path(nodes_file).unlink()

    def test_real_test_fixtures(self):
        """Test with actual test fixtures if they exist."""
        fixtures_path = Path(__file__).parent / 'fixtures' / 'test_nodes.jsonl'

        if not fixtures_path.exists():
            pytest.skip("Test fixtures not found")

        result = load_node_types(str(fixtures_path))

        # Basic sanity checks
        assert len(result) > 0, "Should load at least some nodes"

        # Check that all values are lists
        for node_id, types in result.items():
            assert isinstance(types, list), f"Types for {node_id} should be a list"
            assert len(types) > 0, f"Node {node_id} should have at least one type"

            # All types should be strings without biolink: prefix
            for type_name in types:
                assert isinstance(type_name, str)
                assert not type_name.startswith('biolink:')

    def test_empty_file(self):
        """Test loading from empty file."""
        tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        tmpfile.close()

        try:
            result = load_node_types(tmpfile.name)
            assert len(result) == 0
        finally:
            Path(tmpfile.name).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
