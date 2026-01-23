"""
Unit tests for hierarchical type utilities.

Tests get_all_types() and filter_abstract_types() functions.
"""

import pytest
from metapath_counts import get_all_types, filter_abstract_types


class TestFilterAbstractTypes:
    """Test filter_abstract_types() function."""

    def test_basic_filtering(self):
        """Test basic type filtering."""
        types = ['SmallMolecule', 'ChemicalEntity', 'NamedThing']
        exclude = ['NamedThing', 'PhysicalEssence']
        result = filter_abstract_types(types, exclude)
        assert result == ['SmallMolecule', 'ChemicalEntity']

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        types = ['SmallMolecule', 'ChemicalEntity', 'NamedThing']
        exclude = ['namedthing', 'PHYSICALESSENCE']
        result = filter_abstract_types(types, exclude)
        assert result == ['SmallMolecule', 'ChemicalEntity']

    def test_with_biolink_prefix(self):
        """Test filtering with biolink: prefix."""
        types = ['biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:NamedThing']
        exclude = ['NamedThing']
        result = filter_abstract_types(types, exclude)
        assert result == ['biolink:SmallMolecule', 'biolink:ChemicalEntity']

    def test_mixed_format(self):
        """Test mixed prefix formats."""
        types = ['biolink:SmallMolecule', 'ChemicalEntity', 'biolink:NamedThing']
        exclude = ['biolink:NamedThing']
        result = filter_abstract_types(types, exclude)
        assert result == ['biolink:SmallMolecule', 'ChemicalEntity']

    def test_empty_input(self):
        """Test empty type list."""
        result = filter_abstract_types([], ['NamedThing'])
        assert result == []

    def test_empty_exclude(self):
        """Test empty exclude list."""
        types = ['SmallMolecule', 'ChemicalEntity']
        result = filter_abstract_types(types, [])
        assert result == types

    def test_no_matches(self):
        """Test when no types match exclude list."""
        types = ['SmallMolecule', 'ChemicalEntity']
        exclude = ['NamedThing', 'PhysicalEssence']
        result = filter_abstract_types(types, exclude)
        assert result == types

    def test_all_filtered(self):
        """Test when all types are filtered."""
        types = ['NamedThing', 'PhysicalEssence']
        exclude = ['NamedThing', 'PhysicalEssence']
        result = filter_abstract_types(types, exclude)
        assert result == []

    def test_preserves_order(self):
        """Test that original order is preserved."""
        types = ['SmallMolecule', 'MolecularEntity', 'ChemicalEntity', 'NamedThing']
        exclude = ['NamedThing']
        result = filter_abstract_types(types, exclude)
        assert result == ['SmallMolecule', 'MolecularEntity', 'ChemicalEntity']

    def test_set_as_exclude(self):
        """Test using set as exclude list."""
        types = ['SmallMolecule', 'ChemicalEntity', 'NamedThing']
        exclude = {'NamedThing', 'PhysicalEssence'}
        result = filter_abstract_types(types, exclude)
        assert result == ['SmallMolecule', 'ChemicalEntity']


class TestGetAllTypes:
    """Test get_all_types() function."""

    def test_all_types_no_filtering(self):
        """Test returning all types without filtering."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:NamedThing']
        result = get_all_types(categories)
        # Should return all, ordered by specificity (most specific first)
        assert 'SmallMolecule' in result
        assert 'ChemicalEntity' in result
        assert 'NamedThing' in result
        assert len(result) == 3

    def test_with_exclude_types(self):
        """Test excluding abstract types."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:NamedThing']
        exclude = {'NamedThing', 'PhysicalEssence'}
        result = get_all_types(categories, exclude_types=exclude)
        assert 'SmallMolecule' in result
        assert 'ChemicalEntity' in result
        assert 'NamedThing' not in result

    def test_max_depth_limit(self):
        """Test limiting hierarchy depth."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:NamedThing']
        result = get_all_types(categories, max_depth=2)
        # Should return only 2 most specific types
        assert len(result) == 2
        assert 'SmallMolecule' in result

    def test_include_most_specific_when_excluded(self):
        """Test that most specific is included even if excluded."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:NamedThing']
        exclude = {'SmallMolecule'}
        result = get_all_types(
            categories,
            exclude_types=exclude,
            include_most_specific=True
        )
        # SmallMolecule should be included despite being in exclude list
        assert 'SmallMolecule' in result
        assert 'ChemicalEntity' in result

    def test_dont_include_most_specific_when_excluded(self):
        """Test excluding most specific when include_most_specific=False."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:NamedThing']
        exclude = {'SmallMolecule'}
        result = get_all_types(
            categories,
            exclude_types=exclude,
            include_most_specific=False
        )
        # SmallMolecule should not be included
        assert 'SmallMolecule' not in result
        assert 'ChemicalEntity' in result
        assert 'NamedThing' in result

    def test_empty_categories(self):
        """Test with empty category list."""
        result = get_all_types([])
        assert result == []

    def test_single_type(self):
        """Test with single type."""
        categories = ['biolink:SmallMolecule']
        result = get_all_types(categories)
        assert result == ['SmallMolecule']

    def test_ordering_by_specificity(self):
        """Test that types are ordered by specificity (most specific first)."""
        categories = [
            'biolink:NamedThing',
            'biolink:ChemicalEntity',
            'biolink:SmallMolecule',
            'biolink:MolecularEntity'
        ]
        result = get_all_types(categories)
        # Most specific should be first
        assert result[0] == 'SmallMolecule'
        # Most abstract should be last
        assert result[-1] == 'NamedThing'

    def test_complex_hierarchy_with_filtering(self):
        """Test complex hierarchy with multiple filters."""
        categories = [
            'biolink:SmallMolecule',
            'biolink:MolecularEntity',
            'biolink:ChemicalEntity',
            'biolink:PhysicalEssence',
            'biolink:NamedThing'
        ]
        exclude = {'PhysicalEssence', 'NamedThing'}
        result = get_all_types(
            categories,
            exclude_types=exclude,
            max_depth=3
        )
        assert len(result) <= 3
        assert 'SmallMolecule' in result
        assert 'PhysicalEssence' not in result
        assert 'NamedThing' not in result

    def test_max_depth_one(self):
        """Test max_depth=1 returns only most specific."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:NamedThing']
        result = get_all_types(categories, max_depth=1)
        assert len(result) == 1
        assert result[0] == 'SmallMolecule'

    def test_no_duplicate_types(self):
        """Test that result has no duplicates."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity']
        result = get_all_types(categories)
        assert len(result) == len(set(result))

    def test_default_exclude_types(self):
        """Test with default configuration."""
        # Default config should be applied
        categories = [
            'biolink:SmallMolecule',
            'biolink:ChemicalEntity',
            'biolink:NamedThing',
            'biolink:ThingWithTaxon'
        ]
        # When no config provided, it uses internal defaults
        result = get_all_types(categories)
        # Should include all types since no exclude specified
        assert len(result) >= 3

    def test_biolink_prefix_handling(self):
        """Test that biolink: prefix is removed from output."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity']
        result = get_all_types(categories)
        for type_name in result:
            assert not type_name.startswith('biolink:')

    def test_invalid_types_skipped(self):
        """Test that invalid types are skipped."""
        categories = ['biolink:SmallMolecule', 'biolink:InvalidType123', 'biolink:ChemicalEntity']
        result = get_all_types(categories)
        # Invalid type should be skipped
        assert 'InvalidType123' not in result
        assert 'SmallMolecule' in result

    def test_multiple_hierarchies(self):
        """Test node with types from different hierarchy branches."""
        # Protein has different hierarchy than SmallMolecule
        categories = [
            'biolink:Protein',
            'biolink:Polypeptide',
            'biolink:ChemicalEntityOrProteinOrPolypeptide',
            'biolink:BiologicalEntity'
        ]
        result = get_all_types(categories)
        assert 'Protein' in result
        assert len(result) >= 2


class TestGetAllTypesEdgeCases:
    """Test edge cases for get_all_types()."""

    def test_none_config(self):
        """Test with None config."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity']
        result = get_all_types(categories, exclude_types=None)
        assert len(result) >= 2

    def test_max_depth_zero(self):
        """Test max_depth=0 with include_most_specific."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity']
        # With include_most_specific=True (default), should still include most specific
        result = get_all_types(categories, max_depth=0)
        assert len(result) >= 1
        assert 'SmallMolecule' in result

        # With include_most_specific=False, should return empty
        result = get_all_types(categories, max_depth=0, include_most_specific=False)
        assert result == []

    def test_max_depth_exceeds_hierarchy(self):
        """Test max_depth larger than actual hierarchy."""
        categories = ['biolink:SmallMolecule', 'biolink:ChemicalEntity']
        result = get_all_types(categories, max_depth=100)
        # Should return all available types
        assert len(result) >= 2

    def test_combined_filters(self):
        """Test combining exclude_types, max_depth, and include_most_specific."""
        categories = [
            'biolink:SmallMolecule',
            'biolink:ChemicalEntity',
            'biolink:NamedThing',
            'biolink:PhysicalEssence'
        ]
        result = get_all_types(
            categories,
            exclude_types={'NamedThing', 'SmallMolecule'},
            max_depth=2,
            include_most_specific=True
        )
        # SmallMolecule should be included despite being excluded
        assert 'SmallMolecule' in result
        # NamedThing should be excluded
        assert 'NamedThing' not in result
        # Should respect max_depth
        assert len(result) <= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
