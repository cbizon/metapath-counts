"""Tests for group_by_onehop.py functions."""

import pytest
import math
import sys
from pathlib import Path

# Add scripts to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from metapath_counts import (
    parse_metapath,
    build_metapath,
    calculate_metrics,
)
from group_by_onehop import (
    normalize_1hop,
    reverse_metapath,
    reverse_3hop,
    get_endpoint_types,
    get_canonical_direction,
    canonicalize_row,
    safe_filename,
)


class TestParseMetapath:
    """Tests for parse_metapath function."""

    def test_parse_1hop(self):
        """Test parsing a 1-hop metapath."""
        nodes, predicates, directions = parse_metapath("Drug|treats|F|Disease")

        assert nodes == ['Drug', 'Disease']
        assert predicates == ['treats']
        assert directions == ['F']

    def test_parse_1hop_reverse(self):
        """Test parsing a 1-hop reverse metapath."""
        nodes, predicates, directions = parse_metapath("Disease|treats|R|Drug")

        assert nodes == ['Disease', 'Drug']
        assert predicates == ['treats']
        assert directions == ['R']

    def test_parse_1hop_symmetric(self):
        """Test parsing a 1-hop metapath with symmetric predicate."""
        nodes, predicates, directions = parse_metapath("Gene|interacts_with|A|Gene")

        assert nodes == ['Gene', 'Gene']
        assert predicates == ['interacts_with']
        assert directions == ['A']

    def test_parse_2hop(self):
        """Test parsing a 2-hop metapath."""
        nodes, predicates, directions = parse_metapath(
            "Drug|affects|F|Gene|causes|R|Disease"
        )

        assert nodes == ['Drug', 'Gene', 'Disease']
        assert predicates == ['affects', 'causes']
        assert directions == ['F', 'R']

    def test_parse_3hop(self):
        """Test parsing a 3-hop metapath."""
        nodes, predicates, directions = parse_metapath(
            "Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease"
        )

        assert nodes == ['Drug', 'Gene', 'Protein', 'Disease']
        assert predicates == ['affects', 'regulates', 'causes']
        assert directions == ['F', 'R', 'F']

    def test_parse_4hop(self):
        """Test parsing a 4-hop metapath."""
        nodes, predicates, directions = parse_metapath(
            "A|p1|F|B|p2|R|C|p3|A|D|p4|F|E"
        )

        assert nodes == ['A', 'B', 'C', 'D', 'E']
        assert predicates == ['p1', 'p2', 'p3', 'p4']
        assert directions == ['F', 'R', 'A', 'F']

    def test_parse_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid metapath format"):
            parse_metapath("Drug|treats|Disease")  # Missing direction (3 parts)

        with pytest.raises(ValueError, match="Invalid metapath format"):
            parse_metapath("Drug")  # Too short (1 part)

        with pytest.raises(ValueError, match="Invalid metapath format"):
            parse_metapath("A|B|C|D|E")  # Wrong structure (5 parts)


class TestBuildMetapath:
    """Tests for build_metapath function."""

    def test_build_1hop(self):
        """Test building a 1-hop metapath."""
        result = build_metapath(['Drug', 'Disease'], ['treats'], ['F'])
        assert result == "Drug|treats|F|Disease"

    def test_build_3hop(self):
        """Test building a 3-hop metapath."""
        result = build_metapath(
            ['Drug', 'Gene', 'Protein', 'Disease'],
            ['affects', 'regulates', 'causes'],
            ['F', 'R', 'F']
        )
        assert result == "Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease"

    def test_roundtrip_1hop(self):
        """Test that parsing and building gives back original."""
        original = "Drug|treats|F|Disease"
        nodes, predicates, directions = parse_metapath(original)
        rebuilt = build_metapath(nodes, predicates, directions)
        assert rebuilt == original

    def test_roundtrip_3hop(self):
        """Test roundtrip for 3-hop metapath."""
        original = "Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease"
        nodes, predicates, directions = parse_metapath(original)
        rebuilt = build_metapath(nodes, predicates, directions)
        assert rebuilt == original


class TestNormalize1hop:
    """Tests for normalize_1hop function."""

    def test_normalize_forward(self):
        """Test that forward direction is unchanged."""
        result, was_reversed = normalize_1hop("Drug|treats|F|Disease")

        assert result == "Drug|treats|F|Disease"
        assert was_reversed == False

    def test_normalize_reverse(self):
        """Test that reverse direction is normalized to forward."""
        result, was_reversed = normalize_1hop("Disease|treats|R|Drug")

        assert result == "Drug|treats|F|Disease"
        assert was_reversed == True

    def test_normalize_any(self):
        """Test that 'any' direction is unchanged."""
        result, was_reversed = normalize_1hop("Gene|interacts_with|A|Gene")

        assert result == "Gene|interacts_with|A|Gene"
        assert was_reversed == False


class TestReverseMetapath:
    """Tests for reverse_metapath function."""

    def test_reverse_1hop(self):
        """Test reversing a 1-hop metapath."""
        original = "A|p1|F|B"
        reversed_mp = reverse_metapath(original)

        assert reversed_mp == "B|p1|R|A"

    def test_reverse_2hop(self):
        """Test reversing a 2-hop metapath."""
        original = "A|p1|F|B|p2|R|C"
        reversed_mp = reverse_metapath(original)

        assert reversed_mp == "C|p2|F|B|p1|R|A"

    def test_reverse_3hop_simple(self):
        """Test reversing a simple 3-hop metapath."""
        original = "A|p1|F|B|p2|F|C|p3|F|D"
        reversed_mp = reverse_metapath(original)

        assert reversed_mp == "D|p3|R|C|p2|R|B|p1|R|A"

    def test_reverse_3hop_mixed_directions(self):
        """Test reversing a 3-hop metapath with mixed directions."""
        original = "A|p1|F|B|p2|R|C|p3|F|D"
        reversed_mp = reverse_metapath(original)

        # F->R, R->F
        assert reversed_mp == "D|p3|R|C|p2|F|B|p1|R|A"

    def test_reverse_with_any(self):
        """Test reversing metapath with 'A' direction."""
        original = "A|p1|A|B|p2|F|C|p3|R|D"
        reversed_mp = reverse_metapath(original)

        # A stays A
        assert reversed_mp == "D|p3|F|C|p2|R|B|p1|A|A"

    def test_reverse_4hop(self):
        """Test reversing a 4-hop metapath."""
        original = "A|p1|F|B|p2|R|C|p3|A|D|p4|F|E"
        reversed_mp = reverse_metapath(original)

        assert reversed_mp == "E|p4|R|D|p3|A|C|p2|F|B|p1|R|A"

    def test_double_reverse_identity(self):
        """Test that reversing twice gives back original for any N-hop."""
        for original in [
            "A|p1|F|B",  # 1-hop
            "A|p1|F|B|p2|R|C",  # 2-hop
            "Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease",  # 3-hop
            "A|p1|F|B|p2|R|C|p3|A|D|p4|F|E",  # 4-hop
        ]:
            reversed_once = reverse_metapath(original)
            reversed_twice = reverse_metapath(reversed_once)
            assert reversed_twice == original

    def test_reverse_3hop_backwards_compat(self):
        """Test that reverse_3hop still works (backwards compatibility)."""
        original = "Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease"
        reversed_mp = reverse_3hop(original)

        assert reversed_mp == "Disease|causes|R|Protein|regulates|F|Gene|affects|R|Drug"


class TestGetEndpointTypes:
    """Tests for get_endpoint_types function."""

    def test_endpoints_1hop(self):
        """Test getting endpoints from 1-hop metapath."""
        start, end = get_endpoint_types("Drug|treats|F|Disease")

        assert start == "Drug"
        assert end == "Disease"

    def test_endpoints_3hop(self):
        """Test getting endpoints from 3-hop metapath."""
        start, end = get_endpoint_types("Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease")

        assert start == "Drug"
        assert end == "Disease"

    def test_endpoints_same_type(self):
        """Test getting endpoints when they're the same type."""
        start, end = get_endpoint_types("Gene|interacts|A|Gene")

        assert start == "Gene"
        assert end == "Gene"


class TestGetCanonicalDirection:
    """Tests for get_canonical_direction function."""

    def test_alphabetical_order_correct(self):
        """Test when types are already in alphabetical order."""
        start, end = get_canonical_direction("Disease", "Drug")

        assert start == "Disease"
        assert end == "Drug"

    def test_alphabetical_order_reversed(self):
        """Test when types need to be swapped for alphabetical order."""
        start, end = get_canonical_direction("Drug", "Disease")

        assert start == "Disease"
        assert end == "Drug"

    def test_same_type(self):
        """Test when both types are the same."""
        start, end = get_canonical_direction("Gene", "Gene")

        assert start == "Gene"
        assert end == "Gene"


class TestCanonicalizeRow:
    """Tests for canonicalize_row function."""

    def test_already_canonical(self):
        """Test row that's already in canonical order."""
        threehop = "Disease|p1|F|Gene|p2|F|Protein|p3|F|Drug"
        onehop = "Disease|treats|F|Drug"

        canon_3hop, canon_1hop = canonicalize_row(threehop, onehop)

        assert canon_3hop == threehop
        assert canon_1hop == onehop

    def test_needs_reversal(self):
        """Test row that needs reversal for canonical order."""
        threehop = "Drug|p1|F|Protein|p2|F|Gene|p3|F|Disease"
        onehop = "Drug|treats|F|Disease"

        canon_3hop, canon_1hop = canonicalize_row(threehop, onehop)

        # Should be reversed because Disease < Drug alphabetically
        assert canon_3hop.startswith("Disease|")
        assert canon_3hop.endswith("|Drug")
        assert canon_1hop.startswith("Disease|")
        assert canon_1hop.endswith("|Drug")

    def test_same_endpoints(self):
        """Test when both endpoints are the same type."""
        threehop = "Gene|p1|F|Drug|p2|F|Protein|p3|F|Gene"
        onehop = "Gene|interacts|A|Gene"

        canon_3hop, canon_1hop = canonicalize_row(threehop, onehop)

        # Same type, so no reversal needed
        assert canon_3hop == threehop
        assert canon_1hop == onehop


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_perfect_prediction(self):
        """Test metrics when prediction is perfect (all overlap)."""
        # 100 3-hop paths, all 100 are 1-hop, 100 overlap
        metrics = calculate_metrics(
            nhop_count=100,
            onehop_count=100,
            overlap=100,
            total_possible=1000,
            full_metrics=True
        )

        assert metrics['TP'] == 100
        assert metrics['FP'] == 0
        assert metrics['FN'] == 0
        assert metrics['TN'] == 900
        assert metrics['Precision'] == 1.0
        assert metrics['Recall'] == 1.0
        assert metrics['F1'] == 1.0

    def test_no_overlap(self):
        """Test metrics when there's no overlap."""
        metrics = calculate_metrics(
            nhop_count=100,
            onehop_count=100,
            overlap=0,
            total_possible=1000,
            full_metrics=True
        )

        assert metrics['TP'] == 0
        assert metrics['FP'] == 100
        assert metrics['FN'] == 100
        assert metrics['TN'] == 800
        assert metrics['Precision'] == 0.0
        assert metrics['Recall'] == 0.0
        assert metrics['F1'] == 0.0

    def test_partial_overlap(self):
        """Test metrics with partial overlap."""
        metrics = calculate_metrics(
            nhop_count=100,
            onehop_count=50,
            overlap=25,
            total_possible=1000,
            full_metrics=True
        )

        assert metrics['TP'] == 25
        assert metrics['FP'] == 75
        assert metrics['FN'] == 25
        assert metrics['TN'] == 875
        assert metrics['Total'] == 1000

        # Precision = TP / (TP + FP) = 25 / 100 = 0.25
        assert abs(metrics['Precision'] - 0.25) < 0.0001

        # Recall = TP / (TP + FN) = 25 / 50 = 0.5
        assert abs(metrics['Recall'] - 0.5) < 0.0001

    def test_mcc_calculation(self):
        """Test MCC calculation."""
        # Use a case where MCC can be verified
        metrics = calculate_metrics(
            nhop_count=100,
            onehop_count=100,
            overlap=80,
            total_possible=1000,
            full_metrics=True
        )

        # TP=80, FP=20, FN=20, TN=880
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        # MCC = (80*880 - 20*20) / sqrt(100 * 100 * 900 * 900)
        # MCC = (70400 - 400) / sqrt(8100000000)
        # MCC = 70000 / 90000 = 0.7778

        assert abs(metrics['MCC'] - 0.7778) < 0.001

    def test_division_by_zero_protection(self):
        """Test that division by zero is handled gracefully."""
        # All zeros
        metrics = calculate_metrics(
            nhop_count=0,
            onehop_count=0,
            overlap=0,
            total_possible=0,
            full_metrics=True
        )

        # Should not raise, should return 0 for metrics
        assert metrics['Precision'] == 0.0
        assert metrics['Recall'] == 0.0
        assert metrics['F1'] == 0.0
        assert metrics['Accuracy'] == 0.0


class TestSafeFilename:
    """Tests for safe_filename function."""

    def test_basic_metapath(self):
        """Test converting a basic metapath to filename."""
        result = safe_filename("Drug|treats|F|Disease")
        assert result == "Drug_treats_F_Disease.tsv"

    def test_special_characters(self):
        """Test that special characters are replaced."""
        result = safe_filename("Gene|has:role|F|Pathway")
        # Colon should be replaced with underscore
        assert ":" not in result
        assert result.endswith(".tsv")

    def test_long_metapath(self):
        """Test with a long 3-hop metapath."""
        result = safe_filename("Drug|affects|F|Gene|regulates|R|Protein|causes|F|Disease")
        expected = "Drug_affects_F_Gene_regulates_R_Protein_causes_F_Disease.tsv"
        assert result == expected
