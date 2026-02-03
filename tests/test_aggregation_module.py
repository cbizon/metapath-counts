#!/usr/bin/env python3
"""
Tests for the aggregation module (src/metapath_counts/aggregation.py).

Tests the core functions for metapath parsing, building, and variant generation.
"""

import pytest
from metapath_counts import (
    parse_metapath,
    build_metapath,
    get_type_variants,
    get_predicate_variants,
    generate_metapath_variants,
    expand_metapath_to_variants,
    calculate_metrics
)


class TestParseMetapath:
    """Tests for parse_metapath function."""

    def test_parse_1hop(self):
        """Test parsing a 1-hop metapath."""
        nodes, predicates, directions = parse_metapath("Gene|affects|F|Disease")
        assert nodes == ["Gene", "Disease"]
        assert predicates == ["affects"]
        assert directions == ["F"]

    def test_parse_2hop(self):
        """Test parsing a 2-hop metapath."""
        nodes, predicates, directions = parse_metapath("Gene|affects|F|Disease|treats|R|SmallMolecule")
        assert nodes == ["Gene", "Disease", "SmallMolecule"]
        assert predicates == ["affects", "treats"]
        assert directions == ["F", "R"]

    def test_parse_3hop(self):
        """Test parsing a 3-hop metapath."""
        nodes, predicates, directions = parse_metapath("A|p1|F|B|p2|R|C|p3|A|D")
        assert nodes == ["A", "B", "C", "D"]
        assert predicates == ["p1", "p2", "p3"]
        assert directions == ["F", "R", "A"]

    def test_parse_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_metapath("Gene|affects")  # Too few parts

        with pytest.raises(ValueError):
            parse_metapath("Gene|affects|F")  # Missing target type


class TestBuildMetapath:
    """Tests for build_metapath function."""

    def test_build_1hop(self):
        """Test building a 1-hop metapath."""
        result = build_metapath(["Gene", "Disease"], ["affects"], ["F"])
        assert result == "Gene|affects|F|Disease"

    def test_build_2hop(self):
        """Test building a 2-hop metapath."""
        result = build_metapath(
            ["Gene", "Disease", "SmallMolecule"],
            ["affects", "treats"],
            ["F", "R"]
        )
        assert result == "Gene|affects|F|Disease|treats|R|SmallMolecule"

    def test_roundtrip(self):
        """Test that parsing and building are inverses."""
        original = "Gene|affects|F|Disease|treats|R|SmallMolecule"
        nodes, predicates, directions = parse_metapath(original)
        rebuilt = build_metapath(nodes, predicates, directions)
        assert rebuilt == original


class TestGetTypeVariants:
    """Tests for get_type_variants function."""

    def test_regular_type_includes_self(self):
        """Regular type should include itself."""
        result = get_type_variants("SmallMolecule")
        assert "SmallMolecule" in result

    def test_regular_type_includes_ancestors(self):
        """Regular type should include ancestors."""
        result = get_type_variants("SmallMolecule")
        assert "ChemicalEntity" in result

    def test_exclude_self(self):
        """Can exclude self from variants."""
        result = get_type_variants("SmallMolecule", include_self=False)
        # Ancestors should still be there
        assert "ChemicalEntity" in result

    def test_pseudo_type_includes_constituents(self):
        """Pseudo-type should include constituent types."""
        result = get_type_variants("Gene+SmallMolecule")
        assert "Gene+SmallMolecule" in result
        assert "Gene" in result
        assert "SmallMolecule" in result


class TestGetPredicateVariants:
    """Tests for get_predicate_variants function."""

    def test_predicate_includes_self(self):
        """Predicate should include itself."""
        result = get_predicate_variants("treats")
        assert "treats" in result

    def test_predicate_includes_ancestors(self):
        """Predicate should include ancestors."""
        result = get_predicate_variants("treats")
        assert "related_to" in result

    def test_strips_biolink_prefix(self):
        """Should handle biolink: prefix."""
        result = get_predicate_variants("biolink:treats")
        assert "treats" in result
        # Should not have biolink: prefix in results
        assert not any(v.startswith("biolink:") for v in result)


class TestGenerateMetapathVariants:
    """Tests for generate_metapath_variants function."""

    def test_generates_original(self):
        """Should include the original metapath."""
        variants = list(generate_metapath_variants("Gene|affects|F|Disease"))
        assert "Gene|affects|F|Disease" in variants

    def test_generates_type_ancestors(self):
        """Should include type ancestor variants."""
        variants = list(generate_metapath_variants("Gene|affects|F|Disease"))
        # Should have BiologicalEntity (ancestor of Gene)
        assert any("BiologicalEntity" in v for v in variants)

    def test_generates_predicate_ancestors(self):
        """Should include predicate ancestor variants."""
        variants = list(generate_metapath_variants("Gene|affects|F|Disease"))
        # Should have related_to (ancestor of affects)
        assert any("related_to" in v for v in variants)

    def test_generates_many_variants(self):
        """Should generate multiple variants."""
        variants = list(generate_metapath_variants("Gene|affects|F|Disease"))
        # Should have more than just the original
        assert len(variants) > 10


class TestExpandMetapathToVariants:
    """Tests for expand_metapath_to_variants function."""

    def test_returns_set(self):
        """Should return a set."""
        result = expand_metapath_to_variants("Gene|affects|F|Disease")
        assert isinstance(result, set)

    def test_contains_original(self):
        """Should contain the original metapath."""
        result = expand_metapath_to_variants("Gene|affects|F|Disease")
        assert "Gene|affects|F|Disease" in result

    def test_same_as_set_of_generator(self):
        """Should be equivalent to set of generate_metapath_variants."""
        metapath = "Gene|affects|F|Disease"
        from_expand = expand_metapath_to_variants(metapath)
        from_generator = set(generate_metapath_variants(metapath))
        assert from_expand == from_generator


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_basic_metrics(self):
        """Test basic 6-metric output."""
        metrics = calculate_metrics(1000, 500, 100, 1_000_000)

        # Should have 6 keys
        assert len(metrics) == 6
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "mcc" in metrics
        assert "specificity" in metrics
        assert "npv" in metrics

    def test_full_metrics(self):
        """Test full 18-metric output."""
        metrics = calculate_metrics(1000, 500, 100, 1_000_000, full_metrics=True)

        # Should have 18 keys
        assert len(metrics) == 18
        assert "TP" in metrics
        assert "Precision" in metrics
        assert "MCC" in metrics
        assert "PLR" in metrics

    def test_precision_calculation(self):
        """Precision = overlap / nhop_count."""
        metrics = calculate_metrics(1000, 500, 100, 1_000_000)
        # Precision = 100/1000 = 0.1
        assert abs(metrics["precision"] - 0.1) < 0.0001

    def test_recall_calculation(self):
        """Recall = overlap / onehop_count."""
        metrics = calculate_metrics(1000, 500, 100, 1_000_000)
        # Recall = 100/500 = 0.2
        assert abs(metrics["recall"] - 0.2) < 0.0001

    def test_f1_calculation(self):
        """F1 = 2 * precision * recall / (precision + recall)."""
        metrics = calculate_metrics(1000, 500, 100, 1_000_000)
        # Precision = 0.1, Recall = 0.2
        # F1 = 2 * 0.1 * 0.2 / 0.3 = 0.04 / 0.3 = 0.1333
        assert abs(metrics["f1"] - 0.1333) < 0.001

    def test_division_by_zero(self):
        """Should handle division by zero gracefully."""
        metrics = calculate_metrics(0, 0, 0, 0)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_negative_tn_handling(self):
        """Should handle negative TN from aggregation errors."""
        # This can happen when aggregated counts exceed total_possible
        # nhop_count + onehop_count > total_possible + overlap
        metrics = calculate_metrics(800, 500, 100, 1000)
        # TN = 1000 - 800 - 500 + 100 = -200 (negative!)

        # Specificity and NPV should be 0 when TN is negative
        assert metrics["specificity"] == 0.0
        assert metrics["npv"] == 0.0

    def test_perfect_classifier(self):
        """Test metrics for perfect classifier."""
        metrics = calculate_metrics(100, 100, 100, 1000, full_metrics=True)

        assert metrics["Precision"] == 1.0
        assert metrics["Recall"] == 1.0
        assert metrics["F1"] == 1.0

    def test_confusion_matrix_values(self):
        """Test that confusion matrix values are correct."""
        metrics = calculate_metrics(100, 50, 25, 1000, full_metrics=True)

        # TP = overlap = 25
        assert metrics["TP"] == 25
        # FP = nhop_count - overlap = 100 - 25 = 75
        assert metrics["FP"] == 75
        # FN = onehop_count - overlap = 50 - 25 = 25
        assert metrics["FN"] == 25
        # TN = total_possible - nhop_count - onehop_count + overlap = 1000 - 100 - 50 + 25 = 875
        assert metrics["TN"] == 875
        # Total should be total_possible
        assert metrics["Total"] == 1000
