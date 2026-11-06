"""Unit tests for the validation checks (grounding, survey direction, recovery).

Run: pytest -q tests/test_validation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import validation as V  # noqa: E402


def test_chi_square_fit_passes_when_counts_match_shares():
    observed = pd.Series({"a": 50, "b": 30, "c": 20})
    expected = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    stat, p = V.chi_square_fit(observed, expected)
    assert stat == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_chi_square_fit_fails_when_counts_differ():
    observed = pd.Series({"a": 90, "b": 5, "c": 5})
    expected = pd.Series({"a": 0.3, "b": 0.4, "c": 0.3})
    _, p = V.chi_square_fit(observed, expected)
    assert p < 0.001


def test_chi_square_fit_merges_sparse_categories():
    observed = pd.Series({"a": 60, "b": 38, "c": 1, "d": 1})
    expected = pd.Series({"a": 0.6, "b": 0.38, "c": 0.01, "d": 0.01})
    stat, _ = V.chi_square_fit(observed, expected)
    assert np.isfinite(stat)


def test_size_bins_fold_large_baskets():
    assert V.size_bins(pd.Series([1, 2, 3, 4, 6])).tolist() == ["1", "2", "3", "4+", "4+"]


def test_pair_stats_confidence_and_lift():
    baskets = pd.Series([["Chai", "Pakora"], ["Chai"], ["Pakora"], ["Coffee"]])
    stats = V.pair_stats(baskets, "Chai", "Pakora")
    assert stats["confidence"] == pytest.approx(0.5)
    assert stats["lift"] == pytest.approx(0.25 / (0.5 * 0.5))
    assert stats["trigger_orders"] == 2


def test_pair_stats_matches_biryani_by_name():
    baskets = pd.Series([["Chicken Dum Biryani", "Buttermilk"], ["Plain Biryani"]])
    assert V.pair_stats(baskets, "Biryani", "Buttermilk")["confidence"] == pytest.approx(0.5)


def test_direction_check_compares_signs():
    assert V.same_direction(1.3, "more")
    assert V.same_direction(0.8, "less")
    assert not V.same_direction(1.3, "less")
    assert V.same_direction(1.0, "same")


def test_survey_majority_ignores_same_answers():
    assert V.survey_majority({"less": 18, "same": 5, "more": 4}) == "less"
    assert V.survey_majority({"less": 5, "same": 15, "more": 7}) == "more"
    assert V.survey_majority({"less": 3, "same": 9, "more": 3}) == "same"


def test_ordering_matches_respects_ties_within_tolerance():
    assert V.ordering_matches({"rainy": 0.30, "normal": 0.22}, {"rainy": 0.33, "normal": 0.25})
    assert not V.ordering_matches({"rainy": 0.30, "normal": 0.22}, {"rainy": 0.20, "normal": 0.25})
    assert V.ordering_matches({"a": 0.3, "b": 0.3}, {"a": 0.31, "b": 0.30}, tol=0.02)


def test_summary_row_has_fixed_columns():
    row = V.check_row("grounding", "hours", "same as Swiggy", "p=0.4", True, "chi-square")
    assert list(row) == V.SUMMARY_COLUMNS
