"""Tests for src/eda.py (T19 & T20): verify EDA metrics, data integrity, and chart exports."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import eda  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def eda_data():
    return eda.load_eda_data(ROOT)


def test_eda_data_loading_integrity(eda_data):
    daily = eda_data["daily"]
    orders = eda_data["orders"]
    menu = eda_data["menu"]

    # Verify counts match documented dataset
    assert len(daily) == 272
    assert len(orders) == 53820
    assert len(menu) == 96

    # Ensure no nulls in critical columns
    assert daily["order_count"].isna().sum() == 0
    assert daily["total_revenue"].isna().sum() == 0
    assert orders["order_value"].isna().sum() == 0
    assert orders["items"].isna().sum() == 0


def test_order_distribution_stats(eda_data):
    stats = eda.get_order_distribution_stats(eda_data["daily"])
    assert stats["count"] == 272
    assert stats["mean"] == pytest.approx(197.9, abs=0.5)
    assert stats["median"] == 194.0
    assert stats["min"] == 126
    assert stats["max"] == 282


def test_basket_and_value_stats(eda_data):
    stats = eda.get_basket_and_value_stats(eda_data["orders"])
    assert stats["total_orders"] == 53820
    assert stats["mean_basket_size"] == pytest.approx(1.15, abs=0.05)
    assert stats["mean_order_value"] == pytest.approx(25260.8, abs=5.0)
    assert stats["median_order_value"] == 23000.0


def test_weekday_stats(eda_data):
    stats = eda.get_weekday_stats(eda_data["daily"])
    assert len(stats) == 7
    assert stats["mean"].min() > 100


def test_rain_vs_dry_stats(eda_data):
    stats = eda.get_rain_vs_dry_stats(eda_data["daily"])
    assert "rainy_mean" in stats and "dry_mean" in stats
    assert "t_pvalue" in stats and "mwu_pvalue" in stats
    assert stats["rain_category_summary"]["days"].sum() == 272


def test_item_and_category_shifts(eda_data):
    item_shift, cat_regime = eda.get_item_and_category_shifts(
        eda_data["orders"], eda_data["daily"], eda_data["menu"]
    )
    assert len(item_shift) > 0
    assert len(cat_regime) > 0
    assert "relative_lift_pct" in item_shift.columns
    assert set(cat_regime.index).issubset(set(eda_data["menu"]["category"].unique()))


def test_correlation_matrix(eda_data):
    pearson, spearman = eda.get_correlation_matrix(eda_data["daily"])
    # Revenue should be strongly positively correlated with order count
    assert pearson.loc["total_revenue", "order_count"] > 0.90
    assert spearman.loc["total_revenue", "order_count"] > 0.90
