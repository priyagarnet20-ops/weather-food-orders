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
    daily, orders, menu = eda_data["daily"], eda_data["orders"], eda_data["menu"]
    assert len(daily) == 1097
    assert len(orders) == daily["order_count"].sum()
    assert len(menu) == 80  # 70 real Chinthamani items + 10 sourced snacks and drinks
    assert daily["order_count"].isna().sum() == 0
    assert daily["total_revenue"].isna().sum() == 0
    assert orders["order_value"].isna().sum() == 0
    assert orders["items"].isna().sum() == 0


def test_order_distribution_stats(eda_data):
    stats = eda.get_order_distribution_stats(eda_data["daily"])
    assert stats["count"] == 1097
    expected = eda_data["daily"]["expected_orders"].mean()
    assert stats["mean"] == pytest.approx(expected, rel=0.02)
    assert stats["min"] > 0


def test_basket_and_value_stats(eda_data):
    stats = eda.get_basket_and_value_stats(eda_data["orders"])
    assert stats["total_orders"] == len(eda_data["orders"])
    # Single-item orders dominate, as in the real Swiggy report (65%)
    dist = stats["basket_distribution_pct"]
    assert max(dist, key=dist.get) == 1
    assert stats["mean_order_value"] > 0


def test_weekday_stats(eda_data):
    stats = eda.get_weekday_stats(eda_data["daily"])
    assert len(stats) == 7
    # Weekend days (Fri, Sat, Sun) should have significantly higher volume than Mon-Thu
    mon_thu_mean = stats.loc[["Mon", "Tue", "Wed", "Thu"], "mean"].mean()
    fri_sun_mean = stats.loc[["Fri", "Sat", "Sun"], "mean"].mean()
    assert fri_sun_mean > mon_thu_mean * 1.25


def test_rain_vs_dry_stats(eda_data):
    stats = eda.get_rain_vs_dry_stats(eda_data["daily"])
    assert {"rainy_mean", "dry_mean", "lift_pct", "t_pvalue", "mwu_pvalue"} <= set(stats)
    # Rain x weekday (Swiggy evidence): rain lifts weekdays and lowers weekends
    d = eda_data["daily"]
    means = d.groupby(["is_weekend", "is_rainy"])["order_count"].mean()
    assert means[(False, True)] > means[(False, False)]
    assert means[(True, True)] < means[(True, False)]


def test_item_and_category_shifts(eda_data):
    item_shift, cat_regime = eda.get_item_and_category_shifts(
        eda_data["orders"], eda_data["daily"], eda_data["menu"]
    )
    lift = item_shift.set_index("item_name")["relative_lift_pct"]
    # Survey Q9: chai, pakora and samosa are rainy-day items
    assert lift["Chai"] > 40.0
    assert lift["Pakora"] > 40.0
    assert lift["Samosa"] > 40.0
    # Survey Q10: cold drinks and ice cream are hot-day items
    cold = cat_regime.loc["Cold Drinks & Desserts"]
    assert cold["hot"] > cold["rainy"] * 2.0


def test_correlation_matrix(eda_data):
    pearson, spearman = eda.get_correlation_matrix(eda_data["daily"])
    assert pearson.loc["total_revenue", "order_count"] > 0.9
    # On weekdays, rain raises orders (Swiggy evidence)
    d = eda_data["daily"]
    weekdays = d[~d["is_weekend"]]
    assert weekdays["precip_mm"].corr(weekdays["order_count"], method="spearman") > 0.2
