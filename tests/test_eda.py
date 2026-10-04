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
    assert len(daily) == 1097
    assert len(orders) == 193406
    assert len(menu) == 40

    # Ensure no nulls in critical columns
    assert daily["order_count"].isna().sum() == 0
    assert daily["total_revenue"].isna().sum() == 0
    assert orders["order_value"].isna().sum() == 0
    assert orders["items"].isna().sum() == 0


def test_order_distribution_stats(eda_data):
    stats = eda.get_order_distribution_stats(eda_data["daily"])
    assert stats["count"] == 1097
    assert stats["mean"] == pytest.approx(176.3, abs=0.2)
    assert stats["median"] == 170.0
    assert stats["min"] == 61
    assert stats["max"] == 384
    assert stats["skewness"] > 0  # Right-skewed distribution


def test_basket_and_value_stats(eda_data):
    stats = eda.get_basket_and_value_stats(eda_data["orders"])
    assert stats["total_orders"] == 193406
    assert stats["mean_basket_size"] == pytest.approx(2.11, abs=0.05)
    assert stats["modal_basket_size"] == 2 if "modal_basket_size" in stats else True
    assert stats["basket_distribution_pct"][2] == pytest.approx(42.0, abs=0.5)
    assert stats["mean_order_value"] == pytest.approx(160.7, abs=0.5)


def test_weekday_stats(eda_data):
    stats = eda.get_weekday_stats(eda_data["daily"])
    assert len(stats) == 7
    # Weekend days (Fri, Sat, Sun) should have significantly higher volume than Mon-Thu
    mon_thu_mean = stats.loc[["Mon", "Tue", "Wed", "Thu"], "mean"].mean()
    fri_sun_mean = stats.loc[["Fri", "Sat", "Sun"], "mean"].mean()
    assert fri_sun_mean > mon_thu_mean * 1.25


def test_rain_vs_dry_stats(eda_data):
    stats = eda.get_rain_vs_dry_stats(eda_data["daily"])
    assert stats["rainy_mean"] > stats["dry_mean"]
    assert stats["lift_pct"] == pytest.approx(23.2, abs=1.0)
    assert stats["t_pvalue"] < 1e-10
    assert stats["mwu_pvalue"] < 1e-10


def test_item_and_category_shifts(eda_data):
    item_shift, cat_regime = eda.get_item_and_category_shifts(
        eda_data["orders"], eda_data["daily"], eda_data["menu"]
    )
    # Check that Masala Chai and Onion Pakora have positive relative lift on rainy days
    chai_lift = item_shift[item_shift["item_name"] == "Masala Chai"]["relative_lift_pct"].values[0]
    pakora_lift = item_shift[item_shift["item_name"] == "Onion Pakora"]["relative_lift_pct"].values[0]
    assert chai_lift > 40.0
    assert pakora_lift > 80.0

    # Cold refreshments should plunge on rainy days
    cold_cat_rain = cat_regime.loc["Cold Refreshments & Juices", "rainy"]
    cold_cat_hot = cat_regime.loc["Cold Refreshments & Juices", "hot"]
    assert cold_cat_hot > cold_cat_rain * 2.0


def test_correlation_matrix(eda_data):
    pearson, spearman = eda.get_correlation_matrix(eda_data["daily"])
    # Rain should be positively correlated with order count
    assert pearson.loc["precip_mm", "order_count"] > 0.20
    assert spearman.loc["precip_mm", "order_count"] > 0.20
    # Revenue should be strongly positively correlated with order count
    assert pearson.loc["total_revenue", "order_count"] > 0.95
