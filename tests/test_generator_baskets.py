"""Unit tests for basket generator, meal slots, and planted association rules (T09).

Run: pytest -q tests/test_generator_baskets.py
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generator as G  # noqa: E402

ORDERS_CSV = ROOT / "data" / "raw" / "orders.csv"
MENU_CSV = ROOT / "data" / "raw" / "menu.csv"
DAILY_CSV = ROOT / "data" / "raw" / "daily_orders.csv"

REQUIRED_COLUMNS = [
    "order_id",
    "date",
    "timestamp",
    "meal_slot",
    "items",
    "item_count",
    "order_value",
]


@pytest.fixture(scope="module")
def orders_df():
    assert ORDERS_CSV.exists(), f"Orders CSV missing at {ORDERS_CSV}"
    return pd.read_csv(ORDERS_CSV)


@pytest.fixture(scope="module")
def menu_df():
    assert MENU_CSV.exists(), f"Menu CSV missing at {MENU_CSV}"
    return pd.read_csv(MENU_CSV)


@pytest.fixture(scope="module")
def daily_df():
    assert DAILY_CSV.exists(), f"Daily CSV missing at {DAILY_CSV}"
    return pd.read_csv(DAILY_CSV)


def test_orders_schema_and_types(orders_df):
    """Verify orders.csv schema, non-null constraints, and ID monotonicity."""
    assert list(orders_df.columns) == REQUIRED_COLUMNS
    assert orders_df.isnull().sum().sum() == 0
    assert len(orders_df) >= 50_000
    assert orders_df["order_id"].is_monotonic_increasing
    assert orders_df["order_id"].iloc[0] == 1


def test_basket_sizes_and_item_counts(orders_df):
    """Basket item count must strictly match semicolon-delimited item list length."""
    assert orders_df["item_count"].between(1, 5).all()
    # Check on a subset of 1000 orders
    sample = orders_df.sample(1000, random_state=42)
    split_counts = sample["items"].apply(lambda s: len(s.split("; ")))
    pd.testing.assert_series_equal(
        split_counts, sample["item_count"], check_names=False
    )


def test_meal_slot_timestamp_bounds(orders_df):
    """Timestamps must strictly adhere to their respective meal slot windows."""
    sample = orders_df.sample(2000, random_state=42).copy()
    ts = pd.to_datetime(sample["timestamp"])
    mins_from_midnight = ts.dt.hour * 60 + ts.dt.minute

    # Breakfast: 07:00 to 11:00 (420 to 660 mins)
    b_mask = sample["meal_slot"] == "Breakfast"
    assert mins_from_midnight[b_mask].between(420, 660).all()

    # Lunch: 11:00 to 15:00 (660 to 900 mins)
    l_mask = sample["meal_slot"] == "Lunch"
    assert mins_from_midnight[l_mask].between(660, 900).all()

    # Snacks: 15:00 to 18:00 (900 to 1080 mins)
    s_mask = sample["meal_slot"] == "Snacks"
    assert mins_from_midnight[s_mask].between(900, 1080).all()

    # Dinner: 18:00 to 23:00 (1080 to 1380 mins)
    d_mask = sample["meal_slot"] == "Dinner"
    assert mins_from_midnight[d_mask].between(1080, 1380).all()


def test_order_value_accuracy(orders_df, menu_df):
    """Order total amount must exactly equal the sum of constituent item prices."""
    price_map = dict(zip(menu_df["item_name"], menu_df["price"]))
    sample = orders_df.sample(1000, random_state=42)

    calculated_vals = sample["items"].apply(
        lambda items_str: sum(price_map[i] for i in items_str.split("; "))
    )
    pd.testing.assert_series_equal(
        calculated_vals.astype(int), sample["order_value"].astype(int), check_names=False
    )


def test_association_rule_evaluation_structure(orders_df, daily_df):
    """Rule evaluation must return valid support, confidence, and lift for item combinations."""
    res = G.evaluate_planted_rules(
        orders_df, daily_df, trigger="Friendly Coffee (Hot)", addon="Shakencano"
    )
    assert set(res.columns) == {"regime", "orders", "supp_trigger", "supp_addon", "support", "confidence", "lift"}
    assert len(res) > 0
    assert (res["confidence"] >= 0.0).all()
    assert (res["confidence"] <= 1.0).all()
    assert (res["lift"] >= 0.0).all()

