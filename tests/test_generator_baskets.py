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
    assert len(orders_df) > 100_000
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

    # Breakfast: 07:00 to 10:30 (420 to 630 mins)
    b_mask = sample["meal_slot"] == "Breakfast"
    assert mins_from_midnight[b_mask].between(420, 630).all()

    # Lunch: 12:00 to 15:00 (720 to 900 mins)
    l_mask = sample["meal_slot"] == "Lunch"
    assert mins_from_midnight[l_mask].between(720, 900).all()

    # Evening Snacks: 16:00 to 18:30 (960 to 1110 mins)
    s_mask = sample["meal_slot"] == "Evening Snacks"
    assert mins_from_midnight[s_mask].between(960, 1110).all()

    # Dinner: 19:00 to 22:30 (1140 to 1350 mins)
    d_mask = sample["meal_slot"] == "Dinner"
    assert mins_from_midnight[d_mask].between(1140, 1350).all()


def test_order_value_accuracy(orders_df, menu_df):
    """Order total amount must exactly equal the sum of constituent item prices."""
    price_map = dict(zip(menu_df["item_name"], menu_df["price"]))
    sample = orders_df.sample(1000, random_state=42)

    calculated_vals = sample["items"].apply(
        lambda items_str: sum(price_map[i] for i in items_str.split("; "))
    )
    pd.testing.assert_series_equal(
        calculated_vals, sample["order_value"], check_names=False
    )


def test_planted_association_rain_chai_pakora(orders_df, daily_df):
    """Chai -> Pakora confidence and support must be significantly higher on rainy days."""
    res = G.evaluate_planted_rules(
        orders_df, daily_df, trigger="Masala Chai", addon="Onion Pakora"
    )
    rainy_row = res[res["regime"] == "rainy"].iloc[0]
    normal_row = res[res["regime"] == "normal"].iloc[0]
    hot_row = res[res["regime"] == "hot"].iloc[0]

    # Rainy day confidence should be more than double normal/hot days
    assert rainy_row["confidence"] > 2.0 * normal_row["confidence"]
    assert rainy_row["confidence"] > 2.0 * hot_row["confidence"]
    assert rainy_row["lift"] > 2.0


def test_planted_association_heat_juice_icecream(orders_df, daily_df):
    """Juice -> Ice Cream confidence and support must be significantly higher on hot days."""
    res = G.evaluate_planted_rules(
        orders_df,
        daily_df,
        trigger="Fresh Lime Juice",
        addon="Tender Coconut Ice Cream",
    )
    rainy_row = res[res["regime"] == "rainy"].iloc[0]
    hot_row = res[res["regime"] == "hot"].iloc[0]

    # Hot day confidence must be at least 3x rainy day confidence
    assert hot_row["confidence"] > 3.0 * rainy_row["confidence"]
    assert hot_row["lift"] > 2.5
