"""Unit tests for the evidence-based order and basket generator (T09).

Run: pytest -q tests/test_generator_baskets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import calibration as C  # noqa: E402
import generator as G  # noqa: E402
import real_orders as R  # noqa: E402

ORDERS_CSV = ROOT / "data" / "raw" / "orders.csv"
REQUIRED_COLUMNS = ["order_id", "date", "timestamp", "meal_slot", "items", "item_count", "order_value"]


@pytest.fixture(scope="module")
def cal():
    return C.calibrate()


@pytest.fixture(scope="module")
def menu(cal):
    _, items = R.load_clean()
    return G.build_menu(items, cal.snack_prices)


@pytest.fixture(scope="module")
def sample(cal, menu):
    """Generate 60 days of orders (20 per regime) for fast checks."""
    daily = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=60).strftime("%Y-%m-%d"),
        "regime": ["rainy", "hot", "normal"] * 20,
        "order_count": 150,
    })
    return G.generate_order_baskets(daily, cal, menu, seed=1)


@pytest.fixture(scope="module")
def orders_df():
    assert ORDERS_CSV.exists(), f"Orders CSV missing at {ORDERS_CSV}"
    return pd.read_csv(ORDERS_CSV)


def test_orders_schema(orders_df):
    assert list(orders_df.columns) == REQUIRED_COLUMNS
    assert orders_df.isnull().sum().sum() == 0
    assert len(orders_df) > 100_000
    assert orders_df["order_id"].is_monotonic_increasing


def test_item_count_matches_item_list(sample):
    orders, _ = sample
    assert (orders["items"].str.split("; ").str.len() == orders["item_count"]).all()


def test_order_value_equals_sum_of_lines(sample):
    orders, items = sample
    line_sum = items.groupby("order_id")["price"].sum()
    np.testing.assert_allclose(orders.set_index("order_id")["order_value"], line_sum)


def test_hours_stay_inside_real_trading_hours(sample, cal):
    orders, _ = sample
    hours = pd.to_datetime(orders["timestamp"]).dt.hour
    assert set(hours) <= set(cal.hour_profile)
    assert (orders["meal_slot"] == hours.map(G.meal_slot)).all()


def test_meal_slot_rejects_closed_hours():
    with pytest.raises(ValueError):
        G.meal_slot(8)


def test_food_orders_are_real_swiggy_baskets(sample, cal):
    orders, _ = sample
    real = {tuple(n for n, _ in b) for b in cal.food_baskets}
    food = orders[~orders["items"].map(G.is_snack_order)]
    stripped = food["items"].str.replace("; Buttermilk", "", regex=False).str.split("; ").map(tuple)
    assert stripped.isin(real).all()


def test_snack_share_follows_weather(sample):
    orders, _ = sample
    regime = pd.Series(["rainy", "hot", "normal"] * 20,
                       index=pd.date_range("2025-01-01", periods=60).strftime("%Y-%m-%d"))
    share = orders.assign(regime=orders["date"].map(regime),
                          snack=orders["items"].map(G.is_snack_order)).groupby("regime")["snack"].mean()
    assert share["hot"] > share["rainy"] > share["normal"]


def test_set_pairs_adds_or_removes_addon_by_probability(cal):
    from dataclasses import replace
    rng = np.random.default_rng(0)
    always = replace(cal, pair_probs={("Coffee", "Vada"): {"rainy": 1.0, "hot": 1.0, "normal": 1.0}})
    never = replace(cal, pair_probs={("Coffee", "Vada"): {"rainy": 0.0, "hot": 0.0, "normal": 0.0}})
    assert G.set_pairs(["Coffee"], "rainy", always, rng) == ["Coffee", "Vada"]
    assert G.set_pairs(["Vada", "Coffee"], "rainy", never, rng) == ["Coffee"]
    assert G.set_pairs(["Vada"], "rainy", never, rng) == ["Vada"]  # no trigger, untouched


def test_triggers_match_exact_snacks_and_any_biryani():
    assert G.triggers("Chicken Dum Biryani", "Biryani")
    assert G.triggers("Coffee", "Coffee")
    assert not G.triggers("Cold Coffee", "Coffee")


def test_menu_has_real_food_and_sourced_snacks(menu, cal):
    assert menu["item_name"].is_unique and menu["item_id"].is_unique
    assert set(G.SNACK_CATEGORIES) <= set(menu["item_name"])
    assert (menu["price"] > 0).all()
    snack = menu[menu["item_name"].isin(G.SNACK_CATEGORIES)].set_index("item_name")["price"]
    assert snack["Cold Coffee"] == round(cal.snack_prices["Cold Coffee"])


def test_food_category_keywords():
    assert G.food_category("Chicken Kozhi Rasam") == "Soups"
    assert G.food_category("Egg Fried Rice") == "Biryani & Rice"
    assert G.food_category("Egg Kothu Parotta") == "Breads & Tiffin"
    assert G.food_category("Omelette") == "Egg"
    assert G.food_category("Mutton Chukka") == "Starters & Gravies"
