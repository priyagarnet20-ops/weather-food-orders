"""Unit tests for the menu catalog written by the generator (T07).

Run: pytest -q tests/test_menu.py
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MENU_CSV = ROOT / "data" / "raw" / "menu.csv"
REAL_ITEMS_CSV = ROOT / "data" / "real" / "chinthamani_items.csv"

REQUIRED_COLUMNS = ["item_id", "item_name", "category", "price", "weather_affinity", "description"]
EXPECTED_AFFINITIES = {"hot_comfort", "cold_refresh", "neutral"}
SNACKS_AND_DRINKS = {
    "Chai", "Coffee", "Samosa", "Pakora", "Bajji", "Vada",
    "Ice Cream", "Cold Coffee", "Fresh Lime Juice", "Buttermilk",
}


@pytest.fixture(scope="module")
def menu_df():
    assert MENU_CSV.exists(), f"Menu file not found at {MENU_CSV}"
    return pd.read_csv(MENU_CSV)


def test_menu_schema(menu_df):
    assert list(menu_df.columns) == REQUIRED_COLUMNS
    assert menu_df.isnull().sum().sum() == 0


def test_menu_holds_every_real_item_plus_snacks(menu_df):
    real = set(pd.read_csv(REAL_ITEMS_CSV)["item_name"])
    assert set(menu_df["item_name"]) == real | SNACKS_AND_DRINKS
    assert menu_df["item_id"].is_unique


def test_menu_affinities(menu_df):
    assert set(menu_df["weather_affinity"]) == EXPECTED_AFFINITIES


def test_menu_prices_positive(menu_df):
    assert (menu_df["price"] > 0).all()
