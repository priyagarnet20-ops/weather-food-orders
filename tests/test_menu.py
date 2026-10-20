"""Unit tests for Surabaya menu catalog (T07).

Run: pytest -q tests/test_menu.py
"""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MENU_CSV = ROOT / "data" / "raw" / "menu.csv"

REQUIRED_COLUMNS = [
    "item_id",
    "item_name",
    "category",
    "price",
    "weather_affinity",
    "description",
]

EXPECTED_CATEGORIES = {
    "Coffee",
    "Tea",
    "Menu Gedhe",
    "Snacks",
    "Other",
    "Add on",
}

EXPECTED_AFFINITIES = {"hot_comfort", "cold_refresh", "neutral"}


@pytest.fixture
def menu_df():
    assert MENU_CSV.exists(), f"Menu file not found at {MENU_CSV}"
    df = pd.read_csv(MENU_CSV)
    return df


def test_menu_schema(menu_df):
    """Catalog must contain the exact required columns with no nulls."""
    assert list(menu_df.columns) == REQUIRED_COLUMNS
    assert menu_df.isnull().sum().sum() == 0


def test_menu_item_count(menu_df):
    """Must contain the 96 catalog products."""
    assert len(menu_df) == 96
    assert menu_df["item_id"].is_unique
    assert menu_df["item_name"].is_unique


def test_menu_categories_and_affinities(menu_df):
    """Categories and affinities must conform to project definitions."""
    assert set(menu_df["category"].unique()) == EXPECTED_CATEGORIES
    assert set(menu_df["weather_affinity"].unique()) == EXPECTED_AFFINITIES


def test_menu_prices(menu_df):
    """Prices must be realistic IDR values."""
    assert (menu_df["price"] >= 0).all()
    assert (menu_df["price"] <= 50000).all()


def test_required_surabaya_items_present(menu_df):
    """Verify headline items local to Surabaya F&B store."""
    items = " ".join(menu_df["item_name"].str.lower())
    assert "americano" in items
    assert "latte" in items
    assert "tea" in items
    assert "banana bread" in items
    assert "waffle" in items
    assert "jus" in items

