"""Unit tests for Coimbatore menu catalog (T07).

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
    "Beverages (Hot)",
    "Snacks & Quick Bites",
    "South Indian Tiffin",
    "Biryani & Mains",
    "Cold Refreshments & Juices",
    "Desserts & Ice Cream",
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
    """Must contain around 40 items (exactly 40 items)."""
    assert len(menu_df) == 40
    assert menu_df["item_id"].is_unique
    assert menu_df["item_name"].is_unique


def test_menu_categories_and_affinities(menu_df):
    """Categories and affinities must conform to project definitions."""
    assert set(menu_df["category"].unique()) == EXPECTED_CATEGORIES
    assert set(menu_df["weather_affinity"].unique()) == EXPECTED_AFFINITIES


def test_menu_prices(menu_df):
    """Prices must be positive realistic INR values."""
    assert (menu_df["price"] > 0).all()
    assert (menu_df["price"] <= 500).all()


def test_required_coimbatore_items_present(menu_df):
    """Verify headline items local to Coimbatore specified in rubric."""
    items = " ".join(menu_df["item_name"].str.lower())
    assert "filter coffee" in items
    assert "chai" in items
    assert "pakora" in items
    assert "bajji" in items
    assert "parotta" in items
    assert "biryani" in items
    assert "juice" in items
    assert "ice cream" in items
