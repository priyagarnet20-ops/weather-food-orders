"""Unit tests for the Swiggy order report loader (validation base data).

Run: pytest -q tests/test_real_orders.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import real_orders as R  # noqa: E402

ITEM_COL = "Item1-name_reward_type_quantity_price+Variants+Addons"


def _raw_report(rows: list[dict]) -> pd.DataFrame:
    """Build a frame shaped like the Swiggy sheet after reading with header=5."""
    base = {
        "Order ID": 1,
        "Order-status": "delivered",
        "Order-relay-time(ordered time)": "2026-08-05 12:30:00",
        "Total-bill-amount <bill>": 300,
        "Item-count": 1,
        ITEM_COL: "Non Veg Meals_NA_1_286",
        "Unnamed: 31": None,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_parse_item_cell_reads_name_quantity_and_line_price():
    assert R.parse_item_cell("Chapati 1 pce_NA_2_114") == ("Chapati 1 pce", 2, 114.0)


def test_parse_item_cell_drops_addons_after_plus():
    name, qty, price = R.parse_item_cell("Chicken Fried Rice_NA_1_483+Mutton Attukal Soup")
    assert (name, qty, price) == ("Chicken Fried Rice", 1, 483.0)


def test_parse_item_cell_rejects_malformed_text():
    with pytest.raises(ValueError):
        R.parse_item_cell("not an item")


def test_clean_report_keeps_only_delivered_orders():
    raw = _raw_report([{"Order ID": 1}, {"Order ID": 2, "Order-status": "cancelled"}])
    orders, _ = R.clean_report(raw)
    assert len(orders) == 1


def test_clean_report_replaces_swiggy_ids_with_sequence():
    raw = _raw_report([{"Order ID": 249747056127707}, {"Order ID": 249729292183063}])
    orders, items = R.clean_report(raw)
    assert orders["order_no"].tolist() == [1, 2]
    assert "Order ID" not in orders.columns
    assert set(items["order_no"]) == {1, 2}


def test_clean_report_orders_are_sorted_by_time():
    raw = _raw_report([
        {"Order ID": 1, "Order-relay-time(ordered time)": "2026-08-06 20:00:00"},
        {"Order ID": 2, "Order-relay-time(ordered time)": "2026-08-05 12:00:00"},
    ])
    orders, _ = R.clean_report(raw)
    assert orders["timestamp"].is_monotonic_increasing


def test_clean_report_explodes_item_lines():
    raw = _raw_report([{
        "Item-count": 2,
        ITEM_COL: "Non Veg Meals_NA_1_286",
        "Unnamed: 31": "Chicken Kozhi Milagu Soup_NA_1_148",
    }])
    orders, items = R.clean_report(raw)
    assert orders.loc[0, "item_count"] == 2
    assert items["item_name"].tolist() == ["Non Veg Meals", "Chicken Kozhi Milagu Soup"]
    assert items["unit_price"].tolist() == [286.0, 148.0]


def test_clean_report_adds_date_hour_and_dow():
    raw = _raw_report([{"Order-relay-time(ordered time)": "2026-08-09 19:41:15"}])
    orders, _ = R.clean_report(raw)
    row = orders.iloc[0]
    assert row["date"] == "2026-08-09"
    assert row["hour"] == 19
    assert row["dow"] == 6  # Sunday


def test_clean_report_fails_when_item_count_disagrees_with_lines():
    raw = _raw_report([{"Item-count": 3}])
    with pytest.raises(ValueError, match="item count"):
        R.clean_report(raw)


def test_clean_report_fails_on_missing_columns():
    with pytest.raises(ValueError, match="missing"):
        R.clean_report(pd.DataFrame({"Order ID": [1]}))
