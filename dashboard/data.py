"""Validated dashboard data loading."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_dashboard_data(root: str | Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load current project data and reject incomplete or misaligned inputs."""
    root = Path(root)
    daily = pd.read_csv(root / "data" / "processed" / "daily_features.csv", parse_dates=["date"])
    orders = pd.read_csv(root / "data" / "raw" / "orders.csv", parse_dates=["date"])
    daily_orders = pd.read_csv(root / "data" / "raw" / "daily_orders.csv", parse_dates=["date"])

    required_daily = {"date", "order_count", "precip_mm", "temp_max_c", "is_rainy", "is_hot", "is_festival"}
    required_orders = {"order_id", "date", "items"}
    required_daily_orders = {"date", "is_rainy"}
    if not required_daily.issubset(daily.columns):
        raise ValueError(f"Daily features are missing required columns: {sorted(required_daily - set(daily.columns))}")
    if not required_orders.issubset(orders.columns):
        raise ValueError(f"Orders are missing required columns: {sorted(required_orders - set(orders.columns))}")
    if not required_daily_orders.issubset(daily_orders.columns):
        raise ValueError(f"Daily orders are missing required columns: {sorted(required_daily_orders - set(daily_orders.columns))}")
    if daily["date"].duplicated().any() or daily_orders["date"].duplicated().any():
        raise ValueError("Daily dashboard data must contain exactly one row per date.")
    if orders["order_id"].duplicated().any():
        raise ValueError("Order IDs must be unique before mining association rules.")
    if orders["date"].isna().any() or daily["date"].isna().any():
        raise ValueError("Dashboard dates must not be missing.")

    daily = daily.sort_values("date").reset_index(drop=True)
    orders = orders.sort_values(["date", "order_id"]).reset_index(drop=True)
    daily_orders = daily_orders.sort_values("date").reset_index(drop=True)
    return daily, orders, daily_orders
