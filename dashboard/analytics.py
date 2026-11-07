"""Data loading and association-rule calculations for the dashboard."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONSOON_PHASES = {
    "SW monsoon (Jun-Sep)": {6, 7, 8, 9},
    "NE monsoon (Oct-Dec)": {10, 11, 12},
    "Dry season (Jan-May)": {1, 2, 3, 4, 5},
}


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


def _baskets(items: pd.Series) -> list[set[str]]:
    return [
        {item.strip() for item in str(value).split(";") if item.strip()}
        for value in items.fillna("")
    ]


def mine_pair_rules(
    orders: pd.DataFrame,
    daily_orders: pd.DataFrame,
    min_support: float = 0.005,
) -> pd.DataFrame:
    """Mine directed two-item rules separately for rainy and dry order baskets."""
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be between 0 and 1.")
    required_orders = {"date", "items"}
    required_daily = {"date", "is_rainy"}
    if not required_orders.issubset(orders.columns):
        raise ValueError(f"orders must contain {sorted(required_orders)}.")
    if not required_daily.issubset(daily_orders.columns):
        raise ValueError(f"daily_orders must contain {sorted(required_daily)}.")

    rain_by_date = daily_orders.set_index("date")["is_rainy"]
    segments = orders.assign(is_rainy=orders["date"].map(rain_by_date))
    if segments["is_rainy"].isna().any():
        raise ValueError("Every order date must have a matching rainy/dry weather record.")

    rows: list[dict[str, object]] = []
    for segment_name, rainy in (("Rainy", True), ("Dry", False)):
        baskets = _baskets(segments.loc[segments["is_rainy"].astype(bool) == rainy, "items"])
        transaction_count = len(baskets)
        if not transaction_count:
            continue

        item_counts: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()
        for basket in baskets:
            item_counts.update(basket)
            pair_counts.update(combinations(sorted(basket), 2))

        for (item_a, item_b), count in pair_counts.items():
            support = count / transaction_count
            if support < min_support:
                continue
            for antecedent, consequent in ((item_a, item_b), (item_b, item_a)):
                confidence = count / item_counts[antecedent]
                lift = confidence / (item_counts[consequent] / transaction_count)
                rows.append({
                    "segment": segment_name,
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": support,
                    "confidence": confidence,
                    "lift": lift,
                    "transactions": transaction_count,
                    "pair_count": count,
                })

    columns = [
        "segment", "antecedent", "consequent", "support", "confidence",
        "lift", "transactions", "pair_count",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["segment", "lift", "support", "confidence"],
        ascending=[True, False, False, False],
        ignore_index=True,
    )


def rolling_pair_lift(
    orders: pd.DataFrame,
    daily_orders: pd.DataFrame,
    antecedent: str,
    consequent: str,
    segment: str,
    window_days: int = 14,
    min_transactions: int = 20,
) -> pd.DataFrame:
    """Calculate a selected rule's lift in rolling calendar windows of one weather segment."""
    if segment not in {"Rainy", "Dry"}:
        raise ValueError("segment must be either 'Rainy' or 'Dry'.")
    if window_days < 1 or min_transactions < 1:
        raise ValueError("window_days and min_transactions must be positive.")
    required_orders = {"date", "items"}
    required_daily = {"date", "is_rainy"}
    if not required_orders.issubset(orders.columns):
        raise ValueError(f"orders must contain {sorted(required_orders)}.")
    if not required_daily.issubset(daily_orders.columns):
        raise ValueError(f"daily_orders must contain {sorted(required_daily)}.")

    dates = pd.DatetimeIndex(pd.to_datetime(daily_orders["date"]).sort_values().unique(), name="date")
    rain_by_date = daily_orders.set_index("date")["is_rainy"]
    frame = orders[["date", "items"]].copy()
    frame["is_rainy"] = frame["date"].map(rain_by_date)
    if frame["is_rainy"].isna().any():
        raise ValueError("Every order date must have a matching rainy/dry weather record.")
    is_rainy_segment = segment == "Rainy"
    frame = frame.loc[frame["is_rainy"].astype(bool) == is_rainy_segment].copy()
    tokens = _baskets(frame["items"])
    frame["has_a"] = [antecedent in basket for basket in tokens]
    frame["has_b"] = [consequent in basket for basket in tokens]
    frame["has_both"] = [antecedent in basket and consequent in basket for basket in tokens]
    counts = frame.groupby("date").agg(
        n=("items", "size"),
        a=("has_a", "sum"),
        b=("has_b", "sum"),
        ab=("has_both", "sum"),
    ).reindex(dates, fill_value=0)
    rolling = counts.rolling(window_days, min_periods=window_days).sum()
    n, a, b, ab = (rolling[column] for column in ("n", "a", "b", "ab"))
    valid = (n >= min_transactions) & (a > 0) & (b > 0)
    lift = pd.Series(np.nan, index=dates, dtype=float)
    lift.loc[valid] = ab.loc[valid] * n.loc[valid] / (a.loc[valid] * b.loc[valid])

    result = pd.DataFrame({
        "date": dates,
        "lift": lift.to_numpy(),
        "transactions": n.to_numpy(),
    })
    result["monsoon_phase"] = result["date"].dt.month.map(
        lambda month: next(
            phase for phase, months in MONSOON_PHASES.items() if month in months
        )
    )
    return result
