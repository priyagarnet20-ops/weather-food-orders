"""Data loading and association-rule calculations for the dashboard."""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

MONSOON_PHASES = {
    "SW monsoon (Jun-Sep)": {6, 7, 8, 9},
    "NE monsoon (Oct-Dec)": {10, 11, 12},
    "Dry season (Jan-May)": {1, 2, 3, 4, 5},
}
PLANTED_RAIN_PAIRS = (
    ("Chai", "Pakora"),
    ("Chai", "Samosa"),
    ("Coffee", "Vada"),
)


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


def compare_weather_pair_rules(
    rules: pd.DataFrame,
    min_confidence: float = 0.05,
    max_lift: float = 20.0,
) -> pd.DataFrame:
    """Compare directed pair rules across weather regimes and remove basket shortcuts."""
    if not 0 <= min_confidence < 1:
        raise ValueError("min_confidence must be between 0 and 1.")
    if max_lift <= 0:
        raise ValueError("max_lift must be positive.")
    required = {
        "segment", "antecedent", "consequent", "support", "confidence",
        "lift", "pair_count", "transactions",
    }
    if not required.issubset(rules.columns):
        raise ValueError(f"rules must contain {sorted(required)}.")

    eligible = rules.loc[
        (rules["confidence"] >= min_confidence)
        & (rules["confidence"] < 1.0 - 1e-12)
        & (rules["lift"] <= max_lift)
        & rules["segment"].isin(["Rainy", "Dry"])
    ].copy()
    pair_columns = ["antecedent", "consequent"]
    rainy = eligible.loc[eligible["segment"] == "Rainy"].drop(columns="segment")
    dry = eligible.loc[eligible["segment"] == "Dry"].drop(columns="segment")
    rainy = rainy.rename(columns={
        "support": "rainy_support",
        "confidence": "rainy_confidence",
        "lift": "rainy_lift",
        "pair_count": "rainy_pair_count",
        "transactions": "rainy_orders",
    })
    dry = dry.rename(columns={
        "support": "dry_support",
        "confidence": "dry_confidence",
        "lift": "dry_lift",
        "pair_count": "dry_pair_count",
        "transactions": "dry_orders",
    })
    comparison = rainy.merge(dry, on=pair_columns, how="inner", validate="one_to_one")
    comparison["confidence_change_pp"] = (
        comparison["rainy_confidence"] - comparison["dry_confidence"]
    ) * 100
    comparison["lift_change"] = comparison["rainy_lift"] - comparison["dry_lift"]
    comparison["lift_ratio"] = comparison["rainy_lift"] / comparison["dry_lift"]
    comparison["planted_rain_pair"] = comparison.apply(
        lambda row: (row["antecedent"], row["consequent"]) in PLANTED_RAIN_PAIRS,
        axis=1,
    )
    return comparison.sort_values(
        ["confidence_change_pp", "rainy_support"],
        ascending=[False, False],
        ignore_index=True,
    )


def filter_weather_effect(comparison: pd.DataFrame, effect: str) -> pd.DataFrame:
    """Filter a paired-rule comparison by which weather regime has higher confidence."""
    if effect == "All pairs":
        return comparison.copy()
    if effect == "Stronger on rainy days":
        return comparison.loc[comparison["confidence_change_pp"] > 0].copy()
    if effect == "Stronger on dry days":
        return comparison.loc[comparison["confidence_change_pp"] < 0].copy()
    raise ValueError(f"Unsupported weather effect filter: {effect}")


def rolling_pair_lift(
    orders: pd.DataFrame,
    daily_orders: pd.DataFrame,
    antecedent: str,
    consequent: str,
    segment: str,
    window_days: int = 14,
    min_transactions: int = 20,
    min_antecedent_transactions: int = 20,
) -> pd.DataFrame:
    """Calculate a selected rule's lift in rolling calendar windows of one weather segment."""
    if segment not in {"Rainy", "Dry"}:
        raise ValueError("segment must be either 'Rainy' or 'Dry'.")
    if window_days < 1 or min_transactions < 1 or min_antecedent_transactions < 1:
        raise ValueError("window and transaction thresholds must be positive.")
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
    valid = (n >= min_transactions) & (a >= min_antecedent_transactions) & (b > 0)
    lift = pd.Series(np.nan, index=dates, dtype=float)
    confidence = pd.Series(np.nan, index=dates, dtype=float)
    lift.loc[valid] = ab.loc[valid] * n.loc[valid] / (a.loc[valid] * b.loc[valid])
    confidence.loc[valid] = ab.loc[valid] / a.loc[valid]

    result = pd.DataFrame({
        "date": dates,
        "lift": lift.to_numpy(),
        "confidence": confidence.to_numpy(),
        "transactions": n.to_numpy(),
        "antecedent_orders": a.to_numpy(),
        "pair_orders": ab.to_numpy(),
    })
    result["monsoon_phase"] = result["date"].dt.month.map(
        lambda month: next(
            phase for phase, months in MONSOON_PHASES.items() if month in months
        )
    )
    return result
