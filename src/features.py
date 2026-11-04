"""Feature engineering for daily order prediction (P3 / T14).

Builds lag features, rolling statistics, weather bins, calendar and festival flags,
and monsoon phase indicators from the merged daily dataset.

Note: the merged file also holds same-day outcomes (total_revenue, avg_order_value,
avg_basket_size). They are results of the day's orders, not inputs, so never use them
as predictors of order_count.

Thresholds align with the Surabaya MSME POS dataset:
    Rain:     rain_mm >= 2.5 -> rainy, >= 15 -> heavy
    Hot:      temp_max_c >= 33 °C
    Weekend:  Fri / Sat / Sun  (dow 4, 5, 6)
    Monsoon:  NW Nov-Apr (wet season), SE / dry season May-Oct

Usage
-----
    python src/features.py                  # default paths
    from features import build_features     # import from notebook

Author: P3 - SaiSriSatvic
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Weather and calendar thresholds
# ---------------------------------------------------------------------------
RAIN_THRESHOLD = 2.5        # mm  ->  is_rainy
HEAVY_RAIN_THRESHOLD = 15.0 # mm  ->  heavy_rain
HOT_THRESHOLD = 33.0        # °C  ->  is_hot
WEEKEND_DAYS = {4, 5, 6}    # Fri, Sat, Sun


# ── lag & rolling ──────────────────────────────────────────────────────────
def add_lag_features(
    df: pd.DataFrame, target: str = "order_count"
) -> pd.DataFrame:
    """Lag-1, lag-7, and 7-day rolling mean of *target*.

    Rolling window is applied **after** shift(1) so that only information
    available by the evening before the predicted day is used.
    """
    df = df.sort_values("date").copy()
    df[f"{target}_lag1"] = df[target].shift(1)
    df[f"{target}_lag7"] = df[target].shift(7)
    df[f"{target}_roll7"] = df[target].shift(1).rolling(7).mean()
    return df


# ── weather bins ───────────────────────────────────────────────────────────
def add_rain_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Binary rain flag + four-level intensity bins."""
    df = df.copy()
    df["is_rainy"] = (df["rain_mm"] >= RAIN_THRESHOLD).astype(int)
    df["heavy_rain"] = (df["rain_mm"] >= HEAVY_RAIN_THRESHOLD).astype(int)
    # Left-closed bins [a, b) so a day at exactly 2.5 mm is "moderate" and at exactly 15 mm is "heavy",
    # matching is_rainy (>= 2.5) and heavy_rain (>= 15).
    df["rain_intensity"] = pd.cut(
        df["rain_mm"],
        bins=[-float("inf"), 1e-9, RAIN_THRESHOLD, HEAVY_RAIN_THRESHOLD, float("inf")],
        labels=["none", "trace", "moderate", "heavy"],
        right=False,
    )
    return df


def add_temp_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Binary hot flag + four-level temperature bands."""
    df = df.copy()
    df["is_hot"] = (df["temp_max_c"] >= HOT_THRESHOLD).astype(int)
    # Left-closed bins [a, b) so a day at exactly 33.0 C is "hot", matching is_hot (>= 33).
    df["temp_band"] = pd.cut(
        df["temp_max_c"],
        bins=[-float("inf"), 28, HOT_THRESHOLD, 38, float("inf")],
        labels=["cool", "warm", "hot", "extreme"],
        right=False,
    )
    return df


# ── calendar / monsoon ────────────────────────────────────────────────────
def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week and weekend flag (Fri-Sun, matching Indonesian retail patterns)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin(WEEKEND_DAYS).astype(int)
    return df


def add_festival_flag(df: pd.DataFrame, daily_orders_csv: str | Path | None = None) -> pd.DataFrame:
    """Add `is_festival` (0/1) from the Indonesian national holiday calendar in data/raw/daily_orders.csv.

    Indonesian public holidays capture demand anomalies and shifts. The date is
    known in advance, so it is safe to use as a predictor. Fails loudly if any date has no flag.
    """
    path = Path(daily_orders_csv) if daily_orders_csv else ROOT / "data" / "raw" / "daily_orders.csv"
    flags = pd.read_csv(path, parse_dates=["date"])[["date", "is_festival"]]
    flags["is_festival"] = flags["is_festival"].astype(int)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop(columns=["is_festival"], errors="ignore").merge(flags, on="date", how="left")
    if out["is_festival"].isna().any():
        missing = int(out["is_festival"].isna().sum())
        raise ValueError(f"festival flag missing for {missing} dates; check {path}")
    out["is_festival"] = out["is_festival"].astype(int)
    return out


def add_monsoon_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Tag Surabaya monsoon season: NW monsoon (wet season, Nov-Apr), dry season (May-Oct)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    month = df["date"].dt.month
    df["monsoon_phase"] = np.where(
        month.isin([11, 12, 1, 2, 3, 4]),
        "nw_monsoon",
        "dry_season",
    )
    return df


# ── full pipeline ─────────────────────────────────────────────────────────
def build_features(
    input_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end: load merged data -> add all features -> save.

    Parameters
    ----------
    input_csv  : path to the cleaned merged CSV (default: data/processed/daily_merged.csv)
    output_csv : where to write the feature table (default: data/processed/daily_features.csv)

    Returns
    -------
    pd.DataFrame with all engineered features appended.
    """
    if input_csv is None:
        input_csv = ROOT / "data" / "processed" / "daily_merged.csv"
    if output_csv is None:
        output_csv = ROOT / "data" / "processed" / "daily_features.csv"

    df = pd.read_csv(input_csv, parse_dates=["date"])
    print(f"[features] loaded {input_csv}  ->  {df.shape[0]} rows, {df.shape[1]} cols")

    if "is_festival" not in df.columns:
        df = add_festival_flag(df)
    df = add_lag_features(df)
    df = add_rain_bins(df)
    df = add_temp_bands(df)
    df = add_calendar_flags(df)
    df = add_monsoon_phase(df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[features] saved  {output_csv}  ->  {df.shape[0]} rows, {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    build_features()
