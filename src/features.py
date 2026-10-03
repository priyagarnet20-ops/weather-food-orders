"""Feature engineering for daily order prediction (P3 / T14).

Builds lag features, rolling statistics, weather bins, calendar flags,
and monsoon phase indicators from the merged daily dataset.

Thresholds align with the order generator (docs/generator.md):
    Rain:     precip_mm >= 2.5 → rainy, >= 15 → heavy
    Hot:      temp_max_c >= 33 °C
    Weekend:  Fri / Sat / Sun  (dow 4, 5, 6)
    Monsoon:  SW Jun-Sep, NE Oct-Dec

Usage
-----
    python src/features.py                  # default paths
    from features import build_features     # import from notebook

Author: P3 – SaiSriSatvic
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Generator-consistent thresholds
# ---------------------------------------------------------------------------
RAIN_THRESHOLD = 2.5        # mm  →  is_rainy
HEAVY_RAIN_THRESHOLD = 15.0 # mm  →  heavy_rain
HOT_THRESHOLD = 33.0        # °C  →  is_hot
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
    df["rain_intensity"] = pd.cut(
        df["rain_mm"],
        bins=[-0.1, 0.0, RAIN_THRESHOLD, HEAVY_RAIN_THRESHOLD, float("inf")],
        labels=["none", "trace", "moderate", "heavy"],
    )
    return df


def add_temp_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Binary hot flag + four-level temperature bands."""
    df = df.copy()
    df["is_hot"] = (df["temp_max_c"] >= HOT_THRESHOLD).astype(int)
    df["temp_band"] = pd.cut(
        df["temp_max_c"],
        bins=[0, 28, HOT_THRESHOLD, 38, 50],
        labels=["cool", "warm", "hot", "extreme"],
    )
    return df


# ── calendar / monsoon ────────────────────────────────────────────────────
def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week and weekend flag (Fri-Sun, matching the generator)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin(WEEKEND_DAYS).astype(int)
    return df


def add_monsoon_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Tag monsoon phase: SW (Jun-Sep), NE (Oct-Dec), dry (Jan-May)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    month = df["date"].dt.month
    df["monsoon_phase"] = np.select(
        [month.between(6, 9), month.between(10, 12)],
        ["sw_monsoon", "ne_monsoon"],
        default="dry_season",
    )
    return df


# ── full pipeline ─────────────────────────────────────────────────────────
def build_features(
    input_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end: load merged data → add all features → save.

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
    print(f"[features] loaded {input_csv}  →  {df.shape[0]} rows, {df.shape[1]} cols")

    df = add_lag_features(df)
    df = add_rain_bins(df)
    df = add_temp_bands(df)
    df = add_calendar_flags(df)
    df = add_monsoon_phase(df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[features] saved  {output_csv}  →  {df.shape[0]} rows, {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    build_features()
