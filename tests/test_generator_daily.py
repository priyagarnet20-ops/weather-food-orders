"""Unit tests for weather-driven daily demand generator (T08).

Run: pytest -q tests/test_generator_daily.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generator as G  # noqa: E402
import forecast as F   # noqa: E402

WEATHER_CSV = ROOT / "data" / "raw" / "weather.csv"


@pytest.fixture
def weather_df():
    assert WEATHER_CSV.exists(), f"Missing weather CSV at {WEATHER_CSV}"
    return pd.read_csv(WEATHER_CSV)


def test_seed_reproducibility(weather_df):
    """Running generate_daily_demand with fixed seed must be bitwise deterministic."""
    run1 = G.generate_daily_demand(weather_df, seed=42)
    run2 = G.generate_daily_demand(weather_df, seed=42)
    pd.testing.assert_series_equal(run1["order_count"], run2["order_count"])

    # Different seed should yield different stochastic counts
    run3 = G.generate_daily_demand(weather_df, seed=99)
    assert not run1["order_count"].equals(run3["order_count"])


def test_multipliers_and_regimes(weather_df):
    """Verify multiplier mappings and weather regime categorization."""
    df = G.generate_daily_demand(weather_df, seed=42)

    # Weekday multiplier: Fri(4), Sat(5), Sun(6) are 1.30; Mon-Thu are 1.00
    assert (df.loc[df["is_weekend"], "mult_weekend"] == G.WEEKEND_MULT).all()
    assert (df.loc[~df["is_weekend"], "mult_weekend"] == 1.00).all()

    # Rain multiplier checks
    dry_mask = df["precip_mm"] < G.RAIN_TRACE_MM
    mid_rain_mask = (df["precip_mm"] >= G.RAIN_TRACE_MM) & (df["precip_mm"] < G.RAIN_HEAVY_MM)
    heavy_rain_mask = df["precip_mm"] >= G.RAIN_HEAVY_MM

    assert (df.loc[dry_mask, "mult_rain"] == 1.00).all()
    assert (df.loc[mid_rain_mask, "mult_rain"] == G.RAIN_MID_MULT).all()
    assert (df.loc[heavy_rain_mask, "mult_rain"] == G.RAIN_HEAVY_MULT).all()

    # Heat multiplier checks
    hot_mask = df["temp_max_c"] >= G.TEMP_HOT_C
    assert (df.loc[hot_mask, "mult_temp"] == G.HEAT_MULT).all()
    assert (df.loc[~hot_mask, "mult_temp"] == 1.00).all()

    # Regime categorization
    assert set(df["regime"].unique()).issubset({"rainy", "hot", "normal"})
    assert (df.loc[df["is_rainy"], "regime"] == "rainy").all()


def test_festival_multiplier(weather_df):
    """Verify Coimbatore and Tamil Nadu festival calendar multipliers."""
    df = G.generate_daily_demand(weather_df, seed=42)
    assert (df.loc[df["is_festival"], "mult_festival"] == G.FESTIVAL_MULT).all()
    assert (df.loc[~df["is_festival"], "mult_festival"] == 1.00).all()
    assert df["is_festival"].sum() > 0, "Expected festival days to be identified"


def test_negative_binomial_mean_convergence(weather_df):
    """Empirical mean across the 3-year period must align with expected orders."""
    df = G.generate_daily_demand(weather_df, seed=42)
    emp_mean = df["order_count"].mean()
    exp_mean = df["expected_orders"].mean()
    # Negative binomial stochastic mean should be within 2% of expected mean over 1097 days
    assert abs(emp_mean - exp_mean) / exp_mean < 0.02


def test_compatibility_with_forecast_load_daily(weather_df, tmp_path):
    """Daily output must load seamlessly in P1's time-series load_daily function."""
    df = G.generate_daily_demand(weather_df, seed=42)
    daily_csv = tmp_path / "daily_orders.csv"
    df.to_csv(daily_csv, index=False)

    loaded = F.load_daily(WEATHER_CSV, daily_csv)
    assert len(loaded) == len(weather_df)
    assert "order_count" in loaded.columns
    assert loaded["order_count"].notna().all()
