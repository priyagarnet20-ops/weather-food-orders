"""Unit tests for the evidence-based daily demand generator (T08).

Run: pytest -q tests/test_generator_daily.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import calibration as C  # noqa: E402
import forecast as F  # noqa: E402
import generator as G  # noqa: E402

WEATHER_CSV = ROOT / "data" / "raw" / "weather.csv"


@pytest.fixture(scope="module")
def cal():
    return C.calibrate()


@pytest.fixture(scope="module")
def weather_df():
    assert WEATHER_CSV.exists(), f"Missing weather CSV at {WEATHER_CSV}"
    return pd.read_csv(WEATHER_CSV)


@pytest.fixture(scope="module")
def daily(weather_df, cal):
    return G.generate_daily_demand(weather_df, cal, seed=42)


def test_seed_reproducibility(weather_df, cal, daily):
    again = G.generate_daily_demand(weather_df, cal, seed=42)
    pd.testing.assert_series_equal(daily["order_count"], again["order_count"])
    other = G.generate_daily_demand(weather_df, cal, seed=99)
    assert not daily["order_count"].equals(other["order_count"])


def test_covers_the_full_old_period(daily):
    assert len(daily) == 1097
    assert daily["date"].min() == "2023-09-24"
    assert daily["date"].max() == "2026-09-24"


def test_weekend_multiplier_comes_from_swiggy_fit(daily, cal):
    assert (daily.loc[daily["is_weekend"], "mult_weekend"] == cal.volume["weekend"]["value"]).all()
    assert (daily.loc[~daily["is_weekend"], "mult_weekend"] == 1.0).all()


def test_rain_lifts_weekdays_and_lowers_weekends(daily, cal):
    wkday_rain = daily["is_rainy"] & ~daily["is_weekend"]
    wkend_rain = daily["is_rainy"] & daily["is_weekend"]
    assert (daily.loc[wkday_rain, "mult_rain"] == cal.volume["rain_weekday"]["value"]).all()
    assert (daily.loc[wkend_rain, "mult_rain"] == cal.volume["rain_weekend"]["value"]).all()
    assert (daily.loc[~daily["is_rainy"], "mult_rain"] == 1.0).all()
    assert cal.volume["rain_weekday"]["value"] > 1 > cal.volume["rain_weekend"]["value"]


def test_heat_and_festival_multipliers(daily, cal):
    assert (daily.loc[daily["is_hot"], "mult_temp"] == cal.volume["heat"]["value"]).all()
    assert (daily.loc[daily["is_festival"], "mult_festival"] == cal.festival).all()
    assert (daily.loc[~daily["is_festival"], "mult_festival"] == 1.0).all()
    assert daily["is_festival"].sum() == len(G.COIMBATORE_FESTIVALS)


def test_regimes(daily):
    assert set(daily["regime"]) == {"rainy", "hot", "normal"}
    assert (daily.loc[daily["is_rainy"], "regime"] == "rainy").all()


def test_expected_orders_scale_one_outlet_to_outlets(weather_df, cal):
    one = G.generate_daily_demand(weather_df, cal, outlets=1)
    many = G.generate_daily_demand(weather_df, cal, outlets=G.OUTLETS)
    np.testing.assert_allclose(many["expected_orders"], G.OUTLETS * one["expected_orders"])


def test_negative_binomial_mean_convergence(daily):
    emp, exp = daily["order_count"].mean(), daily["expected_orders"].mean()
    assert abs(emp - exp) / exp < 0.02


def test_compatibility_with_forecast_load_daily(daily, tmp_path):
    daily_csv = tmp_path / "daily_orders.csv"
    daily.to_csv(daily_csv, index=False)
    loaded = F.load_daily(WEATHER_CSV, daily_csv)
    assert len(loaded) == len(daily)
    assert loaded["order_count"].notna().all()
