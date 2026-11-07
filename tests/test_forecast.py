"""Tests for src/forecast.py regressors: weather enters with its rain x weekend interaction."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import forecast as F  # noqa: E402


def test_load_daily_adds_rain_weekend_interaction():
    df = F.load_daily(ROOT / "data/raw/weather.csv", ROOT / "data/raw/daily_orders.csv")
    weekend = df["dow"].isin([4, 5, 6]).astype(int)
    pd.testing.assert_series_equal(df["rain_x_weekend"], df["is_rainy"] * weekend, check_names=False)


def test_exog_matrix_includes_interaction_only_with_weather():
    df = F.load_daily(ROOT / "data/raw/weather.csv", ROOT / "data/raw/daily_orders.csv")
    assert "rain_x_weekend" in F.exog_matrix(df, weather=True).columns
    assert "rain_x_weekend" not in F.exog_matrix(df, weather=False).columns
