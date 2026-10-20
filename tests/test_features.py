"""Tests for src/features.py (T14): no leakage, bins agree with flags, festival flag carried through."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features  # noqa: E402


def _frame(rain, tmax, orders=None):
    n = len(rain)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "rain_mm": rain,
        "temp_max_c": tmax,
        "order_count": orders if orders is not None else list(range(100, 100 + n)),
    })


def test_lags_use_only_past_values():
    df = _frame([0.0] * 12, [30.0] * 12)
    out = features.add_lag_features(df)
    assert out["order_count_lag1"].iloc[5] == df["order_count"].iloc[4]
    assert out["order_count_lag7"].iloc[9] == df["order_count"].iloc[2]
    assert out["order_count_roll7"].iloc[9] == pytest.approx(df["order_count"].iloc[2:9].mean())


def test_first_rows_have_no_lag_values():
    out = features.add_lag_features(_frame([0.0] * 10, [30.0] * 10))
    assert out["order_count_lag1"].isna().sum() == 1
    assert out["order_count_lag7"].isna().sum() == 7


@pytest.mark.parametrize("rain,expected", [
    (0.0, "none"), (0.1, "trace"), (2.4, "trace"),
    (2.5, "moderate"), (14.9, "moderate"), (15.0, "heavy"), (40.0, "heavy"),
])
def test_rain_bin_agrees_with_flags_at_thresholds(rain, expected):
    out = features.add_rain_bins(_frame([rain], [30.0]))
    assert out["rain_intensity"].iloc[0] == expected
    assert out["is_rainy"].iloc[0] == int(rain >= 2.5)
    assert out["heavy_rain"].iloc[0] == int(rain >= 15.0)


@pytest.mark.parametrize("tmax,expected", [
    (27.9, "cool"), (28.0, "warm"), (32.9, "warm"),
    (33.0, "hot"), (37.9, "hot"), (38.0, "extreme"),
])
def test_temp_band_agrees_with_hot_flag_at_thresholds(tmax, expected):
    out = features.add_temp_bands(_frame([0.0], [tmax]))
    assert out["temp_band"].iloc[0] == expected
    assert out["is_hot"].iloc[0] == int(tmax >= 33.0)


def test_weekend_is_friday_to_sunday():
    out = features.add_calendar_flags(_frame([0.0] * 7, [30.0] * 7))  # 2024-01-01 is a Monday
    assert out["is_weekend"].tolist() == [0, 0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("month,expected", [
    (1, "nw_monsoon"), (4, "nw_monsoon"), (5, "dry_season"), (8, "dry_season"),
    (10, "dry_season"), (11, "nw_monsoon"), (12, "nw_monsoon"),
])
def test_monsoon_phase_by_month(month, expected):
    df = _frame([0.0], [30.0]).assign(date=pd.Timestamp(2024, month, 15))
    assert features.add_monsoon_phase(df)["monsoon_phase"].iloc[0] == expected


def test_festival_flag_is_added_from_daily_orders(tmp_path):
    src = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                        "is_festival": [True, False, True], "order_count": [1, 2, 3]})
    csv = tmp_path / "daily_orders.csv"
    src.to_csv(csv, index=False)
    out = features.add_festival_flag(_frame([0.0] * 3, [30.0] * 3), csv)
    assert out["is_festival"].tolist() == [1, 0, 1]


def test_festival_flag_fails_loudly_when_dates_are_missing(tmp_path):
    src = pd.DataFrame({"date": ["2024-01-01"], "is_festival": [True]})
    csv = tmp_path / "daily_orders.csv"
    src.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="festival"):
        features.add_festival_flag(_frame([0.0] * 3, [30.0] * 3), csv)


@pytest.mark.skipif(not (ROOT / "data/processed/daily_features.csv").exists(), reason="processed data not present")
def test_processed_table_has_festival_flag_matching_the_generator():
    f = pd.read_csv(ROOT / "data/processed/daily_features.csv", parse_dates=["date"])
    raw = pd.read_csv(ROOT / "data/raw/daily_orders.csv", parse_dates=["date"])
    assert "is_festival" in f.columns
    merged = f.merge(raw[["date", "is_festival"]], on="date", suffixes=("", "_raw"))
    assert (merged["is_festival"].astype(int) == merged["is_festival_raw"].astype(int)).all()
    assert f["is_festival"].sum() == raw["is_festival"].sum()
    assert f["is_festival"].sum() > 0
