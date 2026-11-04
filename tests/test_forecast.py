"""Offline tests for src/forecast.py (P1 time-series preparation, models and evaluation).

Run:  pytest -q tests/test_forecast.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import forecast as F  # noqa: E402

WEATHER = ROOT / "data" / "raw" / "weather.csv"
ORDERS = ROOT / "data" / "raw" / "daily_orders.csv"


def toy_frame(n=60, seed=0):
    """Small frame with the columns the models expect."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D", name="date")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_count": 200 + rng.normal(0, 10, n).round()}, index=idx)
    df["dow"] = df.index.dayofweek
    df["is_festival"] = 0
    df["is_rainy"] = (rng.random(n) < 0.3).astype(int)
    df["heavy_rain"] = 0
    df["is_hot"] = 0
    return df


# ---------------------------------------------------------------- data preparation
def test_load_daily_follows_order_dates_and_flags_synthetic():
    df = F.load_daily(WEATHER, ORDERS)
    assert len(df) == 731
    assert df.index.min() == pd.Timestamp("2024-01-01") and df.index.max() == pd.Timestamp("2025-12-31")
    assert df["order_count"].notna().all()
    assert df[["precip_mm", "temp_max_c"]].notna().all().all()
    real = df[df["is_synthetic"] == 0]
    assert len(real) == 272
    assert real.index.min() == pd.Timestamp("2025-01-02") and real.index.max() == pd.Timestamp("2025-09-30")


def test_load_daily_weather_flags_use_team_thresholds():
    df = F.load_daily(WEATHER, ORDERS)
    assert (df["is_rainy"] == (df["precip_mm"] >= 2.5)).all()
    assert (df["heavy_rain"] == (df["precip_mm"] >= 15.0)).all()
    assert (df["is_hot"] == (df["temp_max_c"] >= 33.0)).all()


def test_load_daily_aggregates_order_level_file(tmp_path):
    w = tmp_path / "w.csv"
    pd.DataFrame({"date": pd.date_range("2025-03-01", periods=3), "precip_mm": [0.0, 3.0, 20.0],
                  "temp_max_c": [30.0, 31.0, 34.0]}).to_csv(w, index=False)
    o = tmp_path / "o.csv"
    pd.DataFrame({"order_id": [1, 2, 3, 4, 5], "order_time": ["2025-03-01 09:00", "2025-03-01 10:00",
                  "2025-03-02 12:00", "2025-03-03 13:00", "2025-03-03 14:00"]}).to_csv(o, index=False)
    df = F.load_daily(w, o)
    assert df["order_count"].tolist() == [2.0, 1.0, 2.0]
    assert df["is_synthetic"].tolist() == [0, 0, 0]
    assert df["is_rainy"].tolist() == [0, 1, 1] and df["heavy_rain"].tolist() == [0, 0, 1]


def test_exog_matrix_columns():
    df = toy_frame()
    assert F.exog_matrix(df, weather=False, dow=True).shape[1] == 6 + 1
    assert F.exog_matrix(df, weather=True, dow=True).shape[1] == 6 + 1 + 3
    assert list(F.exog_matrix(df, weather=False, dow=False).columns) == ["is_festival"]


# ---------------------------------------------------------------- backtest mechanics
def test_rolling_origin_scores_every_row_after_initial_once():
    df = toy_frame(60)
    fc = F.rolling_origin(df, {"m": F.weekday_mean}, initial=20, horizon=7, step=7, verbose=False)
    assert fc["date"].is_unique
    assert list(fc["date"].sort_values()) == list(df.index[20:])    # includes the shorter last fold
    assert fc["fold"].nunique() == int(np.ceil(40 / 7))


@pytest.mark.parametrize("model", [F.weekday_mean, F.seasonal_naive, F.rolling_mean7])
def test_forecast_never_uses_the_day_it_predicts_or_later(model):
    """Changing actual orders on and after a forecast day must not change that day's or any earlier forecast."""
    df = toy_frame(60)
    base = F.rolling_origin(df, {"m": model}, initial=20, horizon=7, step=7, verbose=False)
    cut = df.index[40]
    changed = df.copy()
    changed.loc[changed.index >= cut, "order_count"] += 500
    alt = F.rolling_origin(changed, {"m": model}, initial=20, horizon=7, step=7, verbose=False)
    a = base[base["date"] < cut].set_index("date")["yhat"]
    b = alt[alt["date"] < cut].set_index("date")["yhat"]
    pd.testing.assert_series_equal(a, b)


def test_baselines_known_values():
    df = toy_frame(28)
    train, test = df.iloc[:21], df.iloc[21:28]
    assert F.rolling_mean7(train, test)["yhat"].nunique() == 1
    assert F.rolling_mean7(train, test)["yhat"].iloc[0] == pytest.approx(train["order_count"].iloc[-7:].mean())
    np.testing.assert_allclose(F.seasonal_naive(train, test)["yhat"].to_numpy(), train["order_count"].iloc[-7:].to_numpy())
    wd = F.weekday_mean(train, test)["yhat"]
    assert wd.iloc[0] == pytest.approx(train.loc[train["dow"] == test["dow"].iloc[0], "order_count"].mean())


def test_sarimax_forecast_shape_and_interval_order():
    df = toy_frame(80)
    spec = F.SarimaxSpec(order=(1, 0, 0), weather=True, dow=False)
    out = F.make_sarimax(spec)(df.iloc[:70], df.iloc[70:77])
    assert len(out) == 7 and out[["yhat", "lo80", "hi80"]].notna().all().all()
    assert (out["lo80"] < out["yhat"]).all() and (out["yhat"] < out["hi80"]).all()


# ---------------------------------------------------------------- evaluation
def _fc(errors_a, errors_b, n=60):
    d = pd.date_range("2025-01-01", periods=n)
    y = np.full(n, 200.0)
    rows = []
    for name, e in (("a", errors_a), ("b", errors_b)):
        rows.append(pd.DataFrame({"date": d, "model": name, "y": y, "yhat": y - e, "lo80": np.nan, "hi80": np.nan}))
    return pd.concat(rows)


def test_metrics_basic_values():
    fc = _fc(np.full(60, 10.0), np.full(60, -4.0))
    m = F.metrics(fc)
    assert m.loc["a", "MAE"] == pytest.approx(10.0) and m.loc["b", "MAE"] == pytest.approx(4.0)
    assert m.loc["a", "bias"] == pytest.approx(10.0) and m.loc["b", "bias"] == pytest.approx(-4.0)
    assert m.loc["a", "RMSE"] == pytest.approx(10.0)


def test_block_bootstrap_identical_models_give_zero_and_clear_winner_is_separable():
    same = F.paired_bootstrap_mae_diff(_fc(np.full(60, 5.0), np.full(60, 5.0)), "a", "b")
    assert same["mae_diff"] == 0 and same["ci95"] == (0.0, 0.0)
    rng = np.random.default_rng(1)
    better = _fc(rng.normal(0, 3, 60), rng.normal(0, 20, 60))
    r = F.paired_bootstrap_mae_diff(better, "a", "b")
    assert r["mae_diff"] < 0 and r["ci95"][1] < 0


def test_block_bootstrap_is_wider_than_iid_for_autocorrelated_errors():
    rng = np.random.default_rng(3)
    e = np.cumsum(rng.normal(0, 4, 140))            # strongly autocorrelated error level
    fc = _fc(e, np.zeros(140), n=140)
    wide = F.paired_bootstrap_mae_diff(fc, "a", "b", block=7)["ci95"]
    narrow = F.paired_bootstrap_mae_diff(fc, "a", "b", block=1)["ci95"]
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_planning_buffer_is_the_quantile_of_overshoot():
    fc = _fc(np.arange(60, dtype=float), np.zeros(60))       # y - yhat = 0..59 for model a
    b = F.planning_buffer(fc, "a", 0.90)
    assert b["buffer_orders"] == pytest.approx(np.quantile(np.arange(60), 0.90))


def test_weather_effect_recovers_a_planted_heavy_rain_effect():
    rng = np.random.default_rng(7)
    n = 240
    idx = pd.date_range("2025-01-01", periods=n, freq="D", name="date")
    heavy = (rng.random(n) < 0.15).astype(int)
    rainy = (rng.random(n) < 0.30).astype(int)               # moderate rain: no true effect
    hot = (rng.random(n) < 0.10).astype(int)                 # no true effect
    level = np.zeros(n)
    for t in range(1, n):                                    # AR(1) level, like the real series
        level[t] = 0.9 * level[t - 1] + rng.normal(0, 0.03)
    y = 200 * np.exp(level + np.log(0.85) * heavy + rng.normal(0, 0.03, n))
    df = pd.DataFrame({"order_count": y, "dow": idx.dayofweek, "is_festival": 0, "is_rainy": rainy,
                       "heavy_rain": heavy, "is_hot": hot}, index=idx)
    w = F.weather_effect(df, log=True)
    assert -20 < w.loc["heavy_rain", "pct_effect"] < -10     # true effect is -15%
    assert w.loc["heavy_rain", "p_value"] < 0.05
