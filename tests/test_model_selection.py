"""Tests for src/model_selection.py: real-only scoring, no future leakage, statistics and the decision rule."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model_selection as ms  # noqa: E402
import predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEATHER = ROOT / "data/raw/weather.csv"
ORDERS = ROOT / "data/raw/daily_orders.csv"
FAST_MODELS = ("Seasonal naive", "Rolling mean", "Weekday mean", "Ridge", "Negative Binomial", "SARIMAX")


@pytest.fixture(scope="module")
def real():
    return ms.load_real_frame(WEATHER, ORDERS)


@pytest.fixture(scope="module")
def full():
    return predict.load_model_frame(WEATHER, ORDERS)


# ---------------------------------------------------------------- data
def test_real_frame_holds_only_the_272_real_days(real):
    assert len(real) == 272
    assert real.index.min() == pd.Timestamp("2025-01-02")
    assert real.index.max() == pd.Timestamp("2025-09-30")


def test_first_test_day_leaves_the_initial_training_rows(real):
    X = predict.make_features(real)
    first = ms.first_test_day(X, initial=120)
    assert (X.index < first).sum() == 120
    assert (X.index >= first).sum() == 145


# ---------------------------------------------------------------- backtest
def _small_models():
    models = ms.make_models(quick=True)
    return {k: models[k] for k in FAST_MODELS}


def test_backtest_scores_each_test_day_once_for_every_model(real):
    X = predict.make_features(real)
    y = real["order_count"].loc[X.index]
    first = X.index[-21]
    preds = ms.rolling_backtest(X, y, _small_models(), first)
    assert list(preds.index) == list(X.index[-21:])
    assert set(preds.columns) == set(FAST_MODELS)
    assert not preds.isna().any().any()


def test_changing_an_actual_never_changes_forecasts_up_to_that_day(real):
    """The core no-leak guarantee: forecasts for day t and earlier cannot depend on the actual of day t."""
    X = predict.make_features(real)
    y = real["order_count"].loc[X.index]
    first = X.index[-21]
    base = ms.rolling_backtest(X, y, _small_models(), first)

    changed_day = X.index[-11]
    tampered = real.copy()
    tampered.loc[changed_day, "order_count"] += 500
    X2 = predict.make_features(tampered)
    y2 = tampered["order_count"].loc[X2.index]
    after = ms.rolling_backtest(X2, y2, _small_models(), first)

    upto = base.index <= changed_day
    pd.testing.assert_frame_equal(base.loc[upto], after.loc[upto])
    # The change must reach later forecasts, otherwise the test above proves nothing.
    assert not base.loc[~upto].equals(after.loc[~upto])


def test_block_is_never_longer_than_the_refit_interval(real):
    X = predict.make_features(real)
    y = real["order_count"].loc[X.index]
    seen = []

    def spy(Xtr, ytr):
        seen.append(Xtr.index.max())
        return lambda Xb, yb: pd.Series(0.0, index=Xb.index)

    ms.rolling_backtest(X, y, {"spy": spy}, X.index[-15], refit_every=7)
    assert len(seen) == 3  # 15 days at 7 per refit
    assert seen[0] == X.index[-16]  # training stops the day before the first forecast
    assert seen[1] == X.index[-9]


def test_weekday_mean_predicts_the_training_mean_of_that_weekday():
    idx = pd.date_range("2025-01-06", periods=14)  # a Monday
    X = pd.DataFrame({f"dow_{i}": (idx.dayofweek == i).astype(float) for i in range(7)}, index=idx)
    y = pd.Series(np.arange(14.0), index=idx)  # Mondays are 0 and 7, mean 3.5
    predictor = ms.make_models(quick=True)["Weekday mean"](X, y)
    next_monday = pd.DataFrame({f"dow_{i}": [float(i == 0)] for i in range(7)}, index=[pd.Timestamp("2025-01-20")])
    assert predictor(next_monday, None).iloc[0] == pytest.approx(3.5)


# ---------------------------------------------------------------- hybrid (version B)
def test_calibration_recovers_known_multipliers():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=2000)
    rainy = rng.random(2000) < 0.4
    df = pd.DataFrame({"is_rainy": rainy.astype(int), "heavy_rain": 0, "is_hot": 0, "is_festival": 0}, index=idx)
    mu = 200 * np.where(rainy, 0.8, 1.0) * np.where(idx.dayofweek.isin([4, 5, 6]), 1.1, 1.0)
    df["order_count"] = rng.negative_binomial(50, 50 / (50 + mu)).astype(float)
    params = ms.calibrate_generator(df, cutoff=idx[-1] + pd.Timedelta(days=1))
    assert params["base"] == pytest.approx(200, rel=0.03)
    assert params["multipliers"]["rain_mid"] == pytest.approx(0.8, abs=0.03)
    assert params["multipliers"]["weekend"] == pytest.approx(1.1, abs=0.03)
    assert params["r"] == pytest.approx(50, rel=0.25)


def test_calibration_ignores_every_day_from_the_cutoff_onward(real):
    cutoff = real.index[127]
    changed = real.copy()
    changed.loc[changed.index >= cutoff, "order_count"] += 1000
    assert ms.calibrate_generator(real, cutoff) == ms.calibrate_generator(changed, cutoff)


def test_rare_condition_gets_a_neutral_multiplier():
    idx = pd.date_range("2025-01-01", periods=100)
    df = pd.DataFrame({"is_rainy": (np.arange(100) % 2).astype(int), "heavy_rain": 0, "is_festival": 0,
                       "is_hot": (np.arange(100) == 3).astype(int),  # a single hot day cannot support an estimate
                       "order_count": 200.0 + np.arange(100) % 7}, index=idx)
    assert ms.calibrate_generator(df, idx[-1] + pd.Timedelta(days=1))["multipliers"]["hot"] == 1.0


def test_hybrid_keeps_real_days_unchanged_and_replaces_the_earlier_days(full, real):
    cutoff = real.index[127]
    hybrid = ms.build_hybrid_frame(full, real, cutoff)
    pd.testing.assert_series_equal(hybrid["order_count"].loc[real.index], real["order_count"])
    before = hybrid.index < real.index.min()
    assert before.sum() == 367  # all of 2024 plus 1 Jan 2025
    assert hybrid.index.is_monotonic_increasing
    assert not hybrid["order_count"].loc[before].equals(full["order_count"].loc[hybrid.index[before]])
    assert hybrid.index.max() == real.index.max()  # synthetic Q4 2025 is never used


def test_hybrid_test_rows_have_the_same_features_as_real_only(full, real):
    cutoff = real.index[127]
    X_real = predict.make_features(real)
    X_hyb = predict.make_features(ms.build_hybrid_frame(full, real, cutoff))
    test_days = X_real.index[X_real.index >= cutoff]
    pd.testing.assert_frame_equal(X_real.loc[test_days], X_hyb.loc[test_days])


# ---------------------------------------------------------------- statistics
def test_diebold_mariano_matches_hand_worked_value():
    # |e1|-|e2| = 1,2,3,4: mean 2.5, gamma0 1.25, DM = 2.5/sqrt(1.25/4) = 4.4721, HLN factor sqrt(3/4) -> 3.8730
    stat, p = ms.diebold_mariano(np.array([2.0, 3, 4, 5]), np.array([1.0, 1, 1, 1]))
    assert stat == pytest.approx(3.8730, abs=1e-3)
    assert p == pytest.approx(0.0305, abs=1e-3)  # two-sided t, 3 degrees of freedom


def test_diebold_mariano_sign_shows_which_model_is_worse():
    stat, _ = ms.diebold_mariano(np.array([1.0, 1, 1, 1]), np.array([2.0, 3, 4, 5]))
    assert stat < 0


def test_diebold_mariano_with_identical_errors_reports_no_difference():
    stat, p = ms.diebold_mariano(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert stat == 0.0 and p == 1.0


def test_holm_adjustment_matches_hand_worked_values():
    # sorted .01*3=.03, .03*2=.06, .04*1=.04 -> raised to .06 to stay monotone
    assert ms.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert ms.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_bootstrap_ci_of_a_constant_is_that_constant():
    assert ms.block_bootstrap_ci(np.full(30, 2.0)) == pytest.approx((2.0, 2.0))


def test_bootstrap_ci_widens_when_blocks_keep_weekly_runs_together():
    d = np.tile(np.r_[np.full(7, 5.0), np.full(7, -5.0)], 10)
    low1, high1 = ms.block_bootstrap_ci(d, block=1)
    low7, high7 = ms.block_bootstrap_ci(d, block=7)
    assert high7 - low7 > high1 - low1


def test_mase_scales_by_the_weekly_naive_error_of_the_training_window():
    train = pd.Series([10.0] * 7 + [14.0] * 7)  # weekly naive error is 4 on each of the 7 comparable days
    assert ms.mase(pd.Series([100.0, 100.0]), pd.Series([98.0, 102.0]), train) == pytest.approx(0.5)


# ---------------------------------------------------------------- comparison and decision
def test_compare_to_incumbent_reports_cost_difference_against_ridge():
    idx = pd.date_range("2025-05-01", periods=70)
    y = pd.Series(100.0, index=idx)
    preds = pd.DataFrame({"Ridge": 103.0, "Better": 101.0, "Worse": 106.0}, index=idx)
    table = ms.compare_to_incumbent(y, preds, train_y=pd.Series(np.arange(30.0)))
    assert table.loc["Better", "delta_MAE"] == pytest.approx(-2.0)
    assert table.loc["Worse", "delta_MAE"] == pytest.approx(3.0)
    assert np.isnan(table.loc["Ridge", "dm_p"])
    assert list(table.index) == ["Better", "Ridge", "Worse"]  # sorted best first


def _decision_table(**rows):
    base = {"Ridge": dict(MAE=10.0, delta_MAE=0.0, dm_p_holm=np.nan, ci_low=0.0, ci_high=0.0)}
    base.update(rows)
    return pd.DataFrame(base).T


@pytest.mark.parametrize("row, reason", [
    (dict(MAE=11.0, delta_MAE=1.0, dm_p_holm=0.001, ci_low=0.5, ci_high=1.5), "higher error"),
    (dict(MAE=9.0, delta_MAE=-1.0, dm_p_holm=0.20, ci_low=-1.5, ci_high=-0.5), "not significant"),
    (dict(MAE=9.0, delta_MAE=-1.0, dm_p_holm=0.01, ci_low=-2.0, ci_high=0.5), "interval includes zero"),
])
def test_ridge_stays_unless_a_candidate_clears_every_condition(row, reason):
    assert ms.decide(_decision_table(New=row))["winner"] == "Ridge", reason


def test_candidate_replaces_ridge_only_when_it_clears_every_condition():
    row = dict(MAE=9.0, delta_MAE=-1.0, dm_p_holm=0.01, ci_low=-2.0, ci_high=-0.2)
    verdict = ms.decide(_decision_table(New=row))
    assert verdict["winner"] == "New" and verdict["replaced"] is True


def test_lowest_error_wins_when_several_candidates_qualify():
    a = dict(MAE=9.0, delta_MAE=-1.0, dm_p_holm=0.01, ci_low=-2.0, ci_high=-0.2)
    b = dict(MAE=8.0, delta_MAE=-2.0, dm_p_holm=0.01, ci_low=-3.0, ci_high=-1.0)
    assert ms.decide(_decision_table(A=a, B=b))["winner"] == "B"


# ---------------------------------------------------------------- data profile
def _profile_row(table, trait):
    return table.set_index("trait").loc[trait]


def test_profile_flags_overdispersion_and_points_to_negative_binomial():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=140)
    y = pd.Series(rng.negative_binomial(5, 5 / 205, 140).astype(float), index=idx)
    X = pd.DataFrame({"a": rng.normal(size=140)}, index=idx)
    row = _profile_row(ms.profile_data(X, y), "Variance / mean")
    assert row["value"] > 1.5
    assert "Negative Binomial" in row["points_to"]


def test_profile_finds_a_weekly_cycle():
    idx = pd.date_range("2025-01-01", periods=140)
    y = pd.Series(100 + 20 * (idx.dayofweek == 5).astype(float), index=idx)
    X = pd.DataFrame({"a": np.arange(140.0)}, index=idx)
    assert _profile_row(ms.profile_data(X, y), "Autocorrelation at 7 days")["value"] > 0.9
