"""Tests for src/model_selection.py: no future leakage, statistics, the decision rule and the data profile."""
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
def full():
    return predict.load_model_frame(WEATHER, ORDERS)


# ---------------------------------------------------------------- data
def test_test_window_is_the_last_year(full):
    X = predict.make_features(full)
    first = ms.first_test_day(X, initial=len(X) - ms.TEST_DAYS)
    assert (X.index >= first).sum() == 365
    assert X.index[-1] == pd.Timestamp("2026-09-24")


def test_no_weather_columns_drop_every_weather_input(full):
    cols = ms.no_weather_columns(predict.make_features(full))
    assert "rain_x_weekend" not in cols
    assert not set(predict.WEATHER_FEATURES) & set(cols)
    assert "is_fri_sun" in cols and "lag7" in cols


def test_generator_expected_value_lines_up_with_the_dates(full):
    X = predict.make_features(full)
    ref = ms.generator_expected(ORDERS, X.index[-10:])
    daily = pd.read_csv(ORDERS, parse_dates=["date"]).set_index("date")
    pd.testing.assert_series_equal(ref, daily["expected_orders"].loc[X.index[-10:]], check_names=False, check_freq=False)


# ---------------------------------------------------------------- backtest
def _small_models():
    models = ms.make_models(quick=True)
    return {k: models[k] for k in FAST_MODELS}


def test_backtest_scores_each_test_day_once_for_every_model(full):
    X = predict.make_features(full)
    y = full["order_count"].loc[X.index]
    first = X.index[-21]
    preds = ms.rolling_backtest(X, y, _small_models(), first)
    assert list(preds.index) == list(X.index[-21:])
    assert set(preds.columns) == set(FAST_MODELS)
    assert not preds.isna().any().any()


def test_changing_an_actual_never_changes_forecasts_up_to_that_day(full):
    """The core no-leak guarantee: forecasts for day t and earlier cannot depend on the actual of day t."""
    X = predict.make_features(full)
    y = full["order_count"].loc[X.index]
    first = X.index[-21]
    base = ms.rolling_backtest(X, y, _small_models(), first)

    changed_day = X.index[-11]
    tampered = full.copy()
    tampered.loc[changed_day, "order_count"] += 500
    X2 = predict.make_features(tampered)
    y2 = tampered["order_count"].loc[X2.index]
    after = ms.rolling_backtest(X2, y2, _small_models(), first)

    upto = base.index <= changed_day
    pd.testing.assert_frame_equal(base.loc[upto], after.loc[upto])
    # The change must reach later forecasts, otherwise the test above proves nothing.
    assert not base.loc[~upto].equals(after.loc[~upto])


def test_block_is_never_longer_than_the_refit_interval(full):
    X = predict.make_features(full)
    y = full["order_count"].loc[X.index]
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


def test_best_possible_reference_never_wins():
    ref = dict(MAE=5.0, delta_MAE=-5.0, dm_p_holm=0.0001, ci_low=-6.0, ci_high=-4.0)
    assert ms.decide(_decision_table(**{ms.REFERENCE: ref}))["winner"] == "Ridge"


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


def test_profile_detects_the_rain_weekend_interaction():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2025-01-01", periods=400)
    wk = (idx.day_of_week >= 4).astype(float)
    rain = (rng.random(400) < 0.3).astype(float)
    temp = rng.normal(30, 3, 400)  # a continuous input, so RESET has a non-degenerate fit to test
    y = pd.Series(100 + 100 * wk + 50 * rain - 120 * rain * wk + 2 * temp + rng.normal(0, 5, 400), index=idx)
    X = pd.DataFrame({"is_fri_sun": wk, "is_rainy": rain, "rain_x_weekend": rain * wk, "temp": temp,
                      **{f"dow_{i}": (idx.day_of_week == i).astype(float) for i in range(7)}}, index=idx)
    table = ms.profile_data(X, y).set_index("trait")
    assert table.loc["Rain x weekend F-test p-value", "value"] < 0.001
    assert table.loc["RESET p-value", "value"] > 0.05  # with the interaction the linear form fits
