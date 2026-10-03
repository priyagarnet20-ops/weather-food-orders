"""Tests for src/predict.py (T25): features must never leak the target or future values."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def daily():
    return predict.load_model_frame(ROOT / "data/raw/weather.csv", ROOT / "data/raw/daily_orders.csv")


def test_frame_has_one_row_per_day_and_no_gaps(daily):
    assert daily.index.is_monotonic_increasing
    assert (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert daily["order_count"].notna().all()


def test_lag_features_use_only_past_values(daily):
    X = predict.make_features(daily)
    y = daily["order_count"]
    day = X.index[10]
    pos = y.index.get_loc(day)
    assert X.loc[day, "lag1"] == y.iloc[pos - 1]
    assert X.loc[day, "lag7"] == y.iloc[pos - 7]
    # rolling mean on day t must exclude day t itself
    assert X.loc[day, "roll7"] == pytest.approx(y.iloc[pos - 7:pos].mean())


def test_features_do_not_contain_target(daily):
    X = predict.make_features(daily)
    assert "order_count" not in X.columns
    assert not X.isna().any().any()


def test_time_split_is_chronological_and_sized(daily):
    X = predict.make_features(daily)
    y = daily["order_count"].loc[X.index]
    Xtr, Xte, ytr, yte = predict.split_train_test(X, y, test_days=56)
    assert len(Xte) == 56
    assert Xtr.index.max() < Xte.index.min()
    assert len(Xtr) + len(Xte) == len(X)


def test_seasonal_naive_equals_value_seven_days_earlier(daily):
    X = predict.make_features(daily)
    assert (predict.seasonal_naive_pred(X) == X["lag7"]).all()


def test_metrics_known_values():
    y = pd.Series([100.0, 200.0])
    m = predict.score(y, pd.Series([110.0, 180.0]))
    assert m["MAE"] == pytest.approx(15.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["MAPE_%"] == pytest.approx((10 + 10) / 2)
    assert m["MSE"] == pytest.approx(250.0)
    assert m["R2"] == pytest.approx(1 - 500 / 5000)


def test_fit_models_returns_all_four_predictions(daily):
    X = predict.make_features(daily)
    y = daily["order_count"].loc[X.index]
    Xtr, Xte, ytr, yte = predict.split_train_test(X, y, test_days=56)
    fitted, preds = predict.fit_and_predict(Xtr, ytr, Xte, quick=True)
    assert set(preds) == {"Seasonal naive", "Ridge", "Random Forest", "XGBoost"}
    assert all(len(p) == len(Xte) for p in preds.values())
