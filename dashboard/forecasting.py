"""Chronological evaluation and recursive future forecasts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("Seasonal naive", "Ridge", "Random Forest", "XGBoost")
WEATHER_FEATURES = ("precip_mm", "temp_max_c", "humidity_mean_pct", "is_rainy", "heavy_rain", "is_hot")


def _load_predict_module():
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import predict

    return predict


def fit_holdout_forecasts(daily: pd.DataFrame, test_days: int = 56) -> pd.DataFrame:
    """Fit the project prediction models and return actuals plus held-out predictions."""
    predict = _load_predict_module()
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")
    features = predict.make_features(frame)
    target = frame["order_count"].loc[features.index]
    train_x, test_x, train_y, test_y = predict.split_train_test(features, target, test_days=test_days)
    if train_x.empty or test_x.empty:
        raise ValueError("Not enough daily observations to create the requested holdout forecast.")

    _, predictions = predict.fit_and_predict(train_x, train_y, test_x, quick=True)
    result = pd.DataFrame({"date": test_y.index, "actual": test_y.to_numpy()})
    for model_name, values in predictions.items():
        result[model_name] = values.clip(lower=0).to_numpy()
    return result


def _future_feature_row(date: pd.Timestamp, prior_orders: list[float], weather: pd.Series) -> dict[str, float]:
    if len(prior_orders) < 7:
        raise ValueError("At least seven prior daily order values are required.")
    is_fri_sun = date.day_of_week in (4, 5, 6)
    is_rainy = float(weather["is_rainy"])
    row = {
        "lag1": prior_orders[-1],
        "lag7": prior_orders[-7],
        "roll7": float(np.mean(prior_orders[-7:])),
        **{column: float(weather[column]) for column in WEATHER_FEATURES},
        "is_fri_sun": float(is_fri_sun),
        "rain_x_weekend": is_rainy * float(is_fri_sun),
        "is_festival": float(weather.get("is_festival", 0)),
    }
    row.update({f"dow_{day}": float(date.day_of_week == day) for day in range(7)})
    row["phase_sw_monsoon"] = float(date.month in range(6, 10))
    row["phase_ne_monsoon"] = float(date.month in range(10, 13))
    return row


def forecast_future_orders(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Forecast each weather date recursively, independently for each model.

    `weather` must cover every date after the final observed order day without gaps.
    Historical weather can bridge a gap in available order data, while forecast weather drives
    the requested live horizon. Those recursive bridge values are estimates, not actuals.
    """
    predict = _load_predict_module()
    history = daily.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history = history.sort_values("date").set_index("date")
    if history.index.has_duplicates or history["order_count"].isna().any():
        raise ValueError("Training data must have unique dates and non-missing order counts.")
    if len(history) < 14:
        raise ValueError("At least 14 historical days are required for recursive forecasting.")

    future_weather = weather.copy()
    future_weather["date"] = pd.to_datetime(future_weather["date"]).dt.normalize()
    future_weather = future_weather.sort_values("date").set_index("date")
    if future_weather.empty:
        raise ValueError("No future weather dates were provided.")
    if future_weather.index.has_duplicates:
        raise ValueError("Future weather must contain one row per date.")
    expected_dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), future_weather.index[-1], freq="D")
    if not future_weather.index.equals(expected_dates):
        raise ValueError("Future weather must cover every date after the final order day without gaps.")
    required_weather = set(WEATHER_FEATURES)
    if not required_weather.issubset(future_weather.columns):
        raise ValueError(f"Future weather is missing required columns: {sorted(required_weather - set(future_weather.columns))}")
    if future_weather[list(required_weather)].isna().any().any():
        raise ValueError("Future weather must not contain missing model features.")

    features = predict.make_features(history)
    target = history["order_count"].loc[features.index]
    fitted, _ = predict.fit_and_predict(features, target, features.iloc[-1:], quick=True)
    output: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}

    for model_name in MODEL_NAMES:
        prior_orders = history["order_count"].astype(float).tolist()
        for date, weather_row in future_weather.iterrows():
            row = _future_feature_row(date, prior_orders, weather_row)
            model_features = pd.DataFrame([row], index=[date]).reindex(columns=features.columns, fill_value=0.0)
            if model_name == "Seasonal naive":
                prediction = row["lag7"]
            else:
                prediction = float(fitted[model_name].predict(model_features)[0])
            prediction = max(0.0, prediction)
            output[model_name].append(prediction)
            prior_orders.append(prediction)

    result = future_weather.reset_index()[["date"]].copy()
    for model_name, values in output.items():
        result[model_name] = values
    return result
