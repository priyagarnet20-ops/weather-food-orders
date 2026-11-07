"""Tests for the Review 2 forecast dashboard."""
import pandas as pd
import pytest

from dashboard.forecasting import _future_feature_row, forecast_future_orders
from dashboard.weather import local_today


def test_local_today_uses_project_timezone():
    today = local_today()
    assert today.tz is None


def test_future_features_match_the_project_model():
    prior_orders = [100.0] * 14
    dry_weather = pd.Series({
        "precip_mm": 0.0,
        "temp_max_c": 30.0,
        "humidity_mean_pct": 80.0,
        "is_rainy": 0,
        "heavy_rain": 0,
        "is_hot": 0,
    })
    rainy_weather = dry_weather.copy()
    rainy_weather["is_rainy"] = 1

    sw_monsoon = _future_feature_row(pd.Timestamp("2025-07-01"), prior_orders, dry_weather)
    ne_monsoon = _future_feature_row(pd.Timestamp("2025-10-01"), prior_orders, dry_weather)
    friday = _future_feature_row(pd.Timestamp("2025-01-03"), prior_orders, rainy_weather)

    assert sw_monsoon["phase_sw_monsoon"] == 1
    assert sw_monsoon["phase_ne_monsoon"] == 0
    assert ne_monsoon["phase_sw_monsoon"] == 0
    assert ne_monsoon["phase_ne_monsoon"] == 1
    assert friday["is_fri_sun"] == 1
    assert friday["rain_x_weekend"] == 1


def test_future_forecast_recursively_covers_weather_dates():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    daily = pd.DataFrame({
        "date": dates,
        "order_count": [150 + i % 7 for i in range(len(dates))],
        "precip_mm": 1.0,
        "temp_max_c": 30.0,
        "humidity_mean_pct": 80.0,
        "is_rainy": 0,
        "heavy_rain": 0,
        "is_hot": 0,
        "is_festival": 0,
    })
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=3, freq="D")
    future_weather = pd.DataFrame({
        "date": future_dates,
        "precip_mm": [0.0, 3.0, 18.0],
        "temp_max_c": [30.0, 31.0, 34.0],
        "humidity_mean_pct": [80.0, 85.0, 90.0],
        "is_rainy": [0, 1, 1],
        "heavy_rain": [0, 0, 1],
        "is_hot": [0, 0, 1],
        "is_festival": [0, 0, 0],
    })

    forecasts = forecast_future_orders(daily, future_weather)

    assert forecasts["date"].tolist() == future_dates.tolist()
    assert {"Seasonal naive", "Ridge", "Random Forest", "XGBoost"}.issubset(forecasts.columns)
    assert forecasts.drop(columns="date").notna().all().all()
    assert (forecasts.drop(columns="date") >= 0).all().all()


def test_future_forecast_rejects_weather_date_gaps():
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    daily = pd.DataFrame({
        "date": dates,
        "order_count": 100,
        "precip_mm": 0,
        "temp_max_c": 30,
        "humidity_mean_pct": 80,
        "is_rainy": 0,
        "heavy_rain": 0,
        "is_hot": 0,
        "is_festival": 0,
    })
    gap_weather = pd.DataFrame({
        "date": [dates[-1] + pd.Timedelta(days=2)],
        "precip_mm": [0],
        "temp_max_c": [30],
        "humidity_mean_pct": [80],
        "is_rainy": [0],
        "heavy_rain": [0],
        "is_hot": [0],
    })

    with pytest.raises(ValueError, match="without gaps"):
        forecast_future_orders(daily, gap_weather)
