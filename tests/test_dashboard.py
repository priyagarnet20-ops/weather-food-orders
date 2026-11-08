"""Tests for the Review 2 forecast dashboard."""
import pandas as pd
import pytest

from dashboard.analytics import mine_pair_rules, rolling_pair_lift
from dashboard.forecasting import _future_feature_row, forecast_future_orders
from dashboard.weather import fetch_weather_extension, local_today


@pytest.fixture
def sample_orders():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2025-01-02", "2025-01-02", "2025-01-02",
            "2025-01-03", "2025-01-03", "2025-01-03",
        ]),
        "items": ["A; B", "A; B", "A; C", "A; B", "A; C", "C"],
    })


@pytest.fixture
def sample_weather():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "is_rainy": [True, False],
    })


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


def test_pair_rules_are_mined_separately_by_weather(sample_orders, sample_weather):
    rules = mine_pair_rules(sample_orders, sample_weather, min_support=0)
    rainy_ab = rules.loc[
        (rules["segment"] == "Rainy")
        & (rules["antecedent"] == "A")
        & (rules["consequent"] == "B")
    ].iloc[0]
    dry_ab = rules.loc[
        (rules["segment"] == "Dry")
        & (rules["antecedent"] == "A")
        & (rules["consequent"] == "B")
    ].iloc[0]

    assert rainy_ab["support"] == pytest.approx(2 / 3)
    assert rainy_ab["confidence"] == pytest.approx(2 / 3)
    assert rainy_ab["lift"] == pytest.approx(1.0)
    assert dry_ab["support"] == pytest.approx(1 / 3)
    assert dry_ab["transactions"] == 3


def test_pair_rules_apply_minimum_support(sample_orders, sample_weather):
    rules = mine_pair_rules(sample_orders, sample_weather, min_support=0.5)
    assert (rules["support"] >= 0.5).all()


def test_invalid_rule_threshold_is_rejected(sample_orders, sample_weather):
    with pytest.raises(ValueError, match="between 0 and 1"):
        mine_pair_rules(sample_orders, sample_weather, min_support=1.1)


def test_rolling_lift_uses_full_calendar_windows_and_weather_segment():
    dates = pd.date_range("2025-01-01", periods=14, freq="D")
    orders = pd.DataFrame({
        "date": dates.repeat(2),
        "items": ["A; B", "A; C"] * len(dates),
    })
    weather = pd.DataFrame({"date": dates, "is_rainy": True})
    lift = rolling_pair_lift(
        orders,
        weather,
        antecedent="A",
        consequent="B",
        segment="Rainy",
        min_transactions=20,
    )

    assert lift["lift"].iloc[:-1].isna().all()
    assert lift["lift"].iloc[-1] == pytest.approx(1.0)
    assert lift["transactions"].iloc[-1] == 28
    assert lift["monsoon_phase"].iloc[-1] == "Dry season (Jan-May)"


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


def test_weather_extension_fills_missing_api_values_from_historical_month(
    tmp_path, monkeypatch
):
    historical = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "precip_mm": [2.0, 4.0],
        "rain_mm": [1.0, 3.0],
        "temp_max_c": [32.0, 34.0],
        "humidity_mean_pct": [60.0, 80.0],
    })
    weather_csv = tmp_path / "weather.csv"
    historical.to_csv(weather_csv, index=False)
    forecast = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "precip_mm": [0.0, float("nan")],
        "rain_mm": [0.0, float("nan")],
        "temp_max_c": [30.0, float("nan")],
        "humidity_mean_pct": [70.0, float("nan")],
    })
    monkeypatch.setattr("dashboard.weather._api_daily", lambda _url: forecast.copy())

    extension = fetch_weather_extension(
        "2025-01-01",
        weather_csv=weather_csv,
        today="2025-01-02",
        end_date="2025-01-15",
    )

    imputed = extension.loc[extension["date"] == pd.Timestamp("2025-01-03")].iloc[0]
    assert extension["date"].tolist() == list(pd.date_range("2025-01-02", "2025-01-15"))
    assert extension["weather_imputed"].tolist() == [False] + [True] * 13
    assert imputed[["precip_mm", "rain_mm", "temp_max_c", "humidity_mean_pct"]].tolist() == [
        3.0, 2.0, 33.0, 70.0
    ]
    assert imputed["is_rainy"] == 1
    assert imputed["is_hot"] == 1
    pongal = extension.loc[extension["date"] == pd.Timestamp("2025-01-14")].iloc[0]
    assert pongal["festival_name"] == "Pongal"
    assert pongal["is_festival"] == 1
