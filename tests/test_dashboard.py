"""Tests for the Review 2 forecast dashboard."""
import pandas as pd
import pytest

from dashboard.analytics import (
    PLANTED_RAIN_PAIRS,
    compare_weather_pair_rules,
    filter_weather_effect,
    mine_pair_rules,
    rolling_pair_lift,
)
from dashboard.app import (
    MODEL_SELECTION_MODELS,
    filter_daily_dataset_sample,
    filter_order_baskets_sample,
    filter_overview_period,
    forecast_weather_label,
    holdout_model_comparison,
    load_model_selection_results,
    model_selection_summary,
    select_holdout_models,
    weather_filter_mask,
)
from dashboard.data import load_dashboard_data
from dashboard.forecasting import MODEL_NAMES, _future_feature_row, forecast_future_orders
from dashboard.weather import fetch_weather_extension, local_today


def test_overview_period_filters_recent_window_and_keeps_all_history():
    dates = pd.date_range("2025-01-01", periods=400, freq="D")
    daily = pd.DataFrame({"date": dates, "order_count": range(len(dates))})

    recent = filter_overview_period(daily, "Last 90 days")
    all_history = filter_overview_period(daily, "All history")

    assert len(recent) == 90
    assert recent["date"].min() == dates[-90]
    pd.testing.assert_frame_equal(all_history, daily)


def test_overview_period_rejects_unknown_period():
    daily = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"])})

    with pytest.raises(ValueError, match="Unsupported overview period"):
        filter_overview_period(daily, "Last 30 days")


def test_weather_filter_includes_overlapping_conditions():
    forecast = pd.DataFrame({
        "is_rainy": [True, True, False, False],
        "is_hot": [False, True, True, False],
    })

    rainy = weather_filter_mask(forecast, ["Rainy"])
    hot = weather_filter_mask(forecast, ["Hot"])
    dry = weather_filter_mask(forecast, ["Dry"])

    assert rainy.tolist() == [True, True, False, False]
    assert hot.tolist() == [False, True, True, False]
    assert dry.tolist() == [False, False, False, True]


def test_forecast_weather_label_shows_rain_and_heat_together():
    forecast = pd.DataFrame({
        "is_rainy": [True, True, False, False],
        "is_hot": [False, True, True, False],
    })

    labels = forecast.apply(forecast_weather_label, axis=1).tolist()

    assert labels == ["Rainy", "Rainy + Hot", "Hot", "Dry"]


def test_holdout_model_comparison_ranks_models_by_mae():
    actual = [100.0, 100.0, 100.0]
    forecasts = pd.DataFrame(
        {
            "actual": actual,
            "Seasonal naive": [120.0, 120.0, 120.0],
            "Ridge": [110.0, 110.0, 110.0],
            "Random Forest": [105.0, 105.0, 105.0],
            "XGBoost": [115.0, 115.0, 115.0],
        }
    )

    comparison = holdout_model_comparison(forecasts)

    assert comparison["Model"].tolist() == [
        "Random Forest", "Ridge", "XGBoost", "Seasonal naive"
    ]
    assert comparison["MAE"].tolist() == [5.0, 10.0, 15.0, 20.0]


def test_holdout_model_comparison_rejects_missing_model():
    forecasts = pd.DataFrame({"actual": [100.0], "Ridge": [95.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        holdout_model_comparison(forecasts)


def test_holdout_model_selection_filters_comparison():
    actual = [100.0, 100.0]
    forecasts = pd.DataFrame(
        {
            "actual": actual,
            "Seasonal naive": [120.0, 120.0],
            "Ridge": [110.0, 110.0],
            "Random Forest": [105.0, 105.0],
            "XGBoost": [115.0, 115.0],
        }
    )
    comparison = holdout_model_comparison(forecasts)

    selected = select_holdout_models(comparison, ["Ridge", "XGBoost"])

    assert selected["Model"].tolist() == ["Ridge", "XGBoost"]
    assert selected["MAE"].tolist() == [10.0, 15.0]


def test_holdout_model_selection_rejects_unknown_model():
    comparison = pd.DataFrame({"Model": ["Ridge"], "MAE": [10.0]})

    with pytest.raises(ValueError, match="Unknown forecast models"):
        select_holdout_models(comparison, ["Not a model"])


def test_full_year_model_selection_marks_ties_and_keeps_ridge_selected():
    metrics = pd.DataFrame(
        {
            "MAE": [14.99, 14.99, 15.03, 15.08, 15.60, 16.06, 46.30],
            "dm_p_holm": [1.0, 1.0, 1.0, float("nan"), 0.182, 0.011, 0.0],
            "ci_low": [-0.317, -0.278, -0.262, float("nan"), 0.024, 0.373, 19.959],
            "ci_high": [0.138, 0.096, 0.155, float("nan"), 1.025, 1.614, 26.282],
        },
        index=[
            "SARIMAX",
            "Negative Binomial",
            "Poisson",
            "Ridge",
            "Random Forest",
            "XGBoost",
            "Seasonal naive",
        ],
    )

    summary = model_selection_summary(metrics).set_index("Model")

    assert summary.index[0] == "SARIMAX"
    assert summary.loc["SARIMAX", "Result"] == "Statistically tied with Ridge"
    assert summary.loc["Poisson", "Result"] == "Statistically tied with Ridge"
    assert summary.loc["Ridge", "Result"] == "Selected incumbent"
    assert summary.loc["Random Forest", "Result"] == "Higher MAE; not significant after correction"
    assert summary.loc["XGBoost", "Result"] == "Significantly worse than Ridge"


def test_saved_full_year_model_selection_results_cover_365_days():
    forecasts, metrics = load_model_selection_results()

    assert len(forecasts) == 365
    assert forecasts["date"].is_monotonic_increasing
    assert set(MODEL_SELECTION_MODELS).issubset(forecasts.columns)
    assert set(MODEL_SELECTION_MODELS).issubset(metrics.index)



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


def test_default_data_comparison_keeps_planted_pairs_and_drops_artifacts():
    _, orders, daily_orders = load_dashboard_data()
    rules = mine_pair_rules(orders, daily_orders, min_support=0.005)
    comparison = compare_weather_pair_rules(rules)
    planted = set(
        zip(
            comparison.loc[comparison["planted_rain_pair"], "antecedent"],
            comparison.loc[comparison["planted_rain_pair"], "consequent"],
        )
    )
    rainy_stronger = filter_weather_effect(comparison, "Stronger on rainy days")
    dry_stronger = filter_weather_effect(comparison, "Stronger on dry days")

    assert set(PLANTED_RAIN_PAIRS).issubset(planted)
    assert comparison["rainy_confidence"].lt(1).all()
    assert comparison["dry_confidence"].lt(1).all()
    assert not (
        comparison["antecedent"].eq("Ghee Roast")
        & comparison["consequent"].eq("Podi Roast")
    ).any()
    assert not rainy_stronger.empty
    assert not dry_stronger.empty
    assert not rainy_stronger.equals(dry_stronger)
    assert comparison["confidence_change_pp"].is_monotonic_decreasing
    assert comparison[["rainy_lift", "dry_lift"]].to_numpy().max() <= 20


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
        min_antecedent_transactions=14,
    )

    assert lift["lift"].iloc[:-1].isna().all()
    assert lift["lift"].iloc[-1] == pytest.approx(1.0)
    assert lift["transactions"].iloc[-1] == 28
    assert lift["monsoon_phase"].iloc[-1] == "Dry season (Jan-May)"


def test_rolling_lift_hides_windows_with_too_few_antecedent_orders():
    dates = pd.date_range("2025-01-01", periods=14, freq="D")
    orders = pd.DataFrame({
        "date": dates,
        "items": ["A; B"] * len(dates),
    })
    weather = pd.DataFrame({"date": dates, "is_rainy": True})

    lift = rolling_pair_lift(
        orders,
        weather,
        antecedent="A",
        consequent="B",
        segment="Rainy",
        min_transactions=1,
        min_antecedent_transactions=15,
    )

    assert lift["lift"].isna().all()
    assert lift["confidence"].isna().all()


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


def test_filter_daily_dataset_sample_returns_slices():
    daily_orders = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "precip_mm": [0.0, 20.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "temp_max_c": [30.0, 25.0, 36.0, 26.0, 35.0, 31.0, 30.0, 30.0, 30.0, 30.0],
        "is_rainy": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "is_festival": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "is_weekend": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        "order_count": [100, 150, 120, 140, 110, 105, 250, 230, 95, 102],
    })

    monsoon = filter_daily_dataset_sample(daily_orders, "monsoon", sample_size=5)
    assert len(monsoon) == 2
    assert (monsoon["precip_mm"] >= 15.0).all()

    heat = filter_daily_dataset_sample(daily_orders, "heat", sample_size=5)
    assert len(heat) == 2
    assert (heat["temp_max_c"] >= 34.0).all()

    festival = filter_daily_dataset_sample(daily_orders, "festival", sample_size=5)
    assert len(festival) == 2

    weekend_peak = filter_daily_dataset_sample(daily_orders, "weekend_peak", sample_size=1)
    assert len(weekend_peak) == 1
    assert weekend_peak.iloc[0]["order_count"] == 250

    head_sample = filter_daily_dataset_sample(daily_orders, "head", sample_size=3)
    assert len(head_sample) == 3
    assert head_sample.iloc[0]["date"] == pd.Timestamp("2024-01-01")

    tail_sample = filter_daily_dataset_sample(daily_orders, "tail", sample_size=3)
    assert len(tail_sample) == 3
    assert tail_sample.iloc[-1]["date"] == pd.Timestamp("2024-01-10")

    random_sample = filter_daily_dataset_sample(daily_orders, "random", sample_size=4)
    assert len(random_sample) == 4


def test_filter_order_baskets_sample_filters_slot_and_search():
    orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "meal_slot": ["Lunch", "Dinner", "Lunch", "Evening Snacks"],
        "items": [
            "Chicken Biryani; Kozhi Milagu Soup",
            "Mutton Biryani; Parotta",
            "Veg Meals; Curd",
            "Chai; Onion Pakora",
        ],
        "item_count": [2, 2, 2, 2],
        "order_value": [350.0, 420.0, 150.0, 70.0],
    })

    lunch_orders = filter_order_baskets_sample(orders, meal_slot="Lunch")
    assert len(lunch_orders) == 2
    assert (lunch_orders["meal_slot"] == "Lunch").all()

    biryani_orders = filter_order_baskets_sample(orders, search_dish="Biryani")
    assert len(biryani_orders) == 2

    pakora_lunch = filter_order_baskets_sample(orders, meal_slot="Lunch", search_dish="Pakora")
    assert len(pakora_lunch) == 0

    pakora_snack = filter_order_baskets_sample(orders, meal_slot="Evening Snacks", search_dish="Pakora")
    assert len(pakora_snack) == 1

