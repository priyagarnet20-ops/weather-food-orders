"""Streamlit forecast dashboard for Review 2 task T22."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import load_dashboard_data
from dashboard.forecasting import fit_holdout_forecasts, forecast_future_orders
from dashboard.weather import fetch_weather_extension, local_today

WEATHER_OPTIONS = ("Dry", "Rainy", "Hot")

st.set_page_config(page_title="Coimbatore Food Orders | Review 2", page_icon="🌦️", layout="wide")


@st.cache_data(show_spinner="Loading Coimbatore order and weather data...")
def load_data():
    return load_dashboard_data(ROOT)


@st.cache_data(show_spinner="Training chronological holdout models...")
def get_forecasts(daily: pd.DataFrame) -> pd.DataFrame:
    return fit_holdout_forecasts(daily)


@st.cache_data(ttl=3600, show_spinner="Fetching recent and forecast weather from Open-Meteo...")
def get_weather_extension(last_order_date: str) -> pd.DataFrame:
    return fetch_weather_extension(last_order_date)


@st.cache_data(show_spinner="Training demand models and projecting future dates...")
def get_future_forecasts(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    return forecast_future_orders(daily, weather)


def weather_label(row: pd.Series) -> str:
    if row["is_rainy"]:
        return "Rainy"
    if row["is_hot"]:
        return "Hot"
    return "Dry"


def forecast_view(daily: pd.DataFrame) -> None:
    st.header("Order forecast")
    forecast_mode = st.radio(
        "Forecast type",
        ("Upcoming dates", "Historical holdout evaluation"),
        horizontal=True,
    )
    selected_weather = st.sidebar.multiselect("Weather filter", WEATHER_OPTIONS, default=list(WEATHER_OPTIONS))
    models = ["Seasonal naive", "Ridge", "Random Forest", "XGBoost"]
    selected_model = st.selectbox("Forecast model", models, index=models.index("Ridge"))

    if forecast_mode == "Upcoming dates":
        last_order_date = pd.to_datetime(daily["date"]).max().normalize()
        if st.button("Refresh weather forecast"):
            get_weather_extension.clear()
            get_future_forecasts.clear()
        try:
            weather_extension = get_weather_extension(last_order_date.date().isoformat())
            forecasts = get_future_forecasts(daily, weather_extension)
        except (OSError, RuntimeError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Could not generate future forecasts: {error}")
            return

        today = local_today()
        future_weather = weather_extension.loc[weather_extension["date"] >= today].copy()
        view = forecasts.loc[forecasts["date"] >= today].merge(
            future_weather[["date", "is_rainy", "is_hot"]],
            on="date",
            how="left",
        )
        if view.empty:
            st.error("Open-Meteo returned no forecast dates from today onward.")
            return
        view["weather"] = view.apply(weather_label, axis=1)

        min_date, max_date = view["date"].min().date(), view["date"].max().date()
        date_range = st.sidebar.date_input(
            "Forecast date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            st.info("Select both a start and end date.")
            return
        view = view.loc[
            view["date"].dt.date.between(date_range[0], date_range[1])
            & view["weather"].isin(selected_weather)
        ]
        if view.empty:
            st.info("No upcoming forecast days match these filters.")
            return

        gap_days = max(0, (today - last_order_date).days - 1)
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecast days shown", f"{len(view):,}")
        col2.metric("Mean forecast orders", f"{view[selected_model].mean():.1f}")
        col3.metric("Last observed order date", last_order_date.strftime("%d %b %Y"))
        st.line_chart(view.set_index("date")[[selected_model]])
        st.caption("Vertical axis: projected orders per day.")
        if gap_days:
            st.warning(
                f"The available order data ends on {last_order_date:%d %b %Y}, {gap_days} days before today. "
                "The model recursively projects demand across that gap using observed weather, then "
                "uses Open-Meteo's current 16-day weather forecast. These are exploratory projections, "
                "not reliable operational forecasts; update the order data before using them for staffing or inventory."
            )
        else:
            st.caption("Uses the latest available orders and Open-Meteo's current 16-day weather forecast.")
        return

    forecasts = get_forecasts(daily)
    forecast_dates = forecasts["date"]
    min_date, max_date = forecast_dates.min().date(), forecast_dates.max().date()
    date_range = st.sidebar.date_input(
        "Holdout date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="holdout_date_range",
    )
    weather = daily[["date", "is_rainy", "is_hot"]].copy()
    weather["weather"] = weather.apply(weather_label, axis=1)
    view = forecasts.merge(weather[["date", "weather"]], on="date", how="left")
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Select both a start and end date.")
        return
    view = view.loc[
        view["date"].dt.date.between(date_range[0], date_range[1])
        & view["weather"].isin(selected_weather)
    ]
    if view.empty:
        st.info("No historical holdout days match these filters.")
        return

    actual_mae = (view["actual"] - view[selected_model]).abs().mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Days shown", f"{len(view):,}")
    col2.metric("Mean actual orders", f"{view['actual'].mean():.1f}")
    col3.metric(f"{selected_model} MAE", f"{actual_mae:.1f} orders")
    st.line_chart(view.set_index("date")[["actual", selected_model]])
    st.caption(
        "Historical evaluation on a chronological 56-day holdout. Same-day observed weather is used, "
        "so this retrospective result is an optimistic upper bound for a live forecast."
    )


def main() -> None:
    st.title("🌦️ Coimbatore weather & food-order dashboard")
    st.write("Explore weather-driven daily food-order demand and upcoming forecasts.")
    try:
        daily, _, _ = load_data()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load dashboard data: {error}")
        st.stop()

    forecast_view(daily)


if __name__ == "__main__":
    main()
