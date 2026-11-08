"""Streamlit forecast dashboard for Review 2 task T22."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import load_dashboard_data
from dashboard.forecasting import fit_holdout_forecasts, forecast_future_orders
from dashboard.weather import fetch_weather_extension, local_today
from dashboard.analytics import mine_pair_rules, rolling_pair_lift

WEATHER_OPTIONS = ("Dry", "Rainy", "Hot")

st.set_page_config(page_title="Coimbatore Food Orders | Review 2", page_icon="🌦️", layout="wide")


@st.cache_data(show_spinner="Loading Coimbatore order and weather data...")
def load_data():
    return load_dashboard_data(ROOT)


@st.cache_data(show_spinner="Training chronological holdout models...")
def get_forecasts(daily: pd.DataFrame) -> pd.DataFrame:
    return fit_holdout_forecasts(daily)


@st.cache_data(ttl=3600, show_spinner="Fetching recent and forecast weather from Open-Meteo...")
def get_weather_extension(last_order_date: str, end_date: str) -> pd.DataFrame:
    return fetch_weather_extension(last_order_date, end_date=end_date)


@st.cache_data(show_spinner="Training demand models and projecting future dates...")
def get_future_forecasts(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    return forecast_future_orders(daily, weather)


@st.cache_data(show_spinner="Mining rainy and dry basket rules...")
def get_pair_rules(orders: pd.DataFrame, daily_orders: pd.DataFrame, min_support: float) -> pd.DataFrame:
    return mine_pair_rules(orders, daily_orders, min_support=min_support)


@st.cache_data(show_spinner="Calculating the selected rule's rolling lift...")
def get_rule_timeline(
    orders: pd.DataFrame,
    daily_orders: pd.DataFrame,
    antecedent: str,
    consequent: str,
    segment: str,
) -> pd.DataFrame:
    return rolling_pair_lift(
        orders,
        daily_orders,
        antecedent=antecedent,
        consequent=consequent,
        segment=segment,
    )


def weather_label(row: pd.Series) -> str:
    if row["is_rainy"]:
        return "Rainy"
    if row["is_hot"]:
        return "Hot"
    return "Dry"


def persistent_date_input(
    label: str,
    state_key: str,
    min_date,
    max_date,
    default_date,
):
    """Keep independent date selections stable across reruns and view switches."""
    widget_key = f"{state_key}_widget"
    saved_date = st.session_state.get(state_key, default_date)
    if not min_date <= saved_date <= max_date:
        saved_date = default_date

    widget_date = st.session_state.get(widget_key, saved_date)
    if not min_date <= widget_date <= max_date:
        widget_date = saved_date
    if widget_key not in st.session_state or st.session_state[widget_key] != widget_date:
        st.session_state[widget_key] = widget_date

    selected_date = st.sidebar.date_input(
        label,
        min_value=min_date,
        max_value=max_date,
        key=widget_key,
    )
    st.session_state[state_key] = selected_date
    return selected_date


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
        min_forecast_date = (last_order_date + pd.Timedelta(days=1)).date()
        max_forecast_date = pd.Timestamp("2027-01-01").date()
        if min_forecast_date > max_forecast_date:
            st.error("The latest order date is beyond the supported forecast window ending 1 Jan 2027.")
            return
        today = local_today()
        default_forecast_date = min(max(today.date(), min_forecast_date), max_forecast_date)
        start_date = persistent_date_input(
            "Forecast start date",
            "upcoming_forecast_start",
            min_forecast_date,
            max_forecast_date,
            default_forecast_date,
        )
        end_date = persistent_date_input(
            "Forecast end date",
            "upcoming_forecast_end",
            min_forecast_date,
            max_forecast_date,
            default_forecast_date,
        )
        if start_date > end_date:
            st.info("The forecast end date must be on or after the start date.")
            return
        if st.button("Refresh weather forecast"):
            get_weather_extension.clear()
            get_future_forecasts.clear()
        try:
            weather_extension = get_weather_extension(
                last_order_date.date().isoformat(),
                max_forecast_date.isoformat(),
            )
            forecasts = get_future_forecasts(daily, weather_extension)
        except (OSError, RuntimeError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Could not generate future forecasts: {error}")
            return

        forecast_details = forecasts.merge(
            weather_extension[
                [
                    "date",
                    "is_rainy",
                    "is_hot",
                    "temp_max_c",
                    "precip_mm",
                    "festival_name",
                ]
            ],
            on="date",
            how="left",
        )
        timeline = forecast_details.loc[
            forecast_details["date"].dt.date.between(start_date, end_date)
        ].copy()
        if timeline.empty:
            st.error(f"No forecasts are available from {start_date:%d %b %Y} to {end_date:%d %b %Y}.")
            return
        timeline["weather"] = timeline.apply(weather_label, axis=1)
        timeline = timeline.loc[timeline["weather"].isin(selected_weather)]
        if timeline.empty:
            st.info("No forecast days in this range match the weather filter.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Forecast period", f"{start_date:%d %b} – {end_date:%d %b %Y}")
        col2.metric("Mean predicted orders", f"{timeline[selected_model].mean():.1f}")
        col3.metric("Last observed order date", last_order_date.strftime("%d %b %Y"))
        st.subheader(f"{selected_model} forecast: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
        st.line_chart(
            timeline.set_index("date")[[selected_model]],
            x_label="Date",
            y_label="Projected orders per day",
        )
        st.caption(
            f"{selected_model} is projected recursively over the selected range. Ridge is the "
            "dashboard default from the predictive-model comparison; the seven-day SARIMAX result "
            "and +25-order planning buffer in the Review 2 slides are separate and are not applied here."
        )
        st.subheader("Daily forecast, weather and occasions")
        daily_view = timeline[
            [
                "date",
                selected_model,
                "weather",
                "temp_max_c",
                "precip_mm",
                "festival_name",
            ]
        ].rename(
            columns={
                selected_model: "Predicted orders",
                "weather": "Weather",
                "temp_max_c": "High temp (°C)",
                "precip_mm": "Precipitation (mm)",
                "festival_name": "Festival / occasion",
            }
        )
        daily_view["Festival / occasion"] = daily_view["Festival / occasion"].replace("", "—")
        st.dataframe(daily_view, hide_index=True, use_container_width=True)

        st.caption("Uses Open-Meteo weather where available and historical monthly medians beyond its forecast horizon.")
        return

    forecasts = get_forecasts(daily)
    forecast_dates = forecasts["date"]
    min_date, max_date = forecast_dates.min().date(), forecast_dates.max().date()
    start_date = persistent_date_input(
        "Holdout start date",
        "holdout_start",
        min_date,
        max_date,
        min_date,
    )
    end_date = persistent_date_input(
        "Holdout end date",
        "holdout_end",
        min_date,
        max_date,
        max_date,
    )
    if start_date > end_date:
        st.info("The holdout end date must be on or after the start date.")
        return
    weather = daily[["date", "is_rainy", "is_hot"]].copy()
    weather["weather"] = weather.apply(weather_label, axis=1)
    view = forecasts.merge(weather[["date", "weather"]], on="date", how="left")
    view = view.loc[
        view["date"].dt.date.between(start_date, end_date)
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
    st.line_chart(
        view.set_index("date")[["actual", selected_model]],
        x_label="Date",
        y_label="Orders per day",
    )
    st.caption(
        "Historical evaluation on a chronological 56-day holdout. Same-day observed weather is used, "
        "so this retrospective result is an optimistic upper bound for a live forecast."
    )


def rules_view(orders: pd.DataFrame, daily_orders: pd.DataFrame) -> None:
    st.header("Rainy vs dry basket rules")
    st.caption(
        "Rules are mined from the generated order baskets. Support is the share of orders containing both items; "
        "confidence is the share of antecedent orders that also contain the consequent. "
        "Pair counts can be small, so read lift as exploratory."
    )
    min_support = st.slider(
        "Minimum pair support",
        min_value=0.00001,
        max_value=0.0002,
        value=0.00005,
        step=0.00001,
        format="%.5f",
    )
    min_lift = st.slider("Minimum lift", min_value=1.0, max_value=5.0, value=1.2, step=0.1)
    rules = get_pair_rules(orders, daily_orders, min_support)
    rainy_col, dry_col = st.columns(2)
    table_columns = ["antecedent", "consequent", "support", "confidence", "lift", "pair_count"]
    for column, segment, title in (
        (rainy_col, "Rainy", "Rainy days"),
        (dry_col, "Dry", "Dry days"),
    ):
        selected = rules.loc[(rules["segment"] == segment) & (rules["lift"] >= min_lift)]
        column.subheader(title)
        if selected.empty:
            column.info("No rules meet the current support and lift thresholds.")
            continue
        column.dataframe(
            selected[table_columns].head(25).style.format({
                "support": "{:.2%}",
                "confidence": "{:.1%}",
                "lift": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    eligible = rules.loc[rules["lift"] >= min_lift]
    if eligible.empty:
        st.info("Lower the support or lift threshold to choose a rule for the timeline.")
        return

    eligible = eligible.sort_values(["segment", "lift", "support"], ascending=[True, False, False])
    options = [
        (str(row.segment), str(row.antecedent), str(row.consequent))
        for row in eligible.itertuples(index=False)
    ]
    labels = {option: f"{option[0]}: {option[1]} → {option[2]}" for option in options}
    choice = st.selectbox("Rule for the 14-day lift timeline", options, format_func=labels.__getitem__)
    timeline = get_rule_timeline(orders, daily_orders, choice[1], choice[2], choice[0])
    plot_timeline(timeline, choice[1], choice[2], choice[0])


def plot_timeline(timeline: pd.DataFrame, antecedent: str, consequent: str, segment: str) -> None:
    available = timeline.dropna(subset=["lift"])
    if available.empty:
        st.info("There are not enough matching baskets in any 14-day window to chart lift.")
        return

    fig, axis = plt.subplots(figsize=(11, 4))
    colors = {
        "SW monsoon (Jun-Sep)": "#3b82f6",
        "NE monsoon (Oct-Dec)": "#f59e0b",
        "Dry season (Jan-May)": "#64748b",
    }
    for phase, group in available.groupby("monsoon_phase", sort=False):
        axis.scatter(group["date"], group["lift"], s=18, color=colors[phase], label=phase)
    axis.plot(available["date"], available["lift"], color="#334155", linewidth=1, alpha=0.65)
    axis.axhline(1, color="#64748b", linestyle="--", linewidth=1, label="No association (lift = 1)")
    axis.set(
        title=f"14-day rolling lift: {antecedent} → {consequent} ({segment} orders)",
        xlabel="Date",
        ylabel="Lift",
    )
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.legend(loc="best", frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Each point uses only rainy or dry orders in its trailing 14 calendar days; "
        "windows with fewer than 20 orders are omitted."
    )


def main() -> None:
    st.title("🌦️ Coimbatore weather & food-order dashboard")
    st.write("Explore weather-driven demand and basket combinations.")
    try:
        daily, orders, daily_orders = load_data()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load dashboard data: {error}")
        st.stop()

    forecast_tab, rules_tab = st.tabs(["Forecast view", "Rain rules & monsoon timeline"])
    with forecast_tab:
        forecast_view(daily)
    with rules_tab:
        rules_view(orders, daily_orders)


if __name__ == "__main__":
    main()
