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
def get_weather_extension(last_order_date: str) -> pd.DataFrame:
    return fetch_weather_extension(last_order_date)


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
