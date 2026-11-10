"""Streamlit forecast dashboard for Review 2 task T22."""
from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import load_dashboard_data
from dashboard.forecasting import MODEL_NAMES, fit_holdout_forecasts, forecast_future_orders
from dashboard.weather import fetch_weather_extension, local_today
from dashboard.analytics import (
    PLANTED_RAIN_PAIRS,
    compare_weather_pair_rules,
    filter_weather_effect,
    mine_pair_rules,
    rolling_pair_lift,
)

WEATHER_OPTIONS = ("Dry", "Rainy", "Hot")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
FORECAST_MODEL = "Ridge"
MODEL_SELECTION_MODELS = (
    "SARIMAX",
    "Negative Binomial",
    "Poisson",
    "Ridge",
    "Random Forest",
    "XGBoost",
    "Seasonal naive",
)
OVERVIEW_PROJECT_SUMMARY = (
    "This capstone studies how Coimbatore weather affects both restaurant order volume and menu combinations. "
    "It combines 1,097 days of real Open-Meteo weather with synthetic orders calibrated from local restaurant "
    "records and a team survey. The analysis covers exploratory data analysis (Unit 1), leakage-aware predictive "
    "models (Unit 2), and rolling time-series forecasts plus weather-segmented association rules (Unit 3). "
    "Because order baskets are generated, the menu results demonstrate recovery of known generator patterns; "
    "they are not evidence of real customer purchasing behaviour."
)
OVERVIEW_FINDINGS = (
    ("Full-year rolling-origin evaluation", "365 days"),
    ("Ridge reference MAE (orders/day)", "15.08"),
    ("MAE increase without weather", "22.4 orders/day"),
)

st.set_page_config(
    page_title="Coimbatore Food Orders",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="Loading Coimbatore order and weather data...")
def load_data():
    return load_dashboard_data(ROOT)


@st.cache_data(show_spinner=False)
def load_dataset_csv(relative_path: str) -> bytes:
    """Read a project CSV for download from the Dataset tab."""
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    return path.read_bytes()


@st.cache_data(show_spinner=False)
def load_model_selection_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved rolling-origin predictions and full-year model metrics."""
    forecast_path = ROOT / "results" / "selection_backtest_forecasts.csv"
    metrics_path = ROOT / "results" / "selection_metrics.csv"
    forecasts = pd.read_csv(forecast_path, parse_dates=["date"])
    metrics = pd.read_csv(metrics_path, index_col=0)
    if "date" not in forecasts:
        raise ValueError("Saved model-selection forecasts are missing the date column.")
    if forecasts["date"].isna().any() or forecasts["date"].duplicated().any():
        raise ValueError("Saved model-selection forecasts must have one valid row per date.")
    missing_forecasts = set(MODEL_SELECTION_MODELS) - set(forecasts.columns)
    if missing_forecasts:
        raise ValueError(f"Saved forecasts are missing models: {sorted(missing_forecasts)}")
    required_metrics = {"MAE", "delta_MAE", "dm_p_holm", "ci_low", "ci_high"}
    missing_metrics = required_metrics - set(metrics.columns)
    if missing_metrics:
        raise ValueError(f"Saved model metrics are missing columns: {sorted(missing_metrics)}")
    missing_models = set(MODEL_SELECTION_MODELS) - set(metrics.index.astype(str))
    if missing_models:
        raise ValueError(f"Saved metrics are missing models: {sorted(missing_models)}")
    metrics.index = metrics.index.astype(str)
    forecasts = forecasts.sort_values("date").reset_index(drop=True)
    return forecasts, metrics


@st.cache_data(show_spinner="Training chronological holdout models...")
def get_forecasts(daily: pd.DataFrame) -> pd.DataFrame:
    return fit_holdout_forecasts(daily)


@st.cache_data(ttl=3600, show_spinner="Fetching recent and forecast weather from Open-Meteo...")
def get_weather_extension(last_order_date: str, end_date: str) -> pd.DataFrame:
    return fetch_weather_extension(last_order_date, end_date=end_date)


@st.cache_data(show_spinner="Training demand models and projecting future dates...")
def get_future_forecasts(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    return forecast_future_orders(daily, weather)


@st.cache_data(show_spinner="Comparing rainy and dry basket rules...")
def get_pair_comparisons(
    orders: pd.DataFrame,
    daily_orders: pd.DataFrame,
    min_support: float,
    min_confidence: float,
) -> pd.DataFrame:
    rules = mine_pair_rules(orders, daily_orders, min_support=min_support)
    return compare_weather_pair_rules(rules, min_confidence=min_confidence, max_lift=20.0)


@st.cache_data(show_spinner="Calculating planted-pair timelines...")
def get_pair_timeline(
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
        min_antecedent_transactions=20,
    )


def weather_label(row: pd.Series) -> str:
    if row["is_rainy"]:
        return "Rainy"
    if row["is_hot"]:
        return "Hot"
    return "Dry"


def forecast_weather_label(row: pd.Series) -> str:
    """Keep simultaneous rain and heat visible in forecast tables."""
    conditions = []
    if row["is_rainy"]:
        conditions.append("Rainy")
    if row["is_hot"]:
        conditions.append("Hot")
    return " + ".join(conditions) if conditions else "Dry"


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

    selected_date = st.date_input(
        label,
        min_value=min_date,
        max_value=max_date,
        key=widget_key,
    )
    st.session_state[state_key] = selected_date
    return selected_date


def add_weather_segment(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with mutually exclusive weather labels for comparisons."""
    result = daily.copy()
    result["weather"] = result.apply(weather_label, axis=1)
    return result


def filter_overview_period(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filter daily observations to a supported overview time window."""
    latest = pd.to_datetime(daily["date"]).max()
    if period == "Last 90 days":
        start = latest - pd.Timedelta(days=89)
    elif period == "Last 12 months":
        start = latest - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    elif period == "All history":
        return daily.copy()
    else:
        raise ValueError(f"Unsupported overview period: {period}")
    return daily.loc[pd.to_datetime(daily["date"]) >= start].copy()


def weather_filter_mask(frame: pd.DataFrame, selected_weather: tuple[str, ...] | list[str]) -> pd.Series:
    """Match overlapping forecast weather flags without hiding hot rainy days."""
    rainy = frame["is_rainy"].astype(bool)
    hot = frame["is_hot"].astype(bool)
    selected = set(selected_weather)
    mask = pd.Series(False, index=frame.index)
    if "Dry" in selected:
        mask |= ~rainy & ~hot
    if "Rainy" in selected:
        mask |= rainy
    if "Hot" in selected:
        mask |= hot
    return mask


def forecast_chart(frame: pd.DataFrame, series: list[tuple[str, str]]) -> None:
    """Plot forecasts with visible markers, including when a selected range has one day."""
    rows = []
    for _, row in frame[["date", *(column for column, _ in series)]].iterrows():
        rows.extend(
            {"date": row["date"], "series": label, "orders": float(row[column])}
            for column, label in series
        )
    labels = [label for _, label in series]
    palette = {
        "Actual": "#274c77",
        "Seasonal naive": "#7a8a99",
        "Ridge": "#c24170",
        "Random Forest": "#2d7f79",
        "XGBoost": "#d18727",
    }
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=48))
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%d %b")),
            y=alt.Y("orders:Q", title="Orders per day", axis=alt.Axis(format="d")),
            color=alt.Color(
                "series:N",
                title="Series",
                scale=alt.Scale(domain=labels, range=[palette[label] for label in labels]),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("orders:Q", title="Orders", format=".0f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)


def holdout_model_comparison(view: pd.DataFrame) -> pd.DataFrame:
    """Rank dashboard model predictions against actual orders on the selected holdout days."""
    required = {"actual", *MODEL_NAMES}
    missing = required - set(view.columns)
    if missing:
        raise ValueError(f"Holdout comparison is missing required columns: {sorted(missing)}")
    scores = pd.DataFrame(
        {
            "Model": MODEL_NAMES,
            "MAE": [
                (view["actual"] - view[model]).abs().mean()
                for model in MODEL_NAMES
            ],
        }
    )
    return scores.sort_values("MAE", ignore_index=True)


def select_holdout_models(
    comparison: pd.DataFrame,
    selected_models: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Keep only user-selected models in a historical comparison."""
    unknown = set(selected_models) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown forecast models: {sorted(unknown)}")
    return comparison.loc[comparison["Model"].isin(selected_models)].reset_index(drop=True)


def model_selection_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank saved model-selection metrics and explain each result against Ridge."""
    missing_columns = {"MAE", "dm_p_holm", "ci_low", "ci_high"} - set(metrics.columns)
    if missing_columns:
        raise ValueError(f"Model-selection metrics are missing columns: {sorted(missing_columns)}")
    missing_models = set(MODEL_SELECTION_MODELS) - set(metrics.index.astype(str))
    if missing_models:
        raise ValueError(f"Model-selection metrics are missing models: {sorted(missing_models)}")

    ranked = metrics.copy()
    ranked.index = ranked.index.astype(str)
    ranked = ranked.loc[list(MODEL_SELECTION_MODELS)].copy()
    ridge_mae = float(ranked.loc[FORECAST_MODEL, "MAE"])
    ranked["MAE gap"] = ranked["MAE"] - ridge_mae

    def result_label(model: str, row: pd.Series) -> str:
        if model == FORECAST_MODEL:
            return "Selected incumbent"
        gap = float(row["MAE gap"])
        p_value = float(row["dm_p_holm"])
        ci_low, ci_high = float(row["ci_low"]), float(row["ci_high"])
        if ci_low <= 0 <= ci_high and p_value >= 0.05:
            return "Statistically tied with Ridge"
        if gap > 0 and p_value < 0.05:
            return "Significantly worse than Ridge"
        if gap > 0:
            return "Higher MAE; not significant after correction"
        if gap < 0 and p_value < 0.05 and ci_high < 0:
            return "Significantly better than Ridge"
        return "Lower MAE; not significant"

    ranked["Result"] = [
        result_label(model, row)
        for model, row in ranked.iterrows()
    ]
    return ranked.sort_values("MAE").rename_axis("Model").reset_index()


def model_selection_view() -> None:
    st.subheader("Full-year model selection")
    try:
        forecasts, metrics = load_model_selection_results()
        comparison = model_selection_summary(metrics)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load the saved full-year model-selection results: {error}")
        return

    start_date = forecasts["date"].min()
    end_date = forecasts["date"].max()
    selected_row = comparison.loc[comparison["Model"] == FORECAST_MODEL].iloc[0]
    best_row = comparison.iloc[0]
    summary = st.columns(3)
    summary[0].metric("Evaluation period", f"{len(forecasts):,} days")
    summary[1].metric("Lowest recorded MAE", f"{best_row['MAE']:.2f}", delta=str(best_row["Model"]))
    summary[2].metric("Selected model", FORECAST_MODEL, delta=f"MAE {selected_row['MAE']:.2f}")
    st.caption(
        f"Rolling-origin evaluation on saved results from {start_date:%d %b %Y} "
        f"to {end_date:%d %b %Y}. Lower MAE indicates smaller average daily error."
    )

    palette = {
        "Selected incumbent": "#c24170",
        "Statistically tied with Ridge": "#2d7f79",
        "Higher MAE; not significant after correction": "#d18727",
        "Significantly worse than Ridge": "#b94a48",
        "Significantly better than Ridge": "#2d7f79",
        "Lower MAE; not significant": "#7a8a99",
    }
    chart = (
        alt.Chart(comparison)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("MAE:Q", title="Mean absolute error (orders/day)"),
            y=alt.Y("Model:N", title=None, sort=alt.SortField(field="MAE", order="ascending")),
            color=alt.Color(
                "Result:N",
                title="Selection result",
                scale=alt.Scale(domain=list(palette), range=list(palette.values())),
            ),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("MAE:Q", title="MAE", format=".2f"),
                alt.Tooltip("MAE gap:Q", title="MAE gap vs Ridge", format="+.2f"),
                alt.Tooltip("Result:N", title="Result"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

    table = comparison[
        ["Model", "MAE", "MAE gap", "dm_p_holm", "ci_low", "ci_high", "Result"]
    ].copy()
    table = table.rename(
        columns={
            "MAE gap": "MAE gap vs Ridge",
            "dm_p_holm": "Holm-adjusted p",
            "ci_low": "95% CI low",
            "ci_high": "95% CI high",
        }
    )
    table["MAE"] = table["MAE"].map(lambda value: f"{value:.2f}")
    table["MAE gap vs Ridge"] = table["MAE gap vs Ridge"].map(
        lambda value: "—" if abs(value) < 0.0005 else f"{value:+.2f}"
    )
    table["Holm-adjusted p"] = table["Holm-adjusted p"].map(
        lambda value: "—" if pd.isna(value) else ("<0.001" if value < 0.001 else f"{value:.3f}")
    )
    for column in ("95% CI low", "95% CI high"):
        table[column] = table[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:+.2f}"
        )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.info(
        "SARIMAX has the lowest recorded MAE, but its advantage over Ridge is only "
        f"{abs(float(best_row['MAE']) - float(selected_row['MAE'])):.2f} orders/day and is not "
        "statistically significant after Holm correction. Ridge remains the selected incumbent; "
        "no challenger met the pre-defined replacement rule."
    )
    st.caption(
        "Statistical ties are identified when the Holm-adjusted Diebold–Mariano p-value is at least "
        "0.05 and the 95% confidence interval for the MAE gap includes zero. This comparison uses "
        "the saved one-year backtest; upcoming forecasts continue to use Ridge."
    )


def overview_view(daily: pd.DataFrame) -> None:
    st.header("Demand overview")
    st.caption(
        "A concise view of order volume, weekday patterns and weather context for Coimbatore."
    )
    with st.container(border=True):
        st.markdown("**About this capstone**")
        st.write(OVERVIEW_PROJECT_SUMMARY)
    st.subheader("What the full-year evaluation found")
    finding_columns = st.columns(len(OVERVIEW_FINDINGS))
    for column, (label, value) in zip(finding_columns, OVERVIEW_FINDINGS):
        column.metric(label, value)
    st.caption(
        "Project-wide results: a rolling-origin evaluation used observed same-day weather, so its "
        "accuracy is optimistic for live use. Orders are synthetic; these results measure recovery "
        "of the documented generator patterns, not guaranteed restaurant performance."
    )
    period = st.selectbox(
        "Time period",
        ("All history", "Last 12 months", "Last 90 days"),
        key="overview_period",
    )
    view = filter_overview_period(daily, period).sort_values("date")
    if view.empty:
        st.info("There are no daily observations in this time period.")
        return

    rainy_mean = view.loc[view["is_rainy"].astype(bool), "order_count"].mean()
    dry_mean = view.loc[~view["is_rainy"].astype(bool), "order_count"].mean()
    rain_change = (rainy_mean / dry_mean - 1) * 100 if pd.notna(rainy_mean) and dry_mean else None
    weekday_mean = view.groupby("dow")["order_count"].mean().reindex(range(7))
    busiest_day = int(weekday_mean.idxmax())
    weekday_mean.index = [WEEKDAYS[int(day)] for day in weekday_mean.index]
    weekday_mean.name = "Average orders"
    weekday_mean = weekday_mean.to_frame()
    weekday_mean["Average orders"] = weekday_mean["Average orders"].round(1)

    weekend_mean = view.loc[view["is_weekend"].astype(bool), "order_count"].mean()
    weekday_mean_orders = view.loc[~view["is_weekend"].astype(bool), "order_count"].mean()
    weekend_change = (
        (weekend_mean / weekday_mean_orders - 1) * 100
        if pd.notna(weekend_mean) and weekday_mean_orders
        else None
    )
    metrics = st.columns(4)
    metrics[0].metric("Orders in period", f"{int(view['order_count'].sum()):,}")
    metrics[1].metric("Daily average", f"{view['order_count'].mean():.1f}")
    metrics[2].metric(
        "Busiest weekday",
        WEEKDAYS[busiest_day],
        delta=f"{weekday_mean.loc[WEEKDAYS[busiest_day], 'Average orders']:.1f} orders/day",
        delta_color="off",
    )
    metrics[3].metric(
        "Rain-day difference",
        f"{rainy_mean - dry_mean:+.1f}" if rain_change is not None else "N/A",
        delta=f"{rain_change:+.1f}% vs non-rainy" if rain_change is not None else None,
        delta_color="off",
        help="Difference in average orders per day: rainy-day average minus non-rainy-day average.",
    )
    st.caption(
        f"{view['date'].min():%d %b %Y} – {view['date'].max():%d %b %Y} · "
        f"{len(view):,} days · {period}"
    )

    st.subheader("Weekly demand trend")
    weekly = (
        view.assign(week=view["date"].dt.to_period("W-SUN").dt.start_time)
        .groupby("week")["order_count"]
        .mean()
        .sort_index()
    )
    trend = pd.DataFrame(
        {
            "Weekly average": weekly,
            "4-week average": weekly.rolling(4, min_periods=1).mean(),
        }
    )
    trend_data = (
        trend.rename_axis("week")
        .reset_index()
        .melt(id_vars="week", var_name="Series", value_name="Average orders")
    )
    trend_chart = (
        alt.Chart(trend_data)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=28))
        .encode(
            x=alt.X("week:T", title="Week starting", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("Average orders:Q", title="Average orders per day"),
            color=alt.Color(
                "Series:N",
                title="Measure",
                scale=alt.Scale(
                    domain=["Weekly average", "4-week average"],
                    range=["#78bdf5", "#1769aa"],
                ),
            ),
            tooltip=[
                alt.Tooltip("week:T", title="Week starting", format="%d %b %Y"),
                alt.Tooltip("Series:N", title="Measure"),
                alt.Tooltip("Average orders:Q", title="Orders/day", format=".1f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(trend_chart, use_container_width=True)
    st.caption(
        "**How to read this:** the light line is the average daily order count for each week; "
        "the dark line smooths it using the current and previous three weeks. This is a descriptive "
        "trend, not a forecast. **Source:** project-generated daily order counts; orders are synthetic "
        "and represent the documented generator, not observed customer demand."
    )

    weekday_col, weather_col = st.columns(2)
    with weekday_col:
        st.subheader("Average demand by weekday")
        st.bar_chart(
            weekday_mean,
            x_label="Day",
            y_label="Orders per day",
        )
        if weekend_change is not None:
            st.caption(
                f"Weekend orders average {weekend_change:+.1f}% versus weekdays in this period."
            )
    with weather_col:
        st.subheader("Rainy vs. non-rainy days")
        weather_summary = (
            view.assign(
                weather_group=view["is_rainy"]
                .astype(bool)
                .map({True: "Rainy", False: "Non-rainy"})
            )
            .groupby("weather_group")
            .agg(
                days=("date", "size"),
                average_orders=("order_count", "mean"),
                median_orders=("order_count", "median"),
            )
            .reindex(["Non-rainy", "Rainy"])
            .dropna()
        )
        st.dataframe(
            weather_summary.rename(
                columns={
                    "days": "Days",
                    "average_orders": "Average orders",
                    "median_orders": "Median orders",
                }
            ).style.format(
                {
                    "Days": "{:,.0f}",
                    "Average orders": "{:.1f}",
                    "Median orders": "{:.0f}",
                }
            ),
            use_container_width=True,
        )
        st.caption(
            "Rainy days use the source rainy-day flag; generated orders make this an exploratory association."
        )

    with st.expander("About this data"):
        st.markdown(
            f"- **Coverage:** {daily['date'].min():%d %b %Y} to {daily['date'].max():%d %b %Y}\n"
            f"- **Daily records:** {len(daily):,}\n"
            "- **Weather:** historical Coimbatore observations\n"
            "- **Orders:** synthetically generated baskets; patterns are illustrative and not causal"
        )


def dataset_view(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    daily_orders: pd.DataFrame,
) -> None:
    st.header("About the dataset")
    st.caption(
        "This project joins historical Coimbatore weather with generated restaurant orders "
        "to explore demand patterns. Weather is observed/reanalysis data; orders are synthetic."
    )

    daily = daily.sort_values("date")
    start_date = pd.to_datetime(daily["date"]).min()
    end_date = pd.to_datetime(daily["date"]).max()
    basket_items = (
        orders["items"]
        .dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )
    menu_item_count = int(basket_items.loc[basket_items.ne("")].nunique())

    summary = st.columns(4)
    coverage_years = (end_date - start_date).days / 365.25
    summary[0].metric("Coverage period", f"{coverage_years:.1f} years")
    summary[1].metric("Daily records", f"{len(daily):,}")
    summary[2].metric("Synthetic orders", f"{len(orders):,}")
    summary[3].metric("Menu items", f"{menu_item_count:,}")
    st.caption(f"Full date range: {start_date:%d %B %Y} to {end_date:%d %B %Y}")
    detail = st.columns(4)
    detail[0].metric("Average orders per day", f"{daily['order_count'].mean():.1f}")
    detail[1].metric(
        "Daily order range",
        f"{int(daily['order_count'].min()):,}–{int(daily['order_count'].max()):,}",
    )
    detail[2].metric(
        "Rainy days (>= 2.5 mm)",
        f"{int(daily['is_rainy'].astype(bool).sum()):,}",
    )
    detail[3].metric(
        "Festival days",
        f"{int(daily_orders['is_festival'].astype(bool).sum()):,}",
    )
    st.subheader("What is included")
    st.markdown(
        f"The analysis covers **{start_date:%d %B %Y} to {end_date:%d %B %Y}**. Each day combines "
        "weather observations with calendar features and a generated order count. The transaction "
        "file contains one generated basket per order, with its time, meal slot, items and value."
    )
    inventory = pd.DataFrame(
        [
            {
                "File": "data/raw/weather.csv",
                "Rows": len(daily),
                "Columns": 11,
                "Contents": "Daily weather observations from the Open-Meteo Archive API.",
            },
            {
                "File": "data/processed/daily_features.csv",
                "Rows": len(daily),
                "Columns": len(daily.columns),
                "Contents": "Joined daily weather, generated demand, calendar flags and model features.",
            },
            {
                "File": "data/raw/orders.csv",
                "Rows": len(orders),
                "Columns": len(orders.columns),
                "Contents": "One generated order per row; item names in each basket are separated by semicolons.",
            },
            {
                "File": "data/raw/daily_orders.csv",
                "Rows": len(daily_orders),
                "Columns": len(daily_orders.columns),
                "Contents": "Daily generated counts, weather regime, festival fields and generation parameters.",
            },
        ]
    )
    st.dataframe(inventory, hide_index=True, use_container_width=True)

    st.subheader("Download dataset files")
    st.caption("Download the CSV files used by the dashboard.")
    downloads = (
        ("Weather observations", "data/raw/weather.csv"),
        ("Daily features", "data/processed/daily_features.csv"),
        ("Generated orders", "data/raw/orders.csv"),
        ("Daily order summary", "data/raw/daily_orders.csv"),
    )
    download_columns = st.columns(2)
    for index, (label, relative_path) in enumerate(downloads):
        download_columns[index % 2].download_button(
            label=f"Download {label} CSV",
            data=load_dataset_csv(relative_path),
            file_name=Path(relative_path).name,
            mime="text/csv",
            key=f"dataset_download_{relative_path}",
            use_container_width=True,
        )

    st.subheader("Download real restaurant base data")
    st.caption(
        "Cleaned records from the Hotel Chinthamani Swiggy report. These real orders informed "
        "the generated history and are separate from it."
    )
    restaurant_downloads = (
        ("Restaurant orders", "data/real/chinthamani_orders.csv"),
        ("Restaurant order items", "data/real/chinthamani_items.csv"),
    )
    restaurant_columns = st.columns(2)
    for index, (label, relative_path) in enumerate(restaurant_downloads):
        restaurant_columns[index].download_button(
            label=f"Download {label} CSV",
            data=load_dataset_csv(relative_path),
            file_name=Path(relative_path).name,
            mime="text/csv",
            key=f"source_download_{relative_path}",
            use_container_width=True,
        )

    st.subheader("Download supporting calibration data")
    st.caption(
        "The survey responses informed demand and basket assumptions; outlet listings supplied "
        "snack and drink prices."
    )
    calibration_downloads = (
        ("Team survey responses", "data/real/survey_responses.csv"),
        ("Reference snack and drink prices", "data/reference/snack_drink_prices.csv"),
    )
    calibration_columns = st.columns(2)
    for index, (label, relative_path) in enumerate(calibration_downloads):
        calibration_columns[index].download_button(
            label=f"Download {label} CSV",
            data=load_dataset_csv(relative_path),
            file_name=Path(relative_path).name,
            mime="text/csv",
            key=f"source_download_{relative_path}",
            use_container_width=True,
        )

    st.subheader("Sources and construction")
    weather_source, order_source = st.columns(2)
    with weather_source:
        st.markdown("**Weather — real archive data**")
        st.write(
            "Daily historical weather for Coimbatore (11.00 N, 76.99 E) comes from the Open-Meteo "
            "Archive API, using the Asia/Kolkata time zone. The dataset includes precipitation, "
            "rainfall, temperature, wind, weather codes, daylight, rainy hours and humidity. "
            "Values are gridded reanalysis estimates rather than measurements from a city weather station."
        )
    with order_source:
        st.markdown("**Orders — generated from real source data**")
        st.write(
            "The three-year order history is generated, not a record of actual transactions. Its "
            "generation is informed by 464 orders from Hotel Chinthamani's Swiggy report, responses "
            "from 27 adult Coimbatore survey participants, and local menu-price references. These "
            "sources calibrate daily order volumes and basket composition; the resulting dataset "
            "adds weather-linked patterns for analysis."
        )

    st.subheader("Key fields")
    field_guide = pd.DataFrame(
        [
            {
                "Dataset": "Daily features",
                "Fields": "date, order_count, total_revenue, avg_order_value, avg_basket_size",
                "Meaning": "Calendar day, generated demand and daily order-value summaries.",
            },
            {
                "Dataset": "Weather",
                "Fields": "precip_mm, rain_mm, temp_max_c, temp_min_c, temp_mean_c, wind_max_kmh, weather_code, daylight_s, precip_hours, humidity_mean_pct",
                "Meaning": "Daily precipitation, temperature, wind, WMO condition and humidity measures.",
            },
            {
                "Dataset": "Weather flags",
                "Fields": "is_rainy, heavy_rain, is_hot, rain_intensity, temp_band",
                "Meaning": "Project thresholds: rainy >= 2.5 mm, heavy rain >= 15 mm, hot >= 33 C.",
            },
            {
                "Dataset": "Calendar",
                "Fields": "dow, is_weekend, is_festival, festival_name, monsoon_phase",
                "Meaning": "Weekday, project weekend (Friday–Sunday), festival and seasonal context.",
            },
            {
                "Dataset": "Demand history",
                "Fields": "order_count_lag1, order_count_lag7, order_count_roll7",
                "Meaning": "Previous-day, previous-week and rolling seven-day order features.",
            },
            {
                "Dataset": "Order baskets",
                "Fields": "order_id, date, timestamp, meal_slot, items, item_count, order_value",
                "Meaning": "Generated transaction identifiers, timing, semicolon-separated items and order value.",
            },
            {
                "Dataset": "Generator details",
                "Fields": "regime, mult_weekend, mult_rain, mult_temp, mult_festival, expected_orders",
                "Meaning": "Weather regime, generation multipliers and expected daily volume in daily_orders.csv.",
            },
        ]
    )
    st.dataframe(field_guide, hide_index=True, use_container_width=True)
    st.caption(
        "The first rows of lag-based features have missing values when earlier history is unavailable; "
        "this is expected and does not indicate missing source weather."
    )

    with st.expander("Generator assumptions and limitations"):
        st.markdown(
            "- Daily demand is generated from a negative-binomial process with a fixed seed. "
            "The documented calibration includes a Friday–Sunday multiplier of 2.47, weekday-rain "
            "multiplier of 1.56, weekend-rain multiplier of 0.54, heat multiplier of 1.07 and "
            "festival multiplier of 0.89.\n"
            "- Weather flags use project-specific thresholds, not official meteorological definitions.\n"
            "- The source restaurant report covers one outlet and a short period; the team survey "
            "contains 27 adults. Neither is representative of all Coimbatore restaurants or customers.\n"
            "- Menu association rules measure patterns intentionally introduced by the generator. "
            "They are useful for demonstrating the method, not for claiming observed buying behaviour."
        )


def demand_patterns_view(daily: pd.DataFrame) -> None:
    st.header("Demand patterns")
    st.caption("Compare order volume by weekday, weather and season using the project’s daily dataset.")
    weather_data = add_weather_segment(daily)

    weekday_summary = (
        weather_data.groupby("dow", observed=True)["order_count"]
        .mean()
        .reindex(range(7))
        .rename(index=dict(enumerate(WEEKDAYS)))
        .rename("Average orders")
        .to_frame()
    )
    weather_summary = (
        weather_data.groupby("weather", observed=True)
        .agg(
            days=("date", "size"),
            average_orders=("order_count", "mean"),
            median_orders=("order_count", "median"),
        )
        .reindex(["Dry", "Rainy", "Hot"])
        .dropna()
    )
    weather_summary["change_vs_dry"] = (
        weather_summary["average_orders"] / weather_summary.loc["Dry", "average_orders"] - 1
    ) * 100

    weekday_col, weather_col = st.columns(2)
    with weekday_col:
        st.subheader("By day of week")
        st.bar_chart(
            weekday_summary,
            x_label="Day",
            y_label="Average orders per day",
            color="#4d7898",
        )
    with weather_col:
        st.subheader("By weather")
        st.dataframe(
            weather_summary.rename(
                columns={
                    "days": "Days",
                    "average_orders": "Average orders",
                    "median_orders": "Median orders",
                    "change_vs_dry": "Change vs. dry",
                }
            ).style.format(
                {
                    "Average orders": "{:.1f}",
                    "Median orders": "{:.0f}",
                    "Change vs. dry": "{:+.1f}%",
                    "Days": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

    monthly = (
        weather_data.assign(month=weather_data["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")["order_count"]
        .mean()
        .rename("Average orders")
        .to_frame()
    )
    st.subheader("Monthly demand")
    st.line_chart(
        monthly,
        x_label="Month",
        y_label="Average orders per day",
        color="#8b245f",
    )
    st.caption(
        "Weather groups are mutually exclusive: rainy days are classified first, then hot non-rainy days, "
        "with the remaining days classified as dry."
    )


def forecast_view(daily: pd.DataFrame) -> None:
    st.header("Order forecast")
    forecast_mode = st.radio(
        "Forecast type",
        ("Upcoming dates", "Historical holdout evaluation", "Full-year model selection"),
        horizontal=True,
        key="forecast_mode",
    )
    if forecast_mode == "Full-year model selection":
        model_selection_view()
        return
    selected_weather = st.multiselect(
        "Weather filter",
        WEATHER_OPTIONS,
        default=list(WEATHER_OPTIONS),
        help="Rainy and hot conditions can overlap; selecting either includes days with that condition.",
    )
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
                    "weather_imputed",
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
        timeline["weather"] = timeline.apply(forecast_weather_label, axis=1)
        timeline = timeline.loc[weather_filter_mask(timeline, selected_weather)]
        if timeline.empty:
            st.info("No forecast days in this range match the weather filter.")
            return

        if "weather_imputed" not in timeline:
            timeline["weather_imputed"] = False
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecast period", f"{start_date:%d %b} – {end_date:%d %b %Y}")
        col2.metric("Mean predicted orders", f"{timeline[FORECAST_MODEL].mean():.0f}")
        col3.metric("Last observed order date", last_order_date.strftime("%d %b %Y"))
        st.subheader(f"Ridge forecast: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
        forecast_chart(timeline, [(FORECAST_MODEL, FORECAST_MODEL)])
        bridge_days = max(0, (start_date - last_order_date.date()).days - 1)
        st.caption(
            f"Ridge, the project's selected model, is projected recursively. "
            f"There are {bridge_days} intervening days "
            "between the last observed order date and the requested range; those order counts are "
            "estimates fed back into the lag features, so forecast uncertainty increases with distance. "
            "The 90th-percentile planning buffer from the latest backtest is about +25 orders overall "
            "(+27 rainy, +24 dry); no buffer is applied to the displayed forecast."
        )
        st.subheader("Daily forecast, weather and occasions")
        daily_view = timeline[
            [
                "date",
                FORECAST_MODEL,
                "weather",
                "temp_max_c",
                "precip_mm",
                "festival_name",
                "weather_imputed",
            ]
        ].rename(
            columns={
                FORECAST_MODEL: "Predicted orders",
                "weather": "Weather",
                "temp_max_c": "High temp (°C)",
                "precip_mm": "Precipitation (mm)",
                "festival_name": "Festival / occasion",
                "weather_imputed": "Weather estimated",
            }
        )
        daily_view["Festival / occasion"] = daily_view["Festival / occasion"].replace("", "—")
        st.dataframe(
            daily_view,
            hide_index=True,
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "Predicted orders": st.column_config.NumberColumn("Predicted orders", format="%.0f"),
                "High temp (°C)": st.column_config.NumberColumn("High temp (°C)", format="%.1f"),
                "Precipitation (mm)": st.column_config.NumberColumn("Precipitation (mm)", format="%.1f"),
                "Weather estimated": st.column_config.CheckboxColumn(
                    "Weather estimated",
                    help="Checked when monthly historical medians substitute for unavailable weather values.",
                ),
            },
        )

        st.caption(
            "Uses Open-Meteo weather where available. Rows marked Weather estimated use historical "
            "monthly medians beyond the available weather forecast horizon."
        )
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
    view = forecasts.merge(weather, on="date", how="left")
    view = view.loc[
        view["date"].dt.date.between(start_date, end_date)
        & weather_filter_mask(view, selected_weather)
    ]
    if view.empty:
        st.info("No historical holdout days match these filters.")
        return

    all_comparisons = holdout_model_comparison(view)
    selected_models = st.multiselect(
        "Models to compare with actual orders",
        options=list(MODEL_NAMES),
        default=[FORECAST_MODEL],
        help="Choose one or more models to show alongside actual historical orders.",
    )
    if not selected_models:
        st.info("Select at least one model to display the historical graph and conclusion.")
        return
    comparison = select_holdout_models(all_comparisons, selected_models)
    best_model = comparison.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Days shown", f"{len(view):,}")
    col2.metric("Mean actual orders", f"{view['actual'].mean():.1f}")
    col3.metric(
        "Lowest historical MAE",
        f"{best_model['MAE']:.1f} orders/day",
        delta=str(best_model["Model"]),
        delta_color="off",
    )
    st.subheader("Historical model comparison")
    st.caption(
        "Selected model predictions are compared with actual orders over the historical holdout dates."
    )
    forecast_chart(
        view,
        [("actual", "Actual"), *((model, model) for model in selected_models)],
    )
    comparison_chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title="Model", sort="-y"),
            y=alt.Y("MAE:Q", title="Mean absolute error (orders/day)"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=list(MODEL_NAMES),
                    range=["#7a8a99", "#c24170", "#2d7f79", "#d18727"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("MAE:Q", title="MAE", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(comparison_chart, use_container_width=True)
    st.success(
        (
            f"Conclusion for these historical dates: {best_model['Model']} has the lowest MAE "
            f"among the selected models ({best_model['MAE']:.1f} orders/day)."
            if len(comparison) > 1
            else f"Selected model: {best_model['Model']} | MAE {best_model['MAE']:.1f} orders/day."
        )
    )
    st.caption(
        "This is a retrospective comparison, not a future forecast. Same-day observed weather makes it "
        "an optimistic upper bound for live use. Random Forest and XGBoost use dashboard quick settings; "
        "this selected holdout window can differ from the full project model-selection results. "
        "Upcoming forecasts remain fixed to Ridge."
    )


def rules_view(orders: pd.DataFrame, daily_orders: pd.DataFrame) -> None:
    st.header("Menu combinations by weather")
    st.caption(
        "Compare each directed item pair's confidence on rainy versus dry days. Since baskets are generated, "
        "this view evaluates planted generator patterns rather than observed customer behaviour."
    )
    min_support = st.slider(
        "Minimum support per weather segment (%)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        format="%.1f%%",
        help="Support is the share of rainy or dry baskets containing both items. Default 0.5% filters out rare copied-basket quirks.",
    )
    min_confidence = st.slider(
        "Minimum rule confidence (%)",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Confidence is the share of baskets containing the first item that also contain the second.",
    )
    support_fraction = min_support / 100
    rainy_dates = set(daily_orders.loc[daily_orders["is_rainy"].astype(bool), "date"])
    rainy_count = int(orders["date"].isin(rainy_dates).sum())
    dry_count = len(orders) - rainy_count
    rainy_min_pair_count = ceil(support_fraction * rainy_count)
    dry_min_pair_count = ceil(support_fraction * dry_count)
    st.caption(
        f"At this support, a pair must occur in about {rainy_min_pair_count:,} rainy or "
        f"{dry_min_pair_count:,} dry orders. Rules with 100% confidence or lift above 20 are "
        "excluded as likely copied-basket shortcuts; lift remains visible for context."
    )
    summaries = st.columns(2)
    summaries[0].metric("Rainy-day baskets", f"{rainy_count:,}")
    summaries[1].metric("Dry-day baskets", f"{dry_count:,}")
    try:
        comparisons = get_pair_comparisons(
            orders, daily_orders, support_fraction, min_confidence / 100
        )
    except (ValueError, KeyError) as error:
        st.error(f"Could not compare weather-segmented rules: {error}")
        return

    scope = st.radio(
        "Rule set",
        ("Planted rain-effect pairs", "All eligible pairs"),
        horizontal=True,
        help="The default focuses on the rain-related pairs intentionally planted in the generator.",
    )
    if scope == "Planted rain-effect pairs":
        comparisons = comparisons.loc[comparisons["planted_rain_pair"]]
    effect = st.selectbox(
        "Weather effect",
        ("All pairs", "Stronger on rainy days", "Stronger on dry days"),
        help="Filters the comparison table by the sign of rainy confidence minus dry confidence.",
    )
    comparisons = filter_weather_effect(comparisons, effect)
    if comparisons.empty:
        st.info("No paired rules meet these support, confidence and weather-effect settings.")
        return

    table = comparisons[
        [
            "antecedent",
            "consequent",
            "rainy_support",
            "dry_support",
            "rainy_confidence",
            "dry_confidence",
            "confidence_change_pp",
            "rainy_lift",
            "dry_lift",
            "lift_ratio",
            "rainy_pair_count",
            "dry_pair_count",
        ]
    ].rename(
        columns={
            "antecedent": "If basket has",
            "consequent": "Also contains",
            "rainy_support": "Rainy support",
            "dry_support": "Dry support",
            "rainy_confidence": "Rainy confidence",
            "dry_confidence": "Dry confidence",
            "confidence_change_pp": "Rainy − dry confidence (pp)",
            "rainy_lift": "Rainy lift",
            "dry_lift": "Dry lift",
            "lift_ratio": "Rainy / dry lift",
            "rainy_pair_count": "Rainy pair orders",
            "dry_pair_count": "Dry pair orders",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Rainy support": "{:.2%}",
                "Dry support": "{:.2%}",
                "Rainy confidence": "{:.1%}",
                "Dry confidence": "{:.1%}",
                "Rainy − dry confidence (pp)": "{:+.1f}",
                "Rainy lift": "{:.2f}",
                "Dry lift": "{:.2f}",
                "Rainy / dry lift": "{:.2f}×",
                "Rainy pair orders": "{:,.0f}",
                "Dry pair orders": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Sorted by rainy-minus-dry confidence in percentage points. Lift may move in the opposite "
        "direction because item prevalence also changes with weather."
    )

    st.subheader("14-day planted-pair trend")
    planted_pairs = [
        (row["antecedent"], row["consequent"])
        for _, row in comparisons.iterrows()
        if (row["antecedent"], row["consequent"]) in PLANTED_RAIN_PAIRS
    ]
    if not planted_pairs:
        st.info("Select planted rain-effect pairs to view a planted-pair timeline.")
        return
    selected_pair = st.selectbox(
        "Planted pair",
        planted_pairs,
        format_func=lambda pair: f"{pair[0]} → {pair[1]}",
    )
    trend_metric = st.radio(
        "Trend metric",
        ("Confidence", "Lift"),
        horizontal=True,
        help="Confidence directly shows how often the consequent appears when the antecedent is present.",
    )
    rainy_trend = get_pair_timeline(
        orders, daily_orders, *selected_pair, "Rainy"
    ).rename(columns={trend_metric.lower(): "Rainy"})
    dry_trend = get_pair_timeline(
        orders, daily_orders, *selected_pair, "Dry"
    ).rename(columns={trend_metric.lower(): "Dry"})
    trend = rainy_trend[["date", "Rainy"]].merge(
        dry_trend[["date", "Dry"]], on="date", how="outer"
    ).set_index("date")
    if trend.dropna(how="all").empty:
        st.info("Not enough antecedent orders in any 14-day window to show this pair.")
        return
    st.line_chart(
        trend,
        x_label="Window end date",
        y_label="Confidence" if trend_metric == "Confidence" else "Lift",
        color=["#c24170", "#274c77"],
    )
    st.caption(
        "Each point uses a rolling 14-day calendar window and is hidden unless the segment contains "
        "at least 20 antecedent orders. Empty stretches indicate insufficient evidence, not zero lift."
    )

def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f6f8fb; color: #202a36; }
        [data-testid="stHeader"] { background: rgba(246, 248, 251, 0.92); }
        [data-testid="stMainBlockContainer"] { padding-top: 1.8rem; }
        h1 { font-size: 1.8rem !important; line-height: 1.2 !important; }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e4e9f0; border-radius: 12px;
            padding: 1rem 1.1rem; box-shadow: 0 2px 8px rgba(31, 45, 61, 0.04);
        }
        [data-testid="stMetricLabel"] { color: #586777; }
        [data-testid="stMetricValue"] { color: #26384a; }
        [data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid #e2e8f0; }
        [data-baseweb="tab"] {
            height: 3rem; padding: 0 1rem; border-radius: 9px 9px 0 0;
            color: #526273; font-weight: 600;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff; color: #84265b; border-bottom: 2px solid #9a326c;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background: #ffffff; border: 1px solid #e4e9f0; border-radius: 10px;
        }
        div[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e7ebf1; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Coimbatore Food Orders")
    try:
        daily, orders, daily_orders = load_data()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load dashboard data: {error}")
        st.stop()

    overview_tab, dataset_tab, patterns_tab, forecast_tab, menu_tab = st.tabs(
        ["Overview", "Dataset", "Demand patterns", "Forecasts", "Menu insights"]
    )
    with overview_tab:
        overview_view(daily)
    with dataset_tab:
        dataset_view(daily, orders, daily_orders)
    with patterns_tab:
        demand_patterns_view(daily)
    with forecast_tab:
        forecast_view(daily)
    with menu_tab:
        rules_view(orders, daily_orders)


if __name__ == "__main__":
    main()
