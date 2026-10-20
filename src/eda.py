"""Exploratory Data Analysis (EDA) module for Coimbatore Food Orders (P4 / T19 & T20).

Provides data loaders, analytical aggregations, statistical hypothesis testing,
and presentation-ready figure generation conforming to the shared plot style.

Author: P4 (EDA, Presentation Chart Set & Review 1 Deck)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

try:
    from . import plotstyle
except ImportError:
    import plotstyle

plotstyle.apply()
C = plotstyle.C


def load_eda_data(root: Path | None = None) -> Dict[str, pd.DataFrame]:
    """Load and validate all raw datasets for exploratory analysis."""
    if root is None:
        root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data/raw"

    daily = pd.read_csv(raw_dir / "daily_orders.csv")
    orders = pd.read_csv(raw_dir / "orders.csv")
    weather = pd.read_csv(raw_dir / "weather.csv")
    weather_hourly = pd.read_csv(raw_dir / "weather_hourly.csv")
    menu = pd.read_csv(raw_dir / "menu.csv")

    # Parse dates and timestamps
    daily["date"] = pd.to_datetime(daily["date"])
    orders["date"] = pd.to_datetime(orders["date"])
    orders["timestamp"] = pd.to_datetime(orders["timestamp"])
    orders["hour"] = orders["timestamp"].dt.hour
    weather["date"] = pd.to_datetime(weather["date"])
    weather_hourly["datetime"] = pd.to_datetime(weather_hourly["datetime"])

    # Ensure daily financial aggregates exist
    if "total_revenue" not in daily.columns:
        daily_rev = orders.groupby("date").agg(
            total_revenue=("order_value", "sum"),
            mean_order_val=("order_value", "mean"),
            median_order_val=("order_value", "median"),
            total_items=("item_count", "sum")
        ).reset_index()
        daily_merged = daily.merge(daily_rev, on="date", how="left")
    else:
        daily_merged = daily.copy()

    # Add categorical rain bin
    def classify_rain(p: float) -> str:
        if p < 2.5:
            return "Dry (<2.5mm)"
        elif p < 15.0:
            return "Moderate Rain (2.5-15mm)"
        return "Heavy Rain (>=15mm)"

    daily_merged["rain_category"] = daily_merged["precip_mm"].apply(classify_rain)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_merged["dow_name"] = daily_merged["dow"].apply(lambda d: dow_names[d])

    return {
        "daily": daily_merged,
        "orders": orders,
        "weather": weather,
        "weather_hourly": weather_hourly,
        "menu": menu
    }


def get_order_distribution_stats(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for daily order volumes."""
    s = daily_df["order_count"]
    regimes = daily_df.groupby("regime")["order_count"].agg(
        count="count", mean="mean", std="std", median="median",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75)
    )
    regimes["iqr"] = regimes["p75"] - regimes["p25"]

    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": int(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": int(s.max()),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurt()),
        "regime_stats": regimes
    }


def get_basket_and_value_stats(orders_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basket size and monetary value metrics across transactions."""
    basket_counts = orders_df["item_count"].value_counts(normalize=True).sort_index() * 100
    val_series = orders_df["order_value"]

    slot_summary = orders_df.groupby("meal_slot").agg(
        orders=("order_id", "count"),
        share_pct=("order_id", lambda x: len(x) / len(orders_df) * 100),
        mean_val=("order_value", "mean"),
        median_val=("order_value", "median"),
        std_val=("order_value", "std"),
        mean_items=("item_count", "mean")
    )

    return {
        "total_orders": len(orders_df),
        "mean_basket_size": float(orders_df["item_count"].mean()),
        "basket_distribution_pct": basket_counts.to_dict(),
        "mean_order_value": float(val_series.mean()),
        "median_order_value": float(val_series.median()),
        "std_order_value": float(val_series.std()),
        "p25_value": float(val_series.quantile(0.25)),
        "p75_value": float(val_series.quantile(0.75)),
        "p95_value": float(val_series.quantile(0.95)),
        "max_value": int(val_series.max()),
        "meal_slot_stats": slot_summary
    }


def get_weekday_stats(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate order stats across days of the week."""
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    summary = daily_df.groupby("dow_name")["order_count"].agg(
        days="count", mean="mean", std="std", median="median", min="min", max="max"
    ).reindex(dow_order)

    summary["pct_vs_mon_thu"] = (
        summary["mean"] - summary.loc[["Mon", "Tue", "Wed", "Thu"], "mean"].mean()
    ) / summary.loc[["Mon", "Tue", "Wed", "Thu"], "mean"].mean() * 100
    return summary


def get_rain_vs_dry_stats(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comparative analysis and statistical testing between rainy and dry periods."""
    rainy = daily_df[daily_df["is_rainy"]]["order_count"]
    dry = daily_df[~daily_df["is_rainy"]]["order_count"]

    ttest = stats.ttest_ind(rainy, dry, equal_var=False)
    mwu = stats.mannwhitneyu(rainy, dry, alternative="two-sided")

    cat_stats = daily_df.groupby("rain_category")["order_count"].agg(
        days="count", mean="mean", std="std", median="median",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75)
    )
    cat_stats["lift_vs_dry_pct"] = (cat_stats["mean"] - cat_stats.loc["Dry (<2.5mm)", "mean"]) / cat_stats.loc["Dry (<2.5mm)", "mean"] * 100

    return {
        "rainy_mean": float(rainy.mean()),
        "dry_mean": float(dry.mean()),
        "absolute_diff": float(rainy.mean() - dry.mean()),
        "lift_pct": float((rainy.mean() - dry.mean()) / dry.mean() * 100),
        "t_statistic": float(ttest.statistic),
        "t_pvalue": float(ttest.pvalue),
        "mwu_statistic": float(mwu.statistic),
        "mwu_pvalue": float(mwu.pvalue),
        "rain_category_summary": cat_stats
    }


def get_item_and_category_shifts(
    orders_df: pd.DataFrame, daily_df: pd.DataFrame, menu_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze item popularity and category substitution across weather regimes."""
    # Merge order items with day regime
    merged = orders_df.merge(daily_df[["date", "is_rainy", "is_hot", "regime"]], on="date")

    # Unnest individual ordered items
    items_split = merged["items"].str.split("; ")
    expanded = pd.DataFrame({
        "order_id": np.repeat(merged["order_id"].values, items_split.str.len()),
        "is_rainy": np.repeat(merged["is_rainy"].values, items_split.str.len()),
        "regime": np.repeat(merged["regime"].values, items_split.str.len()),
        "item_name": [item for sublist in items_split for item in sublist]
    })
    expanded = expanded.merge(menu_df[["item_name", "category", "weather_affinity", "price"]], on="item_name", how="left")

    total_rain_orders = (merged["is_rainy"]).sum()
    total_dry_orders = (~merged["is_rainy"]).sum()

    item_rain_counts = expanded[expanded["is_rainy"]].groupby("item_name").size()
    item_dry_counts = expanded[~expanded["is_rainy"]].groupby("item_name").size()

    item_shift = pd.DataFrame({
        "rain_orders": item_rain_counts,
        "dry_orders": item_dry_counts,
        "rain_freq": item_rain_counts / total_rain_orders,
        "dry_freq": item_dry_counts / total_dry_orders
    }).fillna(0)
    item_shift["relative_lift_pct"] = (item_shift["rain_freq"] - item_shift["dry_freq"]) / item_shift["dry_freq"] * 100
    item_shift = item_shift.merge(menu_df[["item_name", "category", "weather_affinity", "price"]], left_index=True, right_on="item_name")

    # Category proportions across regimes
    cat_regime_counts = expanded.groupby(["regime", "category"]).size().unstack(level=0)
    cat_regime_pct = cat_regime_counts.div(cat_regime_counts.sum(axis=0), axis=1) * 100

    return item_shift, cat_regime_pct


def get_correlation_matrix(daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson and Spearman correlation matrices between weather and order metrics."""
    cols = [
        "order_count", "total_revenue", "mean_order_val",
        "precip_mm", "precip_hours", "temp_max_c",
        "temp_min_c", "temp_mean_c", "humidity_mean_pct",
        "wind_max_kmh", "daylight_s"
    ]
    pearson = daily_df[cols].corr(method="pearson")
    spearman = daily_df[cols].corr(method="spearman")
    return pearson, spearman


# -----------------------------------------------------------------------------
# PRESENTATION CHART GENERATION (T20)
# -----------------------------------------------------------------------------

def save_fig(fig: plt.Figure, filepath: Path) -> None:
    """Save figure with presentation-ready styling and clean padding."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=200, bbox_inches="tight", facecolor=C["bg"])


def plot_fig01_order_distribution(daily_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 01: Overall and regime-wise daily order volume distribution."""
    fig, ax = plt.subplots(figsize=(10, 5.2))

    # Distribution histogram
    palette = {"rainy": C["blue"], "hot": C["orange"], "normal": C["gray"]}
    present_regimes = [r for r in ["rainy", "hot", "normal"] if r in daily_df["regime"].unique()]
    sns.histplot(
        data=daily_df, x="order_count", hue="regime", multiple="stack",
        palette={r: palette[r] for r in present_regimes},
        bins=25, ax=ax, edgecolor=C["bg"], alpha=0.9
    )

    mean_val = daily_df["order_count"].mean()
    median_val = daily_df["order_count"].median()
    std_val = daily_df["order_count"].std()
    iqr_val = daily_df["order_count"].quantile(0.75) - daily_df["order_count"].quantile(0.25)
    skew_val = daily_df["order_count"].skew()
    rain_mean = daily_df.loc[daily_df["is_rainy"], "order_count"].mean() if daily_df["is_rainy"].any() else mean_val
    dry_mean = daily_df.loc[~daily_df["is_rainy"], "order_count"].mean() if (~daily_df["is_rainy"]).any() else mean_val
    diff_pct = (rain_mean - dry_mean) / dry_mean * 100 if dry_mean > 0 else 0
    hot_mean = daily_df.loc[daily_df["is_hot"], "order_count"].mean() if daily_df["is_hot"].any() else mean_val
    normal_mean = daily_df.loc[daily_df["regime"] == "normal", "order_count"].mean() if (daily_df["regime"] == "normal").any() else mean_val

    ax.axvline(mean_val, color=C["ink"], linestyle="--", linewidth=1.8, label=f"Overall Mean: {mean_val:.1f}")
    ax.axvline(median_val, color=C["aqua"], linestyle="-", linewidth=1.8, label=f"Overall Median: {median_val:.0f}")

    ax.set_title("Distribution of Daily Food Orders in Surabaya, Indonesia (2025)")
    ax.set_xlabel("Daily Order Count")
    ax.set_ylabel("Frequency (Number of Days)")

    # Dynamic annotation summary box
    stats_text = (
        f"Sample: {len(daily_df)} days\n"
        f"Mean: {mean_val:.1f} orders/day\n"
        f"Median: {median_val:.1f} orders/day\n"
        f"Std Dev: {std_val:.1f} | IQR: {iqr_val:.1f}\n"
        f"Skewness: {skew_val:+.2f}\n"
        f"Rainy Mean: {rain_mean:.1f} ({diff_pct:+.1f}%)\n"
        f"Hot Mean: {hot_mean:.1f} | Normal: {normal_mean:.1f}"
    )
    ax.text(
        0.73, 0.94, stats_text, transform=ax.transAxes,
        verticalalignment="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=C["light"], alpha=0.95),
        color=C["ink2"]
    )

    ax.legend(loc="upper center", bbox_to_anchor=(0.38, 0.98), ncol=2, frameon=False)
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig02_basket_and_value(orders_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 02: Basket size distribution and order value breakdown across meal slots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.4]})

    # Subplot 1: Basket size distribution
    basket_pct = orders_df["item_count"].value_counts(normalize=True).sort_index() * 100
    bars = ax1.bar(basket_pct.index, basket_pct.values, color=C["blue"], width=0.55, edgecolor=C["bg"])
    ax1.set_title("Customer Basket Size Distribution")
    ax1.set_xlabel("Number of Items in Order Basket")
    ax1.set_ylabel("Share of Total Transactions (%)")
    ax1.set_xticks(sorted(basket_pct.index))
    ax1.set_ylim(0, max(basket_pct.values) * 1.25)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, color=C["ink"])

    modal_size = basket_pct.idxmax()
    modal_pct = basket_pct.max()
    ax1.text(
        0.05, 0.92, f"Total Orders: {len(orders_df):,}\nMean Basket: {orders_df['item_count'].mean():.2f} items\nModal Basket: {modal_size} item ({modal_pct:.1f}%)",
        transform=ax1.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C["light"])
    )

    # Subplot 2: Order value distribution by meal slot
    slot_order = [s for s in ["Breakfast", "Lunch", "Snacks", "Dinner"] if s in orders_df["meal_slot"].unique()]
    palette = [C["gray"], C["blue"], C["orange"], C["aqua"]][:len(slot_order)]
    box = ax2.boxplot(
        [orders_df[orders_df["meal_slot"] == s]["order_value"] for s in slot_order],
        tick_labels=slot_order, patch_artist=True, showmeans=True,
        meanprops=dict(marker="o", markeredgecolor=C["ink"], markerfacecolor=C["ink"], markersize=5),
        medianprops=dict(color=C["ink"], linewidth=1.5),
        whiskerprops=dict(color=C["ink2"]),
        capprops=dict(color=C["ink2"]),
        flierprops=dict(marker=".", markersize=2, alpha=0.2, color=C["gray"])
    )

    for patch, col in zip(box["boxes"], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        patch.set_edgecolor(C["ink2"])

    ax2.set_title("Order Value Distribution by Meal Slot (IDR / Rp)")
    ax2.set_xlabel("Operational Meal Slot")
    ax2.set_ylabel("Order Total (Rp)")
    ax2.set_ylim(0, max(orders_df["order_value"]) * 1.15)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"Rp {int(y):,}"))

    # Value callouts
    means = [orders_df[orders_df["meal_slot"] == s]["order_value"].mean() for s in slot_order]
    for i, (m, s) in enumerate(zip(means, slot_order)):
        ax2.text(i + 1, m + 2000, f"Avg: Rp {m/1000:.1f}k", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C["ink"])

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig03_weekday_patterns(daily_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 03: Weekday demand profile showing weekend surges vs weekday baselines."""
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_stats = daily_df.groupby("dow_name")["order_count"].agg(["mean", "std", "median"]).reindex(dow_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dow_names))

    # Color weekend days prominently
    bar_colors = [C["gray"]] * 4 + [C["blue"]] * 3
    bars = ax.bar(x, dow_stats["mean"], yerr=dow_stats["std"] / 2, capsize=4, color=bar_colors, width=0.55, edgecolor=C["bg"])

    weekday_mean = dow_stats.loc[["Mon", "Tue", "Wed", "Thu"], "mean"].mean()
    weekend_mean = dow_stats.loc[["Fri", "Sat", "Sun"], "mean"].mean()
    weekend_diff = (weekend_mean - weekday_mean) / weekday_mean * 100
    peak_day = dow_stats["mean"].idxmax()
    peak_val = dow_stats["mean"].max()

    ax.axhline(weekday_mean, color=C["gray"], linestyle="--", linewidth=1.5, label=f"Mon–Thu Baseline: {weekday_mean:.1f} orders")
    ax.axhline(weekend_mean, color=C["blue"], linestyle="--", linewidth=1.5, label=f"Fri–Sun Mean: {weekend_mean:.1f} orders ({weekend_diff:+.1f}%)")

    ax.set_title("Day-of-Week Order Patterns: Surabaya MSME Demand Profile")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Daily Orders (with ±0.5σ error bar)")
    ax.set_xticks(x)
    ax.set_xticklabels(dow_names)
    ax.set_ylim(0, max(peak_val * 1.3, 270))

    for bar, (_, row) in zip(bars, dow_stats.iterrows()):
        h = row["mean"]
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{h:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=C["ink"])

    # Insight annotation
    ax.text(
        0.03, 0.88,
        f"Key Operational Finding:\n• Mon–Thu baseline: ~{weekday_mean:.1f} orders/day\n• Fri–Sun weekend mean: ~{weekend_mean:.1f} orders/day ({weekend_diff:+.1f}%)\n• Peak volume day: {peak_day} ({peak_val:.1f} orders/day)",
        transform=ax.transAxes, fontsize=9.5, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C["light"])
    )

    ax.legend(loc="upper right", frameon=False)
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig04_hourly_meal_slots(orders_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 04: Hourly order distribution mapped into operational meal windows."""
    hourly_counts = orders_df["hour"].value_counts().sort_index()
    hours = np.arange(24)
    counts = [hourly_counts.get(h, 0) for h in hours]

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Plot hourly volume
    bars = ax.bar(hours, counts, color=C["blue"], width=0.7, edgecolor=C["bg"])

    slot_stats = orders_df.groupby("meal_slot").agg(
        count=("order_id", "count"),
        avg_val=("order_value", "mean")
    )
    b_pct = slot_stats.loc["Breakfast", "count"] / len(orders_df) * 100 if "Breakfast" in slot_stats.index else 0
    b_val = slot_stats.loc["Breakfast", "avg_val"] if "Breakfast" in slot_stats.index else 0
    l_pct = slot_stats.loc["Lunch", "count"] / len(orders_df) * 100 if "Lunch" in slot_stats.index else 0
    l_val = slot_stats.loc["Lunch", "avg_val"] if "Lunch" in slot_stats.index else 0
    s_pct = slot_stats.loc["Snacks", "count"] / len(orders_df) * 100 if "Snacks" in slot_stats.index else 0
    s_val = slot_stats.loc["Snacks", "avg_val"] if "Snacks" in slot_stats.index else 0
    d_pct = slot_stats.loc["Dinner", "count"] / len(orders_df) * 100 if "Dinner" in slot_stats.index else 0
    d_val = slot_stats.loc["Dinner", "avg_val"] if "Dinner" in slot_stats.index else 0

    slots = [
        (f"Breakfast (7:00–11:00)\n{b_pct:.1f}% | Avg Rp {b_val/1000:.0f}k", 7, 11, C["gray"]),
        (f"Lunch (11:00–15:00)\n{l_pct:.1f}% | Avg Rp {l_val/1000:.0f}k", 11, 15, C["blue"]),
        (f"Snacks (15:00–18:00)\n{s_pct:.1f}% | Avg Rp {s_val/1000:.0f}k", 15, 18, C["orange"]),
        (f"Dinner (18:00–23:00)\n{d_pct:.1f}% | Avg Rp {d_val/1000:.0f}k", 18, 23, C["aqua"])
    ]

    for label, start, end, col in slots:
        ax.axvspan(start - 0.4, end - 0.4, alpha=0.15, color=col)
        mid = (start + end) / 2 - 0.4
        ax.text(mid, max(counts) * 1.05, label, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C["ink"])

    ax.set_title("Hourly Food Order Volumes Across Operational Meal Windows (Surabaya)")
    ax.set_xlabel("Hour of the Day (0:00 to 23:00 WIB)")
    ax.set_ylabel("Total Transactions (2025)")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" if h % 2 == 0 else "" for h in hours])
    ax.set_ylim(0, max(counts) * 1.25)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

    ax.text(
        0.03, 0.72,
        f"Operational Insights:\n"
        f"• Total Orders Analyzed: {len(orders_df):,}\n"
        f"• Peak meal window: Dinner ({d_pct:.1f}% of total transactions)\n"
        f"• Lunch window accounts for {l_pct:.1f}% of transactions\n"
        f"• Operating hours 07:00–23:00 WIB cover 100% of store foot traffic",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C["light"])
    )

    if save_path:
        save_fig(fig, save_path)
    return fig

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig05_rain_vs_dry_comparison(daily_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 05: Order volume distribution across precipitation intensity categories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})

    # Subplot 1: Boxplot by precipitation category
    cats = ["Dry (<2.5mm)", "Moderate Rain (2.5-15mm)", "Heavy Rain (>=15mm)"]
    palette = [C["gray"], C["blue"], C["orange"]]

    sns.boxplot(
        data=daily_df, x="rain_category", y="order_count", order=cats,
        hue="rain_category", palette=palette, ax=ax1, width=0.45,
        boxprops=dict(alpha=0.75), medianprops=dict(color=C["ink"], linewidth=2),
        legend=False
    )
    sns.stripplot(
        data=daily_df, x="rain_category", y="order_count", order=cats,
        hue="rain_category", palette=palette, ax=ax1, alpha=0.3, jitter=0.2, size=4,
        legend=False
    )

    ax1.set_title("Daily Orders by Rainfall Intensity")
    ax1.set_xlabel("Weather Condition Category")
    ax1.set_ylabel("Daily Order Count")
    ax1.set_xticks(range(len(cats)))
    ax1.set_xticklabels(["Dry\n(< 2.5 mm)", "Moderate Rain\n(2.5 – 15 mm)", "Heavy Rain\n(≥ 15 mm)"])

    # Add means
    cat_means = daily_df.groupby("rain_category")["order_count"].mean().reindex(cats)
    for i, m in enumerate(cat_means):
        ax1.text(i, m + 15, f"Mean: {m:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9, color=C["ink"])

    # Subplot 2: Cumulative density / KDE comparison
    n_dry = (~daily_df["is_rainy"]).sum()
    n_rain = daily_df["is_rainy"].sum()
    sns.kdeplot(data=daily_df[~daily_df["is_rainy"]]["order_count"], ax=ax2, color=C["gray"], fill=True, alpha=0.3, label=f"Dry Days (< 2.5mm, n={n_dry})")
    sns.kdeplot(data=daily_df[daily_df["is_rainy"]]["order_count"], ax=ax2, color=C["blue"], fill=True, alpha=0.35, label=f"Rainy Days (≥ 2.5mm, n={n_rain})")

    ax2.set_title("Density Shift: Rainy vs Dry Days")
    ax2.set_xlabel("Daily Order Count")
    ax2.set_ylabel("Kernel Density")

    rain_mean = daily_df[daily_df["is_rainy"]]["order_count"].mean() if n_rain > 0 else 0
    dry_mean = daily_df[~daily_df["is_rainy"]]["order_count"].mean() if n_dry > 0 else 0
    lift = (rain_mean - dry_mean) / dry_mean * 100 if dry_mean > 0 else 0
    rain_stats = get_rain_vs_dry_stats(daily_df)

    ax2.axvline(dry_mean, color=C["gray"], linestyle="--", linewidth=1.5)
    ax2.axvline(rain_mean, color=C["blue"], linestyle="--", linewidth=1.5)

    ax2.text(
        0.05, 0.88,
        f"Hypothesis Test Results:\n"
        f"• Dry Mean: {dry_mean:.1f} orders/day\n"
        f"• Rainy Mean: {rain_mean:.1f} orders/day\n"
        f"• Rain Effect: {lift:+.1f}% shift\n"
        f"• Welch's t-test: t = {rain_stats['t_statistic']:.2f}, p = {rain_stats['t_pvalue']:.3e}\n"
        f"• Mann-Whitney U: p = {rain_stats['mwu_pvalue']:.3e}",
        transform=ax2.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C["light"])
    )
    ax2.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig06_weekday_rain_interaction(daily_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 06: Interaction analysis of Day-of-Week and Weather Conditions."""
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pv = daily_df.assign(
        rain=np.where(daily_df["is_rainy"], "Rainy (≥ 2.5mm)", "Dry (< 2.5mm)")
    ).pivot_table(index="dow_name", columns="rain", values="order_count", aggfunc=["mean", "std"])
    pv = pv.reindex(dow_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dow_names))
    width = 0.35

    dry_means = pv[("mean", "Dry (< 2.5mm)")]
    rain_means = pv[("mean", "Rainy (≥ 2.5mm)")]

    rects1 = ax.bar(x - width/2, dry_means, width, label="Dry Days (< 2.5mm)", color=C["gray"], alpha=0.85)
    rects2 = ax.bar(x + width/2, rain_means, width, label="Rainy Days (≥ 2.5mm)", color=C["blue"])

    ax.set_title("Interaction Effect: Day-of-Week Demand Under Rainy vs Dry Weather")
    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Mean Daily Orders")
    ax.set_xticks(x)
    ax.set_xticklabels(dow_names)
    ax.set_ylim(0, max(dry_means.max(), rain_means.max()) * 1.35)

    # Percentage lift annotations above rainy bars
    for i in range(len(dow_names)):
        d_val = dry_means.iloc[i]
        r_val = rain_means.iloc[i]
        lift = (r_val - d_val) / d_val * 100 if d_val > 0 else 0
        ax.text(x[i] + width/2, r_val + 4, f"{lift:+.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C["blue"])

    peak_dow = rain_means.idxmax()
    peak_val = rain_means.max()
    ax.text(
        0.03, 0.92,
        f"Strategic Interaction Insight:\n"
        f"• Weather and weekday dynamics shape customer ordering volume\n"
        f"• Peak rainy volume day: {peak_dow} (~{peak_val:.1f} orders/day)\n"
        f"• Dynamic kitchen staffing can account for weekday-rain interaction",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C["light"])
    )

    ax.legend(loc="upper right", frameon=False)
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig07_rainy_item_spikers(item_shift_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 07: Top items surging during rain vs top items plunging."""
    spikers = item_shift_df.sort_values("relative_lift_pct", ascending=False).head(5)
    droppers = item_shift_df.sort_values("relative_lift_pct", ascending=True).head(5)
    combined = pd.concat([spikers, droppers]).sort_values("relative_lift_pct")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y_pos = np.arange(len(combined))
    colors = [C["orange"] if v < 0 else C["blue"] for v in combined["relative_lift_pct"]]

    bars = ax.barh(y_pos, combined["relative_lift_pct"], color=colors, height=0.65, edgecolor=C["bg"])
    ax.axvline(0, color=C["ink"], linewidth=1.0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{name} ({aff})" for name, aff in zip(combined["item_name"], combined["weather_affinity"])])
    ax.set_title("Weather Substitution Effect: Menu Item Demand Shift on Rainy vs Dry Days")
    ax.set_xlabel("Relative Percentage Change in Order Probability on Rainy Days (%)")

    for bar, val in zip(bars, combined["relative_lift_pct"]):
        offset = 3 if val >= 0 else -4
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}%", ha=ha, va="center", fontsize=9, fontweight="bold", color=C["ink"])

    plunging_lines = "\n".join([f"• {r['item_name']}: {r['relative_lift_pct']:+.1f}%" for _, r in droppers.head(3).iterrows()])
    surging_lines = "\n".join([f"• {r['item_name']}: {r['relative_lift_pct']:+.1f}%" for _, r in spikers.head(3).iterrows()])

    ax.text(
        0.03, 0.22,
        f"Plunging Items:\n{plunging_lines}",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=C["orange"])
    )

    ax.text(
        0.65, 0.82,
        f"Surging Items:\n{surging_lines}",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=C["blue"])
    )

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig08_category_regime_shifts(cat_regime_pct: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 08: Share of menu categories across Hot, Normal, and Rainy weather regimes."""
    fig, ax = plt.subplots(figsize=(10, 5.2))

    regime_order = [r for r in ["hot", "normal", "rainy"] if r in cat_regime_pct.columns]
    regime_labels_dict = {
        "hot": "Hot Regime\n(T_max ≥ 33°C)",
        "normal": "Normal Regime\n(Dry & Moderate)",
        "rainy": "Rainy Regime\n(Precip ≥ 2.5mm)"
    }
    regime_labels = [regime_labels_dict[r] for r in regime_order]
    cat_df = cat_regime_pct[regime_order].T

    cat_colors = [C["blue"], C["gray"], C["orange"], "#d9822b", C["aqua"], "#8e44ad", "#16a085"]

    bottom = np.zeros(len(regime_order))
    for i, col in enumerate(cat_df.columns):
        vals = cat_df[col].values
        ax.bar(regime_labels, vals, bottom=bottom, label=col, color=cat_colors[i % len(cat_colors)], width=0.55, edgecolor=C["bg"])
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 10:
                ax.text(j, b + v/2, f"{v:.1f}%", ha="center", va="center", color="white" if i in [0, 4, 5] else C["ink"], fontsize=8.5, fontweight="bold")
        bottom += vals

    ax.set_title("Menu Category Share Shift Across Ambient Weather Regimes (Surabaya)")
    ax.set_ylabel("Category Proportion of Ordered Items (%)")
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig09_weather_orders_heatmap(daily_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 09: Comprehensive Weather-Orders Correlation Heatmap."""
    cols = [
        "order_count", "total_revenue", "mean_order_val",
        "precip_mm", "precip_hours", "humidity_mean_pct",
        "temp_max_c", "temp_mean_c", "temp_min_c",
        "wind_max_kmh", "daylight_s"
    ]
    labels = [
        "Order Count", "Total Revenue", "Avg Order Value",
        "Precipitation (mm)", "Precip Hours", "Relative Humidity (%)",
        "Max Temp (°C)", "Mean Temp (°C)", "Min Temp (°C)",
        "Max Wind Speed", "Daylight Duration"
    ]

    corr = daily_df[cols].corr(method="pearson")
    corr.columns = labels
    corr.index = labels

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-0.6, vmax=1.0, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8.5},
        square=True, linewidths=0.6, linecolor="white", cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax
    )

    ax.set_title("Weather & Order Variables Correlation Heatmap (Surabaya MSME Analysis)", pad=16)

    r_rain = corr.loc["Precipitation (mm)", "Order Count"]
    r_hours = corr.loc["Precip Hours", "Order Count"]
    r_rev = corr.loc["Total Revenue", "Order Count"]
    r_val = corr.loc["Avg Order Value", "Order Count"]
    r_wind = corr.loc["Max Wind Speed", "Order Count"]

    note = (
        "Key Correlations:\n"
        f"• Rain: r = {r_rain:+.2f} with Order Count\n"
        f"• Precip Hours: r = {r_hours:+.2f}\n"
        f"• Revenue: r = {r_rev:+.2f} with Orders\n"
        f"• Avg Order Value: r = {r_val:+.2f}\n"
        f"• Wind: r = {r_wind:+.2f}"
    )
    ax.text(
        0.02, 0.35, note, transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=C["light"])
    )

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_fig10_monsoon_timeline(daily_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Fig 10: Multi-panel time series showing orders, monsoon rainfall, and heat episodes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})

    # Panel 1: Daily Orders
    ax1.plot(daily_df["date"], daily_df["order_count"], color=C["gray"], lw=0.9, alpha=0.8, label="Daily Orders")
    ax1.plot(daily_df["date"], daily_df["order_count"].rolling(14, center=True).mean(), color=C["orange"], lw=2.2, label="14-day Rolling Mean")

    # Highlight rainy days
    rain_pts = daily_df[daily_df["is_rainy"]]
    ax1.scatter(rain_pts["date"], rain_pts["order_count"], color=C["blue"], s=14, zorder=3, alpha=0.75, label="Rainy Day (≥ 2.5mm)")

    # Festival markers
    fest_pts = daily_df[daily_df["is_festival"]]
    ax1.scatter(fest_pts["date"], fest_pts["order_count"], color=C["aqua"], s=28, marker="^", zorder=4, label="Indonesian Holiday")

    ax1.set_title("Surabaya MSME Longitudinal Profile: Daily Orders vs Meteorological Dynamics (2025)")
    ax1.set_ylabel("Daily Orders")
    ax1.legend(loc="upper left", ncol=4, frameon=False, fontsize=9)
    y_min = max(0, daily_df["order_count"].min() * 0.8)
    y_max = daily_df["order_count"].max() * 1.25
    ax1.set_ylim(y_min, y_max)

    # Panel 2: Daily Rainfall and Max Temperature
    ax2.bar(daily_df["date"], daily_df["precip_mm"], color=C["blue"], width=1.0, alpha=0.85, label="Rainfall (mm)")
    ax2.set_ylabel("Rainfall (mm)", color=C["blue"])
    ax2.set_ylim(0, max(50, daily_df["precip_mm"].max() * 1.15))
    ax2.tick_params(axis="y", labelcolor=C["blue"])

    ax2_temp = ax2.twinx()
    ax2_temp.plot(daily_df["date"], daily_df["temp_max_c"], color=C["orange"], lw=1.2, alpha=0.7, label="Max Temp (°C)")
    ax2_temp.axhline(33.0, color=C["orange"], linestyle=":", lw=1.0, alpha=0.8)
    ax2_temp.set_ylabel("Max Temp (°C)", color=C["orange"])
    ax2_temp.set_ylim(20, max(42, daily_df["temp_max_c"].max() + 3))
    ax2_temp.tick_params(axis="y", labelcolor=C["orange"])
    ax2_temp.grid(False)

    ax2.set_xlabel("Date (January 2025 to September 2025)")

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def generate_all_presentation_charts(output_dir: Path | None = None) -> list[Path]:
    """Generate and save all 10 presentation-ready charts for T20."""
    root = Path(__file__).resolve().parents[1]
    if output_dir is None:
        output_dir = root / "results/eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_eda_data(root)
    daily = data["daily"]
    orders = data["orders"]
    menu = data["menu"]

    item_shift, cat_regime_pct = get_item_and_category_shifts(orders, daily, menu)

    charts = [
        ("eda_01_order_distribution.png", lambda p: plot_fig01_order_distribution(daily, p)),
        ("eda_02_basket_and_value.png", lambda p: plot_fig02_basket_and_value(orders, p)),
        ("eda_03_weekday_patterns.png", lambda p: plot_fig03_weekday_patterns(daily, p)),
        ("eda_04_hourly_meal_slots.png", lambda p: plot_fig04_hourly_meal_slots(orders, p)),
        ("eda_05_rain_vs_dry_comparison.png", lambda p: plot_fig05_rain_vs_dry_comparison(daily, p)),
        ("eda_06_weekday_rain_interaction.png", lambda p: plot_fig06_weekday_rain_interaction(daily, p)),
        ("eda_07_item_rain_spikers.png", lambda p: plot_fig07_rainy_item_spikers(item_shift, p)),
        ("eda_08_category_regime_shifts.png", lambda p: plot_fig08_category_regime_shifts(cat_regime_pct, p)),
        ("eda_09_weather_orders_heatmap.png", lambda p: plot_fig09_weather_orders_heatmap(daily, p)),
        ("eda_10_monsoon_timeline.png", lambda p: plot_fig10_monsoon_timeline(daily, p)),
    ]

    saved_paths = []
    for filename, plot_fn in charts:
        p = output_dir / filename
        plot_fn(p)
        saved_paths.append(p)
        # Also copy/save directly to slides/review1/figures/
        slide_fig_path = root / "slides/review1/figures" / filename
        plot_fn(slide_fig_path)

    return saved_paths


if __name__ == "__main__":
    paths = generate_all_presentation_charts()
    print(f"Successfully generated {len(paths)} presentation charts in results/eda and slides/review1/figures:")
    for p in paths:
        print(f" - {p.name}")
