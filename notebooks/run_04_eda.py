"""Execute notebooks/04_eda.ipynb as a standalone script (alternative to nbconvert on Windows).

This script reproduces every code cell in the notebook directly so that:
 1. All figures are generated and saved to results/eda/ and slides/review1/figures/.
 2. The script can be verified to run end-to-end without errors.

Run from the project root: python notebooks/run_04_eda.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import plotstyle
import eda

plotstyle.apply()
C = plotstyle.C

RESULTS = ROOT / "results/eda"
RESULTS.mkdir(parents=True, exist_ok=True)
SLIDES_FIGS = ROOT / "slides/review1/figures"
SLIDES_FIGS.mkdir(parents=True, exist_ok=True)

print(f"Environment initialized. Root: {ROOT}")

# ---------------------------------------------------------------------------
# 1. Data Ingestion & Quality Audit
# ---------------------------------------------------------------------------
data = eda.load_eda_data(ROOT)
daily = data["daily"]
orders = data["orders"]
weather = data["weather"]
weather_hourly = data["weather_hourly"]
menu = data["menu"]

print("=== DATASET HEALTH & INTEGRITY AUDIT ===")
print(f"Daily Orders:    {daily.shape[0]:,} rows, {daily.shape[1]} columns ({daily['date'].min().date()} to {daily['date'].max().date()})")
print(f"Transactions:    {orders.shape[0]:,} rows, {orders.shape[1]} columns")
print(f"Daily Weather:   {weather.shape[0]:,} rows, {weather.shape[1]} columns")
print(f"Hourly Weather:  {weather_hourly.shape[0]:,} rows, {weather_hourly.shape[1]} columns")
print(f"Menu Catalog:    {menu.shape[0]} items across {menu['category'].nunique()} categories")

assert daily["order_count"].isna().sum() == 0
assert orders["order_value"].isna().sum() == 0
assert daily["precip_mm"].isna().sum() == 0
print("Integrity check PASSED: 0 missing values in core analytical features.")

# ---------------------------------------------------------------------------
# 2. Order Volume Distribution Statistics
# ---------------------------------------------------------------------------
dist_stats = eda.get_order_distribution_stats(daily)

print("\n=== DAILY ORDER VOLUME SUMMARY STATISTICS ===")
for k, v in dist_stats.items():
    if k == "regime_stats":
        continue
    if isinstance(v, float):
        print(f"  {k:16s}: {v:,.2f}")
    else:
        print(f"  {k:16s}: {v}")

print("\n=== SUMMARY STATISTICS BY AMBIENT REGIME ===")
print(dist_stats["regime_stats"].round(2))

fig01 = eda.plot_fig01_order_distribution(daily, RESULTS / "eda_01_order_distribution.png")
eda.save_fig(fig01, SLIDES_FIGS / "eda_01_order_distribution.png")
plt.close(fig01)
print("[OK] Fig 01 - Order Distribution")

# ---------------------------------------------------------------------------
# 3. Basket & Monetary Value Statistics
# ---------------------------------------------------------------------------
basket_stats = eda.get_basket_and_value_stats(orders)

print("\n=== TRANSACTION & BASKET METRICS ===")
print(f"  Total Transactions : {basket_stats['total_orders']:,}")
print(f"  Avg Basket Size    : {basket_stats['mean_basket_size']:.2f} items")
print(f"  Avg Order Value    : Rs.{basket_stats['mean_order_value']:.2f} (Median: Rs.{basket_stats['median_order_value']:.2f})")
print(f"  IQR Order Value    : Rs.{basket_stats['p25_value']:.0f} to Rs.{basket_stats['p75_value']:.0f}")
print("\n=== METRICS BY MEAL SLOT ===")
print(basket_stats["meal_slot_stats"].round(2))

fig02 = eda.plot_fig02_basket_and_value(orders, RESULTS / "eda_02_basket_and_value.png")
eda.save_fig(fig02, SLIDES_FIGS / "eda_02_basket_and_value.png")
plt.close(fig02)
print("[OK] Fig 02 - Basket & Value")

# ---------------------------------------------------------------------------
# 4. Weekday-Wise Demand Analysis
# ---------------------------------------------------------------------------
dow_summary = eda.get_weekday_stats(daily)
print("\n=== WEEKDAY DEMAND SUMMARY ===")
print(dow_summary.round(2))

fig03 = eda.plot_fig03_weekday_patterns(daily, RESULTS / "eda_03_weekday_patterns.png")
eda.save_fig(fig03, SLIDES_FIGS / "eda_03_weekday_patterns.png")
plt.close(fig03)
print("[OK] Fig 03 - Weekday Patterns")

# ---------------------------------------------------------------------------
# 5. Hourly Order Patterns & Operational Meal Windows
# ---------------------------------------------------------------------------
hourly_counts = orders["hour"].value_counts().sort_index()
hourly_df = pd.DataFrame({"hour": hourly_counts.index, "orders": hourly_counts.values})
hourly_df["share_pct"] = (hourly_df["orders"] / len(orders) * 100).round(2)
print("\n=== HOURLY ORDER DISTRIBUTION ===")
print(hourly_df.T.to_string())

fig04 = eda.plot_fig04_hourly_meal_slots(orders, RESULTS / "eda_04_hourly_meal_slots.png")
eda.save_fig(fig04, SLIDES_FIGS / "eda_04_hourly_meal_slots.png")
plt.close(fig04)
print("[OK] Fig 04 - Hourly Meal Slots")

# ---------------------------------------------------------------------------
# 6. Rain vs. Dry Comparative Analysis
# ---------------------------------------------------------------------------
rain_stats = eda.get_rain_vs_dry_stats(daily)

print("\n=== STATISTICAL COMPARISON: RAINY VS DRY DAYS ===")
print(f"  Dry Days Mean  : {rain_stats['dry_mean']:.2f} orders/day (n=768)")
print(f"  Rainy Days Mean: {rain_stats['rainy_mean']:.2f} orders/day (n=329)")
print(f"  Abs Difference : +{rain_stats['absolute_diff']:.2f} orders/day")
print(f"  Relative Surge : +{rain_stats['lift_pct']:.2f}%")
print(f"  Welch's t-test : t={rain_stats['t_statistic']:.4f}, p={rain_stats['t_pvalue']:.4e}")
print(f"  Mann-Whitney U : U={rain_stats['mwu_statistic']:,.1f}, p={rain_stats['mwu_pvalue']:.4e}")
print("\n=== BREAKDOWN BY PRECIPITATION CATEGORY ===")
print(rain_stats["rain_category_summary"].round(2))

fig05 = eda.plot_fig05_rain_vs_dry_comparison(daily, RESULTS / "eda_05_rain_vs_dry_comparison.png")
eda.save_fig(fig05, SLIDES_FIGS / "eda_05_rain_vs_dry_comparison.png")
plt.close(fig05)
print("[OK] Fig 05 - Rain vs Dry Comparison")

fig06 = eda.plot_fig06_weekday_rain_interaction(daily, RESULTS / "eda_06_weekday_rain_interaction.png")
eda.save_fig(fig06, SLIDES_FIGS / "eda_06_weekday_rain_interaction.png")
plt.close(fig06)
print("[OK] Fig 06 - Weekday-Rain Interaction")

# ---------------------------------------------------------------------------
# 7. Menu Item & Category Weather Substitutions
# ---------------------------------------------------------------------------
item_shift, cat_regime_pct = eda.get_item_and_category_shifts(orders, daily, menu)

print("\n=== TOP 5 ITEMS SURGING ON RAINY DAYS ===")
top_rain = item_shift.sort_values("relative_lift_pct", ascending=False)[
    ["item_name", "category", "weather_affinity", "price", "dry_freq", "rain_freq", "relative_lift_pct"]
].head(5)
print(top_rain.round(3).to_string())

print("\n=== TOP 5 ITEMS PLUNGING ON RAINY DAYS ===")
top_drop = item_shift.sort_values("relative_lift_pct", ascending=True)[
    ["item_name", "category", "weather_affinity", "price", "dry_freq", "rain_freq", "relative_lift_pct"]
].head(5)
print(top_drop.round(3).to_string())

print("\n=== CATEGORY SHARE ACROSS REGIMES (%) ===")
print(cat_regime_pct.round(2).to_string())

fig07 = eda.plot_fig07_rainy_item_spikers(item_shift, RESULTS / "eda_07_item_rain_spikers.png")
eda.save_fig(fig07, SLIDES_FIGS / "eda_07_item_rain_spikers.png")
plt.close(fig07)
print("[OK] Fig 07 - Item Rain Spikers")

fig08 = eda.plot_fig08_category_regime_shifts(cat_regime_pct, RESULTS / "eda_08_category_regime_shifts.png")
eda.save_fig(fig08, SLIDES_FIGS / "eda_08_category_regime_shifts.png")
plt.close(fig08)
print("[OK] Fig 08 - Category Regime Shifts")

# ---------------------------------------------------------------------------
# 8. Correlation Analysis
# ---------------------------------------------------------------------------
pearson_corr, spearman_corr = eda.get_correlation_matrix(daily)

corr_summary = pd.DataFrame({
    "Pearson (r)": pearson_corr["order_count"],
    "Spearman (rho)": spearman_corr["order_count"]
}).sort_values("Pearson (r)", ascending=False)
print("\n=== CORRELATION WITH DAILY ORDER COUNT ===")
print(corr_summary.round(3).to_string())

fig09 = eda.plot_fig09_weather_orders_heatmap(daily, RESULTS / "eda_09_weather_orders_heatmap.png")
eda.save_fig(fig09, SLIDES_FIGS / "eda_09_weather_orders_heatmap.png")
plt.close(fig09)
print("[OK] Fig 09 - Weather-Orders Heatmap")

# ---------------------------------------------------------------------------
# 9. Longitudinal Timeline
# ---------------------------------------------------------------------------
fig10 = eda.plot_fig10_monsoon_timeline(daily, RESULTS / "eda_10_monsoon_timeline.png")
eda.save_fig(fig10, SLIDES_FIGS / "eda_10_monsoon_timeline.png")
plt.close(fig10)
print("[OK] Fig 10 - Monsoon Timeline")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("ALL 10 CHARTS SUCCESSFULLY GENERATED")
all_charts = sorted(RESULTS.glob("eda_*.png"))
print(f"Total exports in results/eda/: {len(all_charts)}")
for p in all_charts:
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name}  ({size_kb:.0f} KB)")

slide_charts = sorted(SLIDES_FIGS.glob("eda_*.png"))
print(f"\nSlide copies in slides/review1/figures/: {len(slide_charts)}")
print("="*60)
