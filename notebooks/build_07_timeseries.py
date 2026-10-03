"""Generates notebooks/07_timeseries.ipynb (run: python notebooks/build_07_timeseries.py).

Kept in the repo so the notebook source is reviewable as plain text in pull requests.
Execute afterwards with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/07_timeseries.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# 07 · Time-series analysis and order-volume forecasting
**Owner:** P1 (Weather Data & Forecasting) · **Tasks:** T04 data preparation, T05 forecasting models, T06 evaluation ·
**Rubric:** Review 2, Method 2 (time series): preparation and implementation (2), evaluation and interpretation (3)

**Question.** Can next week's daily order volume be forecast, and does knowing the weather improve the forecast?

**Data.** Real daily weather for Coimbatore from Open-Meteo (`data/raw/weather.csv`, see `docs/dataset.md`) joined to the
*synthetic* daily order counts from the generator. Order counts are synthetic by design, so the results below describe how well
each method recovers the documented generator rules; they are not evidence about a real restaurant.

**Plan.** 1) prepare and test the series, 2) fit baselines and three model families, 3) compare them with a rolling-origin
backtest, 4) check residuals, segment errors (rainy vs dry, Fri–Sun vs Mon–Thu) and translate the error into a planning buffer.
""")

code("""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import forecast as F
import plotstyle as S
S.apply()

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
def save(fig, name):
    fig.savefig(RES / f"{name}.png", dpi=160, bbox_inches="tight")

# ---- configuration -------------------------------------------------------------------------
WEATHER_CSV = ROOT / "data/raw/weather.csv"
ORDERS_CSV  = ROOT / "data/raw/daily_orders.csv"   # P2's final orders file
INITIAL, HORIZON, STEP = 180, 7, 7    # backtest: first 180 days to train, then 7-day forecasts every 7 days
TEST_DAYS = 56                        # static hold-out: last 8 weeks
""")

md("""
## T04 · Data preparation
### 1. Load and check the data
""")
code("""
df = F.load_daily(WEATHER_CSV, ORDERS_CSV)
assert df["order_count"].notna().all(), "orders missing for some weather days"
display(F.data_quality(df).to_frame("value"))
display(df[["order_count", "precip_mm", "temp_max_c", "is_rainy", "heavy_rain", "is_hot", "dow", "regime"]].head())
""")

md("""
### 2. The series and how weather relates to it
Blue dots mark rainy days (≥ 2.5 mm). The 7-day mean shows there is no visible trend; the day-to-day movement is weekday
pattern, weather and noise.
""")
code("""
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(df.index, df["order_count"], color=S.C["gray"], lw=1.1, label="Daily orders")
r = df[df["is_rainy"] == 1]
ax.scatter(r.index, r["order_count"], s=16, color=S.C["blue"], zorder=3, label="Rainy day (≥ 2.5 mm)")
ax.plot(df.index, df["order_count"].rolling(7, center=True).mean(), color=S.C["orange"], lw=2, label="7-day mean")
ax.set_ylabel("Orders per day"); ax.set_title("Daily orders, Sep 2023 – Sep 2026")
ax.legend(ncol=3, loc="upper left")
save(fig, "fig01_series"); plt.show()
""")
code("""
g = df.groupby("regime")["order_count"].agg(days="count", mean="mean", median="median", std="std").round(1)
display(g)

pv = df.assign(rain=np.where(df["is_rainy"] == 1, "rainy", "dry")).pivot_table(index="dow", columns="rain", values="order_count", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(7); w = 0.38
ax.bar(x - w/2, pv["rainy"], w, color=S.C["blue"], label="Rainy days")
ax.bar(x + w/2, pv["dry"], w, color=S.C["gray"], label="Dry days")
ax.set_xticks(x, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); ax.set_ylabel("Mean orders per day")
ax.set_title("Weekday pattern, split by rain"); ax.legend(ncol=2, loc="upper left")
save(fig, "fig02_weekday_rain"); plt.show()
""")

md("""
### 3. Stationarity
ADF tests the null of a unit root; KPSS tests the null of stationarity. Reading them together: ADF rejecting *and* KPSS not
rejecting is the clean "stationary" result. A stationary series needs no differencing, so the models below use `d = 0`.
""")
code("""
st = F.stationarity_tests(df["order_count"])
display(st.round(4))
""")

md("""
### 4. Seasonal decomposition
Additive decomposition with a weekly period (7). Three years are available, so a yearly component could be estimated, but the generator has no yearly term, so none is
included; the weekly period and the festival flag carry the calendar structure.
""")
code("""
dec = F.decompose(df["order_count"], period=7)
fs = F.weekly_strength(dec)
rem_vs_rain = np.corrcoef(dec.resid.dropna(), df.loc[dec.resid.dropna().index, "precip_mm"])[0, 1]
print(f"Weekly seasonal strength: {fs:.2f} (0 = none, 1 = pure seasonality)")
print(f"Correlation of the remainder with same-day rainfall: {rem_vs_rain:.2f}")

fig, axes = plt.subplots(3, 1, figsize=(11, 6.4), sharex=False)
axes[0].plot(dec.trend.index, dec.trend, color=S.C["blue"]); axes[0].set_title("Trend (weekly window)")
sea = dec.seasonal.groupby(dec.seasonal.index.dayofweek).mean()
axes[1].bar(range(7), sea.values, color=S.C["blue"], width=0.6)
axes[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
axes[1].set_title("Weekly seasonal component (effect by weekday)")
axes[2].plot(dec.resid.index, dec.resid, color=S.C["gray"], lw=1); axes[2].set_title("Remainder")
for a in axes: a.set_ylabel("orders")
fig.tight_layout(); save(fig, "fig03_decomposition"); plt.show()
""")

md("""
### 5. Time-ordered train / test split
No shuffling: the last 8 weeks are held out. Model hyper-parameters are chosen using only the first `INITIAL` days.
""")
code("""
train, test = F.time_split(df, TEST_DAYS)
print(f"train: {train.index.min().date()} → {train.index.max().date()} ({len(train)} days)")
print(f"test : {test.index.min().date()} → {test.index.max().date()} ({len(test)} days)")
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(train.index, train["order_count"], color=S.C["gray"], lw=1.1, label="Train")
ax.plot(test.index, test["order_count"], color=S.C["blue"], lw=1.6, label="Test (held out)")
ax.axvline(test.index[0], color=S.C["ink2"], lw=1, ls=":")
ax.set_ylabel("Orders per day"); ax.set_title("Chronological train / test split"); ax.legend(ncol=2, loc="upper left")
save(fig, "fig04_split"); plt.show()
""")

md("""
## T05 · Forecasting models
| Model | What it does | Uses weather? |
|---|---|---|
| Seasonal naive | repeats the last observed week | no |
| Weekday mean | average of each weekday over the training window (stronger baseline) | no |
| SARIMAX (no weather) | day-of-week dummies + festival flag + ARMA errors | no |
| SARIMAX (weather) | the same, plus rain, heavy-rain and hot-day flags | yes |
| SARIMAX log (weather) | as above on log orders (effects of rain and heat are proportional) | yes |
| Prophet (weather) | weekly seasonality + festival flag + the same three weather regressors; no yearly term | yes |

**Weather is oracle weather.** For each forecast day the model is given that day's *observed* weather. A live system would have
to use a weather forecast, so the accuracy below is an upper bound for the weather models.
""")
code("""
init = df.iloc[:INITIAL]
spec_w, tab_w = F.select_sarimax_order(init, weather=True)
spec_l, tab_l = F.select_sarimax_order(init, weather=True, log=True)
spec_n, tab_n = F.select_sarimax_order(init, weather=False)
for name, tab in [("SARIMAX (weather)", tab_w), ("SARIMAX log (weather)", tab_l), ("SARIMAX (no weather)", tab_n)]:
    print(name, "- three lowest AIC candidates (chosen on the first", INITIAL, "days only)")
    display(tab.head(3).round(1))
""")
code("""
models = {
    "Seasonal naive":        F.seasonal_naive,
    "Weekday mean":          F.weekday_mean,
    "SARIMAX (no weather)":  F.make_sarimax(spec_n),
    "SARIMAX (weather)":     F.make_sarimax(spec_w),
    "SARIMAX log (weather)": F.make_sarimax(spec_l),
    "Prophet (weather)":     F.make_prophet(weather=True),
}
fc = F.rolling_origin(df, models, initial=INITIAL, horizon=HORIZON, step=STEP)
fc.to_csv(RES / "backtest_forecasts.csv", index=False)
print(f"{fc['date'].nunique()} forecast days x {len(models)} models; first forecast day {fc['date'].min():%Y-%m-%d}")
""")

md("""
### Forecast versus actual, last 8 weeks of the backtest
""")
code("""
last = fc["date"].sort_values().unique()[-TEST_DAYS:]
fig, ax = plt.subplots(figsize=(11, 3.9))
act = fc[fc["model"] == "SARIMAX (weather)"].set_index("date").loc[last]
ax.plot(act.index, act["y"], color=S.C["ink"], lw=2.2, label="Actual")
for name, col, ls in [("SARIMAX (weather)", S.C["blue"], "-"), ("Prophet (weather)", S.C["orange"], "-"), ("Weekday mean", S.C["gray"], "--")]:
    p = fc[fc["model"] == name].set_index("date").loc[last]
    ax.plot(p.index, p["yhat"], color=col, lw=1.6, ls=ls, label=name)
ax.set_ylabel("Orders per day"); ax.set_title("7-day-ahead forecasts vs actual")
ax.legend(ncol=4, loc="upper left")
save(fig, "fig05_forecast_vs_actual"); plt.show()
""")

md("""
## T06 · Evaluation and interpretation
### 1. Accuracy over the whole backtest
MAE and RMSE are in orders per day; MAPE is relative to actual orders. `cover80_%` is the share of days that fell inside the
model's own 80 % prediction interval (should be near 80).
""")
code("""
M = F.metrics(fc).sort_values("MAE")
base = M.loc["Weekday mean", "MAE"]
M["MAE vs weekday mean, %"] = (M["MAE"] / base - 1) * 100
display(M.round(2))
M.round(3).to_csv(RES / "metrics_overall.csv")

fig, ax = plt.subplots(figsize=(8, 3.4))
order = M.index[::-1]
cols = [S.C["gray"] if m in ("Seasonal naive", "Weekday mean") else S.C["blue"] for m in order]
bars = ax.barh(order, M.loc[order, "MAE"], color=cols, height=0.62)
for b, v in zip(bars, M.loc[order, "MAE"]):
    ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", color=S.C["ink"], fontsize=9)
ax.set_xlabel("Mean absolute error (orders per day)"); ax.set_title("Backtest error by model  (gray = baselines)")
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
save(fig, "fig06_mae_by_model"); plt.show()
""")

md("""
### 2. Are the differences real?
Paired bootstrap over forecast days: the interval is for `MAE(a) − MAE(b)`. Negative means `a` is better; an interval that
includes 0 means the data cannot separate the two.
""")
code("""
best_model = M.index[0]
pairs = [
    (best_model, "Weekday mean"),
    (best_model, "Seasonal naive"),
    ("SARIMAX (weather)", "SARIMAX (no weather)"),
    ("Prophet (weather)", "SARIMAX (weather)"),
    ("SARIMAX log (weather)", "SARIMAX (weather)"),
]
seen, rows = set(), []
for a, b in pairs:
    if a == b or (a, b) in seen: continue
    seen.add((a, b)); rows.append(F.paired_bootstrap_mae_diff(fc, a, b))
B = pd.DataFrame(rows)
B["ci95_low"] = B["ci95"].map(lambda t: t[0]); B["ci95_high"] = B["ci95"].map(lambda t: t[1])
B["separable"] = np.where((B["ci95_low"] > 0) | (B["ci95_high"] < 0), "yes", "no")
B = B.drop(columns="ci95")
display(B.round(2)); B.round(3).to_csv(RES / "bootstrap_comparisons.csv", index=False)
""")

md("""
### 3. Where does the error come from? Rainy vs dry, Fri–Sun vs Mon–Thu
""")
code("""
fcx = F.add_context(fc, df)
seg_rain = F.metrics(fcx, by=["rain"])
seg_day = F.metrics(fcx, by=["day_type"])
display(seg_rain[["n", "MAE", "MAPE_%", "bias"]].round(2))
display(seg_day[["n", "MAE", "MAPE_%", "bias"]].round(2))
pd.concat({"rain": seg_rain, "day_type": seg_day}).round(3).to_csv(RES / "metrics_by_segment.csv")

pr = seg_rain["MAE"].unstack("rain").loc[M.index]
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(pr)); w = 0.38
ax.bar(x - w/2, pr["rainy"], w, color=S.C["blue"], label="Rainy days")
ax.bar(x + w/2, pr["dry"], w, color=S.C["gray"], label="Dry days")
ax.set_xticks(x, [m.replace(" (", "\\n(") for m in pr.index], fontsize=8)
ax.set_ylabel("MAE (orders per day)"); ax.set_title("Forecast error on rainy vs dry days"); ax.legend(ncol=2, loc="upper left")
save(fig, "fig07_error_rain_dry"); plt.show()
""")

md("""
### 4. Residual checks for the best model
If the model has captured the systematic structure, the out-of-sample errors should show no autocorrelation
(Ljung–Box p > 0.05), be centred near zero, and not depend on the rain regime.
""")
code("""
from statsmodels.graphics.tsaplots import plot_acf
diag = F.residual_diagnostics(fc, best_model, lags=(7, 14))
print(f"Best model: {best_model}")
print("Ljung-Box p-values:", {k: round(v, 3) for k, v in diag["ljung_box_p"].items()})
print(f"Residual mean {diag['resid_mean']:.1f}, std {diag['resid_std']:.1f}")

r = fcx[fcx["model"] == best_model].sort_values("date")
res = (r["y"] - r["yhat"]).to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
plot_acf(res, lags=21, ax=axes[0], color=S.C["blue"], vlines_kwargs={"colors": S.C["blue"]}, alpha=0.05)
axes[0].set_title("Autocorrelation of forecast errors"); axes[0].set_xlabel("lag (days)")
for name, col in [("dry", S.C["gray"]), ("rainy", S.C["blue"])]:
    s = r["rain"] == name
    axes[1].scatter(r.loc[s, "yhat"], res[s.to_numpy()], s=14, color=col, label=name.capitalize() + " days", alpha=0.85)
axes[1].axhline(0, color=S.C["ink2"], lw=1)
axes[1].set_xlabel("Forecast (orders)"); axes[1].set_ylabel("Error (actual − forecast)")
axes[1].set_title("Errors vs forecast level"); axes[1].legend(ncol=2)
fig.tight_layout(); save(fig, "fig08_residuals"); plt.show()

display(r.assign(err=res).groupby("regime")["err"].agg(days="count", mean="mean", std="std").round(1))
""")

md("""
### 5. Longer horizon: fit once, forecast the 8-week hold-out
""")
code("""
hold = F.static_holdout(df, models, test_days=TEST_DAYS)
H = F.metrics(hold).sort_values("MAE")
display(H.round(2)); H.round(3).to_csv(RES / "metrics_static_holdout.csv")
""")

md("""
### 6. How close to the best possible?
Because the orders are synthetic, the noise floor is known: no model can beat the generator's own expected value. The cell
below applies the multipliers documented by P2 (`docs/generator.md`) to the backtest days. **These constants are copied from the
final generator (including the festival uplift on the 41 festival days); update them if P2 changes the generator.**
""")
code("""
BASE, WEEKEND, RAIN_MID, RAIN_HEAVY, HEAT, FESTIVAL = 140, 1.30, 1.20, 1.35, 1.10, 1.25
d = df.loc[fc["date"].unique()].copy()
mult = np.where(d["dow"].isin([4, 5, 6]), WEEKEND, 1.0) \\
     * np.select([d["precip_mm"] >= 15, d["precip_mm"] >= 2.5], [RAIN_HEAVY, RAIN_MID], 1.0) \\
     * np.where(d["temp_max_c"] >= 33, HEAT, 1.0) \\
     * np.where(d["is_festival"] == 1, FESTIVAL, 1.0)
floor = pd.DataFrame({"date": d.index, "y": d["order_count"].to_numpy(), "yhat": BASE * mult, "model": "Generator expected value",
                      "lo80": np.nan, "hi80": np.nan})
floor_m = F.metrics(floor)
display(floor_m.round(2))
FLOOR_MAE = float(floor_m["MAE"].iloc[0])
""")

md("""
### 7. What the forecast means for operations
The planning buffer is how far above the forecast to prepare so that 90 % of days are covered.
""")
code("""
buf_all = F.planning_buffer(fc, best_model, 0.90)
buf_rain = F.planning_buffer(fcx[fcx["rain"] == "rainy"], best_model, 0.90)
buf_dry = F.planning_buffer(fcx[fcx["rain"] == "dry"], best_model, 0.90)
buf = pd.DataFrame({"all days": buf_all, "rainy days": buf_rain, "dry days": buf_dry}).T.drop(columns="q").round(1)
display(buf); buf.to_csv(RES / "planning_buffer.csv")
""")

md("""
## Conclusions
The text below is generated from the results above, so it stays correct if the data or the generator changes.
""")
code("""
def mae(m): return float(M.loc[m, "MAE"])
w_gain = mae("SARIMAX (no weather)") - mae("SARIMAX (weather)")
b1 = B[(B["a"] == best_model) & (B["b"] == "Weekday mean")]
b2 = B[(B["a"] == "SARIMAX (weather)") & (B["b"] == "SARIMAX (no weather)")]
rain_mae, dry_mae = seg_rain.loc[(best_model, "rainy"), "MAE"], seg_rain.loc[(best_model, "dry"), "MAE"]
rr = r.assign(err=res)
rain_sd, other_sd = rr.loc[rr["rain"] == "rainy", "err"].std(), rr.loc[rr["rain"] == "dry", "err"].std()
weather_models = ["SARIMAX (weather)", "SARIMAX log (weather)", "Prophet (weather)"]
wm_sep = B[B["a"].isin(weather_models) & B["b"].isin(weather_models)]
cov = M.loc[best_model].get("cover80_%", float("nan"))
hold_best = H.index[0]
lines = [
    f"1. Best model on the rolling backtest: {best_model}, MAE {mae(best_model):.1f} orders/day "
    f"({M.loc[best_model, 'MAPE_%']:.1f}% MAPE), against {mae('Seasonal naive'):.1f} for the seasonal-naive baseline and "
    f"{mae('Weekday mean'):.1f} for the weekday-mean baseline.",
    f"2. Knowing the weather lowers SARIMAX error by {w_gain:.1f} orders/day ({w_gain / mae('SARIMAX (no weather)') * 100:.0f}%); "
    f"paired-bootstrap 95% interval for the MAE difference {b2['ci95_low'].iloc[0]:.1f} to {b2['ci95_high'].iloc[0]:.1f} "
    f"({'distinguishable from zero' if b2['separable'].iloc[0] == 'yes' else 'not distinguishable from zero with this sample'}).",
    f"3. Versus the weekday-mean baseline the best model's MAE difference is {b1['mae_diff'].iloc[0]:.1f} orders/day "
    f"(95% interval {b1['ci95_low'].iloc[0]:.1f} to {b1['ci95_high'].iloc[0]:.1f}).",
    f"4. Error on rainy days is {rain_mae:.1f} orders/day against {dry_mae:.1f} on dry days; the spread of the best model's errors is "
    f"{rain_sd:.0f} on rainy days versus {other_sd:.0f} on dry days, so rainy days are the hardest to plan for.",
    f"5. The generator's own expected value would score MAE {FLOOR_MAE:.1f}. The best model is {mae(best_model) - FLOOR_MAE:.1f} orders/day "
    f"({(mae(best_model) - FLOOR_MAE) / mae(best_model) * 100:.0f}% of its error) above that floor, "
    f"{'so nearly all remaining error is irreducible noise' if (mae(best_model) - FLOOR_MAE) / mae(best_model) < 0.15 else 'so there is still some structure a better model could capture'}.",
    f"6. Residuals: Ljung–Box p-values {', '.join(f'lag {k}: {v:.2f}' for k, v in diag['ljung_box_p'].items())} "
    f"({'no leftover autocorrelation' if min(diag['ljung_box_p'].values()) > 0.05 else 'some autocorrelation remains'}). "
    f"Prediction intervals labelled 80% covered {cov:.0f}% of days ({'close to nominal' if abs(cov - 80) <= 3 else 'narrower than nominal, so use the empirical buffer in point 8 rather than the model interval'}).",
    f"7. Model choice: the weather-aware models ({', '.join(weather_models)}) are "
    f"{'statistically indistinguishable from each other' if (wm_sep['separable'] == 'no').all() else 'not all equivalent'} on this backtest, "
    f"and the winner was picked on the same folds it is scored on, so treat the ranking among them as tentative. "
    f"On the 8-week static hold-out the lowest MAE is {H['MAE'].iloc[0]:.1f} ({hold_best}) against {H.loc['Weekday mean', 'MAE']:.1f} for the weekday-mean baseline.",
    f"8. Planning: prepare for forecast + {buf_all['buffer_orders']:.0f} orders (≈{buf_all['buffer_pct_of_forecast']:.0f}%) to cover 9 days in 10; "
    f"on rainy days the buffer is {buf_rain['buffer_orders']:.0f} orders, on dry days {buf_dry['buffer_orders']:.0f}.",
]
from IPython.display import Markdown
display(Markdown("\\n\\n".join(lines)))
""")

md("""
## Limits of this analysis
* **Three years of history, no yearly term.** The generator has no seasonal-by-month effect other than weather and festivals, so the models do not include one. On real data a yearly cycle would need to be tested.
* **Festival dates are treated as known in advance** (calendar flag), which is realistic for a planning system.
* **Oracle weather.** Weather-model accuracy assumes the day's weather is known. Using a weather forecast would raise the error.
* **Synthetic orders.** The order counts follow the generator's documented rules plus negative-binomial noise. Results show how well
  the methods recover those rules, not real customer behaviour.
* **Hyper-parameters** were chosen on the first 180 days only, which keeps the backtest honest but means the chosen orders may not be optimal.
* **Shared thresholds.** The weather flags (2.5 mm, 15 mm, 33 °C) are the team's regime definitions from `docs/generator.md`.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
out = Path(__file__).with_name("07_timeseries.ipynb")
nbf.write(nb, out)
print("wrote", out)
