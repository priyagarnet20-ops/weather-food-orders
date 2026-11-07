"""Generates notebooks/05b_model_selection.ipynb (run: python notebooks/build_05b_model_selection.py).

Kept in the repo so the notebook source is reviewable as plain text in pull requests.
Execute afterwards with:
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/05b_model_selection.ipynb
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
# 05b · Model selection and validation
**Owner:** P5 (Prediction Model & Monsoon Lift) · **Extends:** T25, T26

**Question.** Why were Ridge, Random Forest and XGBoost chosen, and how do we know the winner is really the best?

**Approach.**
1. Measure the traits of the data, using only days before the first forecast, and let them point to model families.
2. Compare the existing models with stronger baselines and new candidates (11 in total).
3. Test on the **last year** (365 days: both monsoons and both hot seasons) with a rolling backtest in which every
   forecast uses only earlier days; models are refitted and re-tuned every 7 days.
4. Score the generator's own expected value as a reference: the best any model could do, since the orders are synthetic.
5. Test whether the differences are bigger than luck, and apply a decision rule fixed before the results were seen.

**Decision rule.** A candidate replaces Ridge only if its MAE is lower, its Holm-adjusted Diebold-Mariano p-value is
below 0.05, **and** its 95% block-bootstrap interval for the MAE difference lies below zero. Otherwise Ridge stays.
""")

code("""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import model_selection as ms, plotstyle, predict
plotstyle.apply()
C = plotstyle.C

ORDERS_CSV = ROOT / "data/raw/daily_orders.csv"
WEATHER_CSV = ROOT / "data/raw/weather.csv"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
""")

md("""
## 1. Data profile: which traits point to which models
Measured only on the rows before the first forecast, so the test period does not influence the choice of models.
""")
code("""
full = predict.load_model_frame(WEATHER_CSV, ORDERS_CSV)
X = predict.make_features(full)
y = full["order_count"].loc[X.index]
first = ms.first_test_day(X, initial=len(X) - ms.TEST_DAYS)
Xpre, ypre = X[X.index < first], y[y.index < first]
print(f"Training rows before the first forecast: {len(Xpre)}. Test days: {(X.index >= first).sum()}, "
      f"{first.date()} to {X.index.max().date()}")
profile = ms.profile_data(Xpre, ypre)
profile.round(4).to_csv(RESULTS / "selection_data_profile.csv", index=False)
pd.set_option("display.max_colwidth", None)
profile.round(4)
""")

md("""
## 2. Rolling backtest on the last year
Every model is refitted and re-tuned every 7 days using only earlier days. The generator's expected value is added as
a reference row; it is not a candidate and is not part of the significance tests.
""")
code("""
preds = ms.rolling_backtest(X, y, ms.make_models(), first)
preds[ms.REFERENCE] = ms.generator_expected(ORDERS_CSV, preds.index)
y_test = y.loc[preds.index]
table = ms.compare_to_incumbent(y_test, preds, ypre)
preds.round(2).to_csv(RESULTS / "selection_backtest_forecasts.csv")
table.astype(float).round(3).to_csv(RESULTS / "selection_metrics.csv")
table.astype(float).round(3)
""")
code("""
verdict = ms.decide(table)
print("Winner:", verdict["winner"]); print(verdict["reason"])
""")
code("""
others = table.drop(index=[ms.INCUMBENT, ms.REFERENCE]).astype(float).sort_values("delta_MAE")
fig, ax = plt.subplots(figsize=(8, 4.2))
pos = range(len(others))
ax.errorbar(others["delta_MAE"], pos, xerr=[others["delta_MAE"] - others["ci_low"], others["ci_high"] - others["delta_MAE"]],
            fmt="o", color=C["blue"], capsize=3)
ax.axvline(0, color=C["ink"], lw=1)
ref = float(table.loc[ms.REFERENCE, "delta_MAE"])
ax.axvline(ref, color=C["gray"], lw=1, ls="--")
ax.text(ref, len(others) - 0.4, " best possible", color=C["ink2"], fontsize=8, va="bottom")
ax.set_yticks(list(pos)); ax.set_yticklabels(others.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("MAE difference from Ridge (orders/day), with 95% interval. Left of 0 = better than Ridge")
ax.set_title("Is any model really different from Ridge?")
fig.tight_layout(); fig.savefig(RESULTS / "fig20_mae_difference_vs_ridge.png", dpi=150); plt.show()
""")

md("""
### Where each model does well or badly
Rain lifts weekday orders and lowers weekend orders, so errors are split by weather and by day type together.
""")
code("""
ctx = full.loc[preds.index]
day_type = np.where(ctx["dow"].isin([4, 5, 6]), "Fri-Sun", "Mon-Thu")
cell = pd.Series([f"{r} {d}" for r, d in zip(ctx["regime"], day_type)], index=ctx.index)
segments = ms.segment_table(y_test, preds, cell).loc[table.index]
segments.round(2).to_csv(RESULTS / "selection_metrics_by_segment.csv")
print("Days per cell:", segments.attrs.get("days") or ms.segment_table(y_test, preds, cell).attrs["days"])
segments.round(2)
""")
code("""
show = [ms.INCUMBENT, "Random Forest", "SARIMAX", "Rolling mean", ms.REFERENCE]
cols = [c for c in segments.columns if c.endswith(("Mon-Thu", "Fri-Sun"))]
fig, ax = plt.subplots(figsize=(10, 3.8))
pos = np.arange(len(cols)); w = 0.16
colors = [C["blue"], C["orange"], C["aqua"], C["gray"], C["light"]]
for k, (m, col) in enumerate(zip(show, colors)):
    ax.bar(pos + (k - 2) * w, segments.loc[m, cols], w, color=col, label=m)
ax.set_xticks(pos); ax.set_xticklabels([c.replace("MAE ", "") for c in cols], rotation=20, ha="right")
ax.set_ylabel("MAE (orders/day)"); ax.set_title("Error by weather and day type"); ax.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(RESULTS / "fig22_error_by_segment.png", dpi=150); plt.show()
""")

md("""
## 3. How much does weather help?
The models above use the **actual** weather of the predicted day, which a live system would not know in advance, so
their accuracy is a best case. This run removes same-day weather, the rain x weekend term and monsoon phase. Baselines
ignore weather and do not change.
""")
code("""
kept = ms.no_weather_columns(X)
print("Features kept:", ", ".join(kept))
preds_nw = ms.rolling_backtest(X[kept], y, ms.make_models(), first).loc[preds.index]
candidates = [m for m in preds.columns if m != ms.REFERENCE]
rows = {}
for m in candidates:
    e_w, e_nw = y_test - preds[m], y_test - preds_nw[m]
    stat, p = ms.diebold_mariano(e_nw, e_w) if not np.allclose(e_w, e_nw) else (0.0, 1.0)
    low, high = ms.block_bootstrap_ci(e_nw.abs() - e_w.abs())
    rows[m] = {"MAE with weather": e_w.abs().mean(), "MAE without weather": e_nw.abs().mean(),
               "weather saves": (e_nw.abs() - e_w.abs()).mean(), "dm_p": p, "ci_low": low, "ci_high": high}
weather_effect = pd.DataFrame(rows).T
weather_effect["dm_p_holm"] = ms.holm(weather_effect["dm_p"].tolist())
weather_effect = weather_effect.loc[[m for m in table.index if m in candidates]]
weather_effect.round(3).to_csv(RESULTS / "selection_weather_effect.csv")
weather_effect.round(3)
""")
code("""
fig, ax = plt.subplots(figsize=(9, 3.8))
pos = np.arange(len(weather_effect)); w = 0.4
ax.bar(pos - w / 2, weather_effect["MAE with weather"], w, color=C["blue"], label="With weather")
ax.bar(pos + w / 2, weather_effect["MAE without weather"], w, color=C["orange"], label="Without weather")
ax.axhline(float(table.loc[ms.REFERENCE, "MAE"]), color=C["gray"], ls="--", lw=1, label="Best possible")
ax.set_xticks(pos); ax.set_xticklabels(weather_effect.index, rotation=30, ha="right")
ax.set_ylabel("MAE on the last year"); ax.set_title("How much does knowing the weather help?"); ax.legend()
fig.tight_layout(); fig.savefig(RESULTS / "fig21_weather_effect.png", dpi=150); plt.show()
""")

md("""
## 4. Limits of this evidence
- **Synthetic orders.** All days come from the evidence-based generator (`docs/generator.md`), so the comparison shows
  which model recovers its documented rules best; the generator's expected value marks the ceiling.
- **No day-to-day memory.** The generator draws each day independently, so lag features can only learn the weekly
  pattern. Real restaurant data usually has memory, which would favour lag-based models more than here.
- **Best-case weather.** Models use observed weather, not a forecast.
- **One test year.** 2025-09-25 to 2026-09-24. Rainy-weekend and hot cells have fewer days; their counts are printed above.
- **Candidates were not blind.** Ridge was fixed as the incumbent before this run, from the earlier 56-day holdout
  (05_model), whose days fall inside this test year.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parent / "05b_model_selection.ipynb"
nbf.write(nb, out)
print("wrote", out)
