"""Generates notebooks/06_model_selection.ipynb (run: python notebooks/build_06_model_selection.py).

Kept in the repo so the notebook source is reviewable as plain text in pull requests.
Execute afterwards with:
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/06_model_selection.ipynb
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
# 06 · Model selection and validation
**Owner:** P5 (Prediction Model & Monsoon Lift) · **Extends:** T25, T26

**Question.** Why were Ridge, Random Forest and XGBoost chosen, and how do we know the winner is really the best?

**Approach.**
1. Measure the traits of the data, using only days before the first forecast, and let them point to model families.
2. Compare the existing models with stronger baselines and new candidates.
3. Score **only real POS days**, with a rolling backtest in which every forecast uses only earlier days.
4. Test whether the differences are bigger than luck.
5. Apply a decision rule fixed before the results were seen.

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
## 1. Why synthetic days cannot be used for scoring
The generator draws each day independently, so synthetic days have realistic averages but no day-to-day memory.
""")
code("""
full = predict.load_model_frame(WEATHER_CSV, ORDERS_CSV)
real = ms.load_real_frame(WEATHER_CSV, ORDERS_CSV)
flags = pd.read_csv(ORDERS_CSV, parse_dates=["date"]).set_index("date")["is_synthetic"]
y_all = full["order_count"]
groups = {"Real (Jan-Sep 2025)": y_all[flags == 0],
          "Synthetic 2024": y_all[(flags == 1) & (y_all.index.year == 2024)],
          "Synthetic Oct-Dec 2025": y_all[(flags == 1) & (y_all.index >= "2025-10-01")]}
memory = pd.DataFrame({name: {"days": len(s), "mean": s.mean(), "sd": s.std(), "acf_1_day": s.autocorr(1),
                              "acf_7_days": s.autocorr(7)} for name, s in groups.items()}).T
memory.round(3).to_csv(RESULTS / "selection_real_vs_synthetic.csv")
memory.round(3)
""")
code("""
fig, ax = plt.subplots(figsize=(8, 3.4))
lags = range(1, 15)
for (name, s), color in zip(groups.items(), [C["blue"], C["orange"], C["gray"]]):
    ax.plot(lags, [s.autocorr(k) for k in lags], marker="o", color=color, label=name)
ax.axhline(0, color=C["gray"], lw=1); ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
ax.set_title("Real days have memory, synthetic days do not"); ax.legend()
fig.tight_layout(); fig.savefig(RESULTS / "fig14_real_vs_synthetic_memory.png", dpi=150); plt.show()
""")

md("""
## 2. Data profile: which traits point to which models
Measured only on the rows before the first forecast, so the test period does not influence the choice of models.
""")
code("""
X = predict.make_features(real)
y = real["order_count"].loc[X.index]
first = ms.first_test_day(X)
Xpre, ypre = X[X.index < first], y[y.index < first]
print(f"Training rows before the first forecast: {len(Xpre)}. Test days: {(X.index >= first).sum()}, "
      f"{first.date()} to {X.index.max().date()}")
profile = ms.profile_data(Xpre, ypre)
profile.round(3).to_csv(RESULTS / "selection_data_profile.csv", index=False)
pd.set_option("display.max_colwidth", None)
profile.round(3)
""")

md("""
## 3. Version A: real days only (decides the winner)
Each model is refitted and re-tuned every 7 days using only earlier days.
""")
code("""
preds_a = ms.rolling_backtest(X, y, ms.make_models(), first)
y_test = y.loc[preds_a.index]
table_a = ms.compare_to_incumbent(y_test, preds_a, ypre)
preds_a.round(2).to_csv(RESULTS / "selection_backtest_forecasts.csv")
table_a.astype(float).round(3).to_csv(RESULTS / "selection_metrics_real_only.csv")
table_a.astype(float).round(3)
""")
code("""
verdict = ms.decide(table_a)
print("Winner:", verdict["winner"]); print(verdict["reason"])
""")
code("""
others = table_a.drop(index=ms.INCUMBENT).astype(float).sort_values("delta_MAE")
fig, ax = plt.subplots(figsize=(8, 4))
pos = range(len(others))
ax.errorbar(others["delta_MAE"], pos, xerr=[others["delta_MAE"] - others["ci_low"], others["ci_high"] - others["delta_MAE"]],
            fmt="o", color=C["blue"], capsize=3)
ax.axvline(0, color=C["ink"], lw=1)
ax.set_yticks(list(pos)); ax.set_yticklabels(others.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("MAE difference from Ridge (orders/day), with 95% interval. Left of 0 = better than Ridge")
ax.set_title("Is any model really different from Ridge?")
fig.tight_layout(); fig.savefig(RESULTS / "fig15_mae_difference_vs_ridge.png", dpi=150); plt.show()
""")

md("""
### Where each model does well or badly
The test days are mostly dry season, so the rainy and hot columns rest on fewer days. Check the counts before reading them.
""")
code("""
ctx = real.loc[preds_a.index]
by_regime = ms.segment_table(y_test, preds_a, ctx["regime"])
by_day = ms.segment_table(y_test, preds_a, pd.Series(np.where(ctx["dow"].isin([4, 5, 6]), "Fri-Sun", "Mon-Thu"), index=ctx.index))
segments = by_regime.join(by_day).loc[table_a.index]
segments.round(2).to_csv(RESULTS / "selection_metrics_by_segment.csv")
print("Days:", by_regime.attrs["days"], by_day.attrs["days"])
segments.round(2)
""")

md("""
## 4. Version B: does synthetic training data help?
Same test days and same test features as version A. The extra training days are 2024 plus 1 January 2025, regenerated
from a generator calibrated **only on days before the first forecast**, so no test-period information can reach training.
The team's main dataset is not touched. Synthetic October-December 2025 is never used, because it is future data for
every test day.
""")
code("""
real_full = real  # all real days; the cutoff inside the calibration protects the test period
params = ms.calibrate_generator(real_full, first)
print("Calibrated on days before", first.date(), params)
hybrid = ms.build_hybrid_frame(full, real_full, first)
X_b = predict.make_features(hybrid)
y_b = hybrid["order_count"].loc[X_b.index]
preds_b = ms.rolling_backtest(X_b, y_b, ms.make_models(), first).loc[preds_a.index]
preds_b.round(2).to_csv(RESULTS / "selection_backtest_forecasts_hybrid.csv")

def versus(preds_other, label):
    rows = {}
    for m in preds_a.columns:
        ea, eo = y_test - preds_a[m], y_test - preds_other[m]
        stat, p = ms.diebold_mariano(eo, ea) if not np.allclose(ea, eo) else (0.0, 1.0)
        low, high = ms.block_bootstrap_ci(eo.abs() - ea.abs())
        rows[m] = {"MAE real only": ea.abs().mean(), f"MAE {label}": eo.abs().mean(),
                   "delta": (eo.abs() - ea.abs()).mean(), "dm_p": p, "ci_low": low, "ci_high": high}
    out = pd.DataFrame(rows).T
    out["dm_p_holm"] = ms.holm(out["dm_p"].tolist())
    return out.loc[table_a.index]

synth_effect = versus(preds_b, "with synthetic")
synth_effect.round(3).to_csv(RESULTS / "selection_synthetic_training_effect.csv")
synth_effect.round(3)
""")
code("""
fig, ax = plt.subplots(figsize=(9, 3.8))
pos = np.arange(len(synth_effect)); w = 0.4
ax.bar(pos - w / 2, synth_effect["MAE real only"], w, color=C["blue"], label="Real days only")
ax.bar(pos + w / 2, synth_effect["MAE with synthetic"], w, color=C["orange"], label="Synthetic + real")
ax.set_xticks(pos); ax.set_xticklabels(synth_effect.index, rotation=30, ha="right")
ax.set_ylabel("MAE on real test days"); ax.set_title("Does training on synthetic days help on real days?"); ax.legend()
fig.tight_layout(); fig.savefig(RESULTS / "fig16_synthetic_training_effect.png", dpi=150); plt.show()
""")

md("""
## 5. How much does weather help?
The models above use the **actual** weather of the predicted day, which a live system would not know in advance, so
their accuracy is a best case. This run removes same-day weather and monsoon phase. Baselines ignore weather and do
not change.
""")
code("""
kept = ms.no_weather_columns(X)
print("Features kept:", ", ".join(kept))
preds_nw = ms.rolling_backtest(X[kept], y, ms.make_models(), first).loc[preds_a.index]
weather_effect = versus(preds_nw, "without weather")
weather_effect.round(3).to_csv(RESULTS / "selection_weather_effect.csv")
weather_effect.round(3)
""")

md("""
## 6. Limits of this evidence
- **One period.** The test days run from about May to September 2025, mostly dry season, from one business in one year.
- **Few rainy and hot days.** Segment results rest on small counts, printed above.
- **Best-case weather.** Models use observed weather, not a forecast.
- **Candidates were not blind.** The earlier 56-day holdout had already shown Ridge ahead on real August-September days,
  and those days are also in this test period. The incumbent was fixed from that result, so the formal tests here are the
  stronger evidence, but the test period is not fully unseen.
- **Version B and the generator.** Its synthetic days are independent draws by design, so they can add evidence about
  calendar and weather effects but not about day-to-day memory.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parent / "06_model_selection.ipynb"
nbf.write(nb, out)
print("wrote", out)
