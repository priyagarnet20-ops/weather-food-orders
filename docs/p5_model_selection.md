# Model selection: why these models, and how do we know?

**Owner:** P5 · **Evidence:** `notebooks/06_model_selection.ipynb`, `src/model_selection.py`, `results/selection_*.csv`, `fig14`-`fig16`

## The short answer
No model beat Ridge by more than chance, so under our decision rule Ridge stays. We cannot call it the single best
model: a plain 7-day average was statistically tied with it. Ridge is clearly better only than the seasonal naive and
weekday mean baselines.

## How the models were chosen
Measured on the 120 training rows before the first forecast:

| Data trait | Value | What it points to |
|---|---|---|
| Days per feature | 6.3 | Small sample: regularised linear models (Ridge, ElasticNet) |
| Variance / mean of daily orders | 4.67 | Wider than Poisson: Negative Binomial |
| Autocorrelation at 1 and 7 days | 0.79, 0.59 | Lag features, SARIMAX, seasonal baselines |
| ADF p / KPSS p | 0.12 / 0.01 | Level is drifting: rolling mean, differencing in SARIMAX |
| RESET p (linearity) | 0.72 | No evidence against linear form: trees are only a check |

Random Forest and XGBoost were kept as the test of whether non-linear effects exist. The data gave little reason to
expect them to win, and they did not.

## How the models were tested
- **Real days only:** 145 real test days, 9 May to 30 September 2025. Synthetic days are never scored, because they
  have no day-to-day memory (autocorrelation at 1 day: real 0.77, synthetic 0.08 and -0.02).
- **No future data:** a rolling backtest refits and re-tunes every 7 days using earlier days only. An automated test
  confirms that changing a day's actual orders never changes the forecast for that day or any earlier day.
- **Luck check:** Diebold-Mariano test with Holm correction, plus a 95% block-bootstrap interval.
- **Rule fixed in advance:** a model replaces Ridge only with lower MAE, Holm p < 0.05 and an interval below zero.

## Results (real test days, real-only training)
| Model | MAE | RMSE | MAPE % | R² | MAE vs Ridge | Holm p | Reading |
|---|---:|---:|---:|---:|---:|---:|---|
| SARIMAX | 15.59 | 21.02 | 7.59 | 0.59 | -0.50 | 0.75 | Tied with Ridge |
| 7-day rolling mean | 15.74 | 20.84 | 7.59 | 0.60 | -0.35 | 1.00 | Tied with Ridge |
| **Ridge** | **16.09** | 21.44 | 7.90 | 0.58 | | | Incumbent |
| ElasticNet | 16.20 | 21.35 | 7.86 | 0.58 | +0.11 | 1.00 | Tied |
| Poisson | 16.67 | 22.34 | 8.17 | 0.54 | +0.58 | 0.73 | Tied |
| Negative Binomial | 16.85 | 22.92 | 8.26 | 0.52 | +0.76 | 0.73 | Tied |
| Random Forest | 17.32 | 22.50 | 8.43 | 0.53 | +1.23 | 0.73 | Not separable |
| Extra Trees | 17.42 | 22.80 | 8.45 | 0.52 | +1.33 | 0.73 | Not separable |
| XGBoost | 17.52 | 23.43 | 8.46 | 0.49 | +1.43 | 0.67 | Not separable |
| Seasonal naive | 22.42 | 29.08 | 10.74 | 0.22 | +6.33 | <0.001 | **Worse than Ridge** |
| Weekday mean | 29.50 | 36.96 | 13.46 | -0.26 | +13.41 | <0.001 | **Worse than Ridge** |

## Other findings
- **Same-day weather did not improve accuracy.** Removing it moved Ridge from 16.09 to 15.81, and no model changed
  significantly. Most test days are dry season, so this does not prove weather never matters.
- **Synthetic training data did not help.** Every trained model had a higher MAE with it (Ridge 16.09 to 18.79). Only
  ElasticNet's increase is significant after correction, so the safe claim is "no evidence that it helps".
- **Recent orders carry most of the signal.** A 7-day average alone matches the full feature set.

## What we can and cannot claim
**Can say:** the model families were chosen from measured data traits; models were compared on real days using only
past data; simple seasonal baselines are clearly beaten; no candidate cleared the rule for replacing Ridge.

**Cannot say:** "Ridge is the best model", "XGBoost is proven worse", "weather improves the forecast", or that results
hold for rainy season.

## Limits
- One business, one period, mostly dry season: 28 rainy and 8 hot test days.
- Models use the actual weather of the predicted day, so their results with weather are a best case.
- The test period is not fully unseen: Ridge became the incumbent from an earlier 56-day holdout on August-September
  days, which also fall inside it.
- Version B's generator is refitted on pre-test days only. Its rain multipliers came out above 1, the opposite sign to
  the full-period estimates in `docs/generator.md`, so the rain effect depends on the period and needs checking.
