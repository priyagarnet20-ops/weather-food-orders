# Model selection: why these models, and how do we know?

**Owner:** P5 · **Evidence:** `notebooks/05b_model_selection.ipynb`, `src/model_selection.py`, `results/selection_*.csv`, `fig20`-`fig22`
**Data:** the evidence-based Coimbatore dataset (`docs/generator.md`, validated in `docs/validation.md`)

## The short answer
No model beat Ridge by more than chance, so under our decision rule Ridge stays. SARIMAX, Negative Binomial, Poisson,
Extra Trees and ElasticNet are statistically tied with it. All of them are within 0.3 orders/day of the best possible
score (the generator's own expected value). Ridge is clearly better than XGBoost and all three baselines.
Knowing the weather cuts every model's error by 18-22 orders/day, more than half.

## How the models were chosen
Measured on the 725 training days before the first forecast:

| Data trait | Value | What it points to |
|---|---|---|
| Days per feature | 34.5 | Enough data for any family, including trees |
| Variance / mean of daily orders | 29.1 | Wider than Poisson: Negative Binomial |
| Autocorrelation at 1 and 7 days | 0.32, 0.54 | Lag features, SARIMAX, seasonal baselines |
| ADF p / KPSS p | 0.0001 / 0.10 | Stationary: no differencing needed |
| RESET p (linearity) | 0.014 | Some non-linearity left: trees as a serious check |
| Rain x weekend F-test p | < 0.0001 | Rain works differently on weekdays and weekends: linear models need the interaction term |

So the candidate list covers each family the data points to:
- **Ridge and ElasticNet**, regularised linear models with the rain x weekend term.
- **Poisson and Negative Binomial**, count models.
- **SARIMAX**, which models the weekly memory.
- **Random Forest, Extra Trees and XGBoost**, which can find non-linear effects on their own.
- **Three simple baselines**: seasonal naive, 7-day rolling mean and weekday mean.

Ridge was the incumbent from the earlier 56-day holdout (`05_model`).

## How the models were tested
- **One full year of test days:** 365 days, 25 Sep 2025 to 24 Sep 2026, covering both monsoons and both hot seasons.
- **No future data:** a rolling backtest refits and re-tunes every model every 7 days using earlier days only. An
  automated test confirms that changing a day's actual orders never changes the forecast for that day or any earlier
  day.
- **Best possible reference:** the orders are synthetic, so the generator's expected value for each day shows the
  ceiling. It is reported, not ranked.
- **Luck check:** Diebold-Mariano test with Holm correction, plus a 95% block-bootstrap interval.
- **Rule fixed in advance:** a model replaces Ridge only with lower MAE, Holm p < 0.05 and an interval below zero.

## Results (last year, 365 days)
| Model | MAE | RMSE | MAPE % | R² | MAE vs Ridge | Holm p | 95% interval | Reading |
|---|---:|---:|---:|---:|---:|---:|---|---|
| *Generator expected value* | *14.77* | *18.70* | *9.35* | *0.925* | *-0.31* | | | *Best possible* |
| SARIMAX | 14.99 | 19.22 | 9.48 | 0.921 | -0.09 | 1.00 | -0.32 to 0.14 | Tied with Ridge |
| Negative Binomial | 14.99 | 19.03 | 9.45 | 0.923 | -0.09 | 1.00 | -0.28 to 0.10 | Tied |
| Poisson | 15.03 | 19.11 | 9.47 | 0.922 | -0.05 | 1.00 | -0.26 to 0.16 | Tied |
| Extra Trees | 15.08 | 19.40 | 9.52 | 0.920 | 0.00 | 1.00 | -0.41 to 0.43 | Tied |
| **Ridge** | **15.08** | 19.36 | 9.50 | 0.920 | | | | Incumbent |
| ElasticNet | 15.10 | 19.38 | 9.52 | 0.920 | +0.02 | 1.00 | -0.03 to 0.07 | Tied |
| Random Forest | 15.60 | 19.92 | 9.85 | 0.915 | +0.52 | 0.18 | 0.02 to 1.03 | Probably worse, not significant after correction |
| XGBoost | 16.06 | 20.45 | 10.09 | 0.911 | +0.98 | 0.011 | 0.37 to 1.61 | **Worse than Ridge** |
| Weekday mean | 37.98 | 47.52 | 23.87 | 0.518 | +22.90 | <0.001 | 20.0 to 26.3 | **Worse than Ridge** |
| Seasonal naive | 46.30 | 62.66 | 29.40 | 0.162 | +31.23 | <0.001 | 25.4 to 36.1 | **Worse than Ridge** |
| 7-day rolling mean | 59.23 | 70.32 | 38.61 | -0.055 | +44.15 | <0.001 | 38.5 to 49.8 | **Worse than Ridge** |

## Other findings
- **Weather is the biggest single input.** Removing same-day weather, the rain x weekend term and the monsoon phase
  raises MAE by 22.4 orders/day for Ridge (15.1 to 37.4, 95% interval 19.3 to 25.7) and by 18-22 for every other model.
  All of these differences are significant after Holm correction (`selection_weather_effect.csv`, `fig21`).
- **Errors are similar across weather and day types.** Weekends are harder for every model, including the best
  possible (MAE about 20 versus 11 on weekdays), because they carry more orders and so more noise. On rainy weekdays the
  tree models do worst: Random Forest 19.0 and XGBoost 18.8, against Ridge 17.0 (`fig22`).
- **Why the 7-day average fails here.** In the Surabaya data a 7-day average tied Ridge. Here the weekly cycle is strong
  (weekends about 1.8x weekdays) and the generator has no day-to-day memory, so averaging across the week mixes weekdays
  with weekends.
- **Why trees do not win despite RESET.** RESET finds some non-linearity, but most of it is the rain x weekend switch.
  Once the linear models have that term, there is too little left for trees to gain, and XGBoost overfits slightly.

## What we can and cannot claim
**Can say:**
- The model families were chosen from measured data traits.
- Models were compared on a full year using only past data.
- The baselines and XGBoost are clearly beaten.
- No candidate cleared the rule for replacing Ridge.
- The leading models are within 0.3 orders/day of the best possible.
- Weather knowledge more than halves the error.

**Cannot say:**
- That Ridge is the single best model.
- That trees can never help.
- That these numbers hold for a real restaurant. The data are synthetic: they show how well each model recovers the
  evidence-based generator rules.

## Limits
- **Synthetic orders, no memory.** Each day is drawn independently, so lag features can only learn the weekly pattern.
  Real restaurant data usually has day-to-day memory, which would help SARIMAX and lag-based models more than here.
- **Best-case weather.** Models use the actual weather of the predicted day, not a forecast.
- **Small cells.** Some weather x day-type cells are small (33 hot weekend days, 50 rainy weekend days).
- **The test year is not fully unseen.** Ridge became the incumbent from an earlier 56-day holdout whose days fall
  inside it, so the formal tests here are the stronger evidence.
- **One generator.** A different but equally plausible generator (for example a different rain effect) could change
  the ranking among the tied models.
