# P1 slide content

## Review 1 · Data and dataset description (1 slide, about 90 seconds)
**Title:** Real weather, synthetic orders, fully documented

* **Weather (real):** Open-Meteo Historical Weather API, Coimbatore (11.00 N, 76.99 E), 24 Sep 2023 to 24 Sep 2026, 1,097 days, daily and hourly.
* **Orders (synthetic, evidence-based):** generated from documented rules (weekend, rain, heat and festival multipliers, negative-binomial noise, fixed seed): 182,696 orders over 1,097 days, mean 166.5 per day, 80-item menu.
* **Why synthetic, and what it is based on:** no full order history available. Parameters come from a Swiggy order report of one Coimbatore restaurant (464 orders) and a 27-person Coimbatore survey (weekend x2.47, weekday rain x1.56, weekend rain x0.54, heat x1.07, festival x0.89); the rules are known, so we can check whether each method recovers them.
* **Quality:** no gaps or missing values; validated on every load; offline tests.
* **Weather profile:** 329 rainy days (>= 2.5 mm), 67 heavy-rain days, 300 days >= 33 C; 41 festival days.
* **Mean orders by day type:** hot 177, normal 169, rainy 153 (rain raises weekday orders and lowers weekend orders, so the average over all rainy days is lower).
* Figure: `results/fig01_series.png`
* Speaker note: say explicitly that only the order counts are synthetic and that method and parameters are in `docs/dataset.md` and `docs/generator.md`.

## Review 2 · Time-series forecasting (2 slides, about 3 minutes)
### Slide A: Method and setup
* Series is stationary (ADF statistic -6.08, p < 0.001; KPSS p >= 0.10); strong weekly pattern (seasonal strength 0.56) plus festival days, no yearly term (the generator has none).
* Models: seasonal naive, weekday mean, SARIMAX (with and without weather), SARIMAX on log orders, Prophet. Festival flag is given to every regression model.
* Rolling-origin backtest: 131 folds, 7-day forecasts, 917 forecast days. Weather given as observed (oracle).
* Figures: `results/fig03_decomposition.png`, `results/fig05_forecast_vs_actual.png`

### Slide B: Results and what it means
* Best MAE 15.8 orders/day (10.1% MAPE) vs 39.0 for the weekday-mean baseline and 48.2 for seasonal naive.
* Weather knowledge (with the rain x weekend term) lowers SARIMAX error by 23.5 orders/day (60%; 95% CI 21.5-25.5).
* Rainy-day error 15.4 vs dry-day 15.9: once the weather is known, rainy days are no harder to plan.
* Within about 0.2 orders/day of the best possible score (generator expected value, MAE 15.5), so remaining error is noise.
* Plan for forecast + 25 orders (about 17%) to cover 9 days in 10; 27 on rainy days, 24 on dry days.
* Figures: `results/fig06_mae_by_model.png`, `results/fig07_error_rain_dry.png`
* Speaker note: ranking among the three weather models is tentative (not statistically separable, chosen on the same folds it is scored on). Results show recovery of the generator's rules, not real customer behaviour; the generator's parameters come from a small Swiggy report and survey, so they are indicative, not exact.
