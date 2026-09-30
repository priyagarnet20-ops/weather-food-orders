# P1 slide content

## Review 1 · Data and dataset description (1 slide, about 90 seconds)
**Title:** Real weather, synthetic orders, fully documented

* **Weather (real):** Open-Meteo Historical Weather API, Coimbatore (11.00 N, 76.99 E), 24 Sep 2025 to 24 Sep 2026, 366 days, daily and hourly.
* **Orders (synthetic):** generated from documented rules (weekday, rain and heat multipliers, negative-binomial noise, fixed seed).
* **Why synthetic:** no real order data available; rules are known so we can check whether each method recovers them.
* **Quality:** no gaps or missing values; validated on every load; offline tests.
* **Weather profile:** 108 rainy days (>= 2.5 mm), 15 heavy-rain days, 106 days >= 33 C.
* Figure: `results/fig01_series.png`
* Speaker note: say explicitly that only the order counts are synthetic and that method and parameters are in `docs/dataset.md` and `docs/generator.md`.

## Review 2 · Time-series forecasting (2 slides, about 3 minutes)
### Slide A: Method and setup
* Series is stationary (ADF p = 0.001, KPSS p >= 0.10); weekly pattern only (one year of data, so no yearly term).
* Models: seasonal naive, weekday mean, SARIMAX (with and without weather), SARIMAX on log orders, Prophet.
* Rolling-origin backtest: 26 folds, 7-day forecasts, 182 forecast days. Weather given as observed (oracle).
* Figures: `results/fig03_decomposition.png`, `results/fig05_forecast_vs_actual.png`

### Slide B: Results and what it means
* Best MAE 32.7 orders/day (18.9% MAPE) vs 35.5 weekday-mean baseline and 46.6 seasonal naive.
* Weather knowledge lowers error by about 2.5 orders/day (7%), statistically distinguishable but small.
* Rainy-day error 40.9 vs dry-day 28.0: rainy days are hardest to plan for.
* Within about 1 order/day of the best possible score, so remaining error is noise.
* Plan for forecast + 51 orders (about 32%) to cover 9 days in 10; 71 on rainy days, 42 on dry days.
* Figures: `results/fig06_mae_by_model.png`, `results/fig07_error_rain_dry.png`
* Speaker note: model ranking among the three weather models is tentative (not statistically separable, chosen on the same folds it is scored on).
