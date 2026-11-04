# P1 slide content

## Review 1 · Data and dataset description (1 slide, about 90 seconds)
**Title:** Real weather, real POS orders, grounded synthetic extension

* **Weather (real):** Open-Meteo Historical Weather API, Surabaya (-7.26 N, 112.75 E), 24 Sep 2023 to 24 Sep 2026, 1,097 days, daily and hourly. No gaps; validated on every load; offline tests.
* **Orders (real base):** 53,820 POS orders from a local F&B MSME in Surabaya (3 outlets, 96 menu items), 272 days, 2 Jan to 30 Sep 2025. Source: Mendeley Data `kcgf45y24m/2`, Universitas Ciputra Surabaya; mean 197.9 orders per day.
* **Orders (synthetic extension):** 459 days (2024 and Q4 2025) generated from a model fitted on the real base (Negative Binomial GLM, `docs/generator.md`), so the series is continuous at 731 days and 142,618 orders. Every row carries `is_synthetic`.
* **Why an extension:** nine months is short for a time-series method, and the course rules do not allow a pre-built public dataset to be used as it is.
* **Weather profile (all 1,097 days):** 466 rainy days (>= 2.5 mm), 115 heavy-rain days (>= 15 mm), 191 days >= 33 C.
* **Real days:** rainy 123 (45%), mean orders 190 on rainy days against 205 on dry days (raw gap, not adjusted for season).
* Figure: `results/fig01_series.png`
* Speaker note: say which part is real and which is synthetic; method and parameters are in `docs/dataset.md` and `docs/generator.md`.

## Review 2 · Time-series forecasting (2 slides, about 3 minutes)
### Slide A: Method and setup
* **Only real days are scored.** Real days have strong memory (autocorrelation 0.77 at lag 1); synthetic days almost none (0.08), so a forecast of them says nothing about a real outlet.
* **The real series:** a slowly moving level (April about 148 orders/day, June about 251), almost no weekday pattern (weekly seasonal strength 0.02), autocorrelation of an AR(2) type. ADF and KPSS give no clear answer on 272 points.
* **Models:** seasonal naive, weekday mean, 7-day mean, SARIMAX and Prophet, each with and without weather (rain, heavy rain, hot flags) plus a festival flag. Settings chosen on the first 127 days only.
* **Backtest:** rolling origin, 7-day forecasts, refitted weekly: 21 folds, 145 real days (9 May to 30 Sep 2025). Weather is given as observed (best case).
* Figures: `results/fig01_series.png`, `results/fig17_acf_pacf.png`, `results/fig03_decomposition.png`

### Slide B: Results and what it means
* **Accuracy (MAE, orders/day):** 7-day mean 17.2 (8.2% of mean daily orders), SARIMAX 18.4, SARIMAX with weather 18.9, seasonal naive 22.4, weekday mean 29.5, Prophet 30.7 and 31.8.
* **SARIMAX vs 7-day mean:** difference +1.2 orders/day, 95% interval -1.0 to +3.3: not separable. This agrees with P5's model selection (`docs/p5_model_selection.md`): recent orders carry most of the signal.
* **Weather does not improve the forecast:** with vs without weather +0.5 (SARIMAX) and +1.1 (Prophet) orders/day, both intervals include 0.
* **Weather effect is small and mostly heavy rain:** any rain >= 2.5 mm -1.4% of orders (95% interval -4.9% to +2.1%); heavy rain (>= 15 mm) adds -3.9% beyond that (-7.1% to -0.6%), about -5% for a heavy-rain day in total; hot days not distinguishable from 0.
* **The raw rainy vs dry gap (-7.2%) is larger than the adjusted effects** because rainy days cluster in the lower-level months (Jan to May). The model-based estimate controls for that drift; the raw gap does not.
* **Synthetic history does not help:** adding 367 synthetic days to training raises SARIMAX error by 3.7 to 4.6 orders/day, even though that variant has seen information about the test period.
* **Planning:** prepare for forecast + 33 orders (about 15%) to cover 9 days in 10.
* Figures: `results/fig06_mae_by_model.png`, `results/fig19_weather_effect.png`, `results/fig07_error_rain_dry.png`
* Speaker note: the honest result is that a 7-day average is as good as the fitted models and that weather adds a small, heavy-rain-only effect on this outlet. Only 28 of the 145 scored days are rainy and the period is mostly dry season, so the result does not cover the rainy season. Prophet does badly because its trend keeps extending an earlier level; do not present it as the recommended model.

### Where it is in the repository
| What | Where |
|---|---|
| Data preparation, models, backtest, evaluation functions | `src/forecast.py` |
| Notebook (T04 to T06) and its generator | `notebooks/07_timeseries.ipynb`, `notebooks/build_07_timeseries.py` |
| Tests | `tests/test_forecast.py` |
| Metrics tables | `results/metrics_overall.csv`, `bootstrap_comparisons.csv`, `weather_effect.csv`, `metrics_by_segment.csv`, `metrics_static_holdout.csv`, `metrics_synthetic_history.csv`, `planning_buffer.csv` |
| Forecasts | `results/backtest_forecasts.csv`, `results/backtest_forecasts_synthetic_history.csv` |
