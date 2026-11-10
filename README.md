# Weather-Driven Food Order Patterns (Coimbatore)

Capstone project for **23CSE452 Business Analytics**, Amrita School of Computing, Coimbatore (Group 9; guide: Mrs. Rohini S).

How do rain, heat and festivals change daily food orders for a Coimbatore restaurant, and what should the kitchen do about it?
We join **real weather** (Open-Meteo, 1,097 days) with **synthetic orders** (182,696 orders, 80 menu items) generated from documented rules
that are calibrated on a real Swiggy order report and a 27-person survey. On that data we build predictive models, a time-series
forecast, weather-segmented association rules and a Streamlit dashboard.

> **Important:** the orders are synthetic. Findings show that the methods recover the rules built into the generator and agree with
> the real evidence we had. They are not measurements of real customer demand. See [Limitations](#limitations).

## Contents
1. [Team and ownership](#team-and-ownership)
2. [Key results](#key-results)
3. [Repository layout](#repository-layout)
4. [Setup](#setup)
5. [Reproduce the project](#reproduce-the-project)
6. [Dashboard](#dashboard)
7. [Data](#data)
8. [Documentation index](#documentation-index)
9. [Limitations](#limitations)

## Team and ownership

| Member | Module | Main files |
|---|---|---|
| P1 Priyadharshini | Weather data, time-series forecasting (T01-T06) | `src/weather.py`, `src/forecast.py`, `notebooks/07_timeseries.ipynb`, `docs/dataset.md` |
| P2 Aashiq Edavalapati | Menu catalog and order generator (T07-T12) | `src/generator.py`, `src/calibration.py`, `docs/generator.md` |
| P3 Sai Sri Satvic L | Cleaning, features, association rule mining (T13-T17) | `src/features.py`, `src/arm.py`, `notebooks/03_preprocess.ipynb`, `notebooks/06_arm.ipynb` |
| P4 Swasthika P S | Exploratory analysis, Streamlit dashboard (T18-T24) | `src/eda.py`, `dashboard/`, `notebooks/04_eda.ipynb` |
| P5 Deepak S | Predictive models, model selection, validation (T25-T26) | `src/predict.py`, `src/model_selection.py`, `src/validation.py`, `docs/p5_model_selection.md` |

Tasks are tracked on the project board and in `github/issues.csv`.

## Key results

All figures come from the files in `results/` and are reproducible with the commands below.

| Topic | Result | Source |
|---|---|---|
| Dataset | 1,097 days (24 Sep 2023 to 24 Sep 2026), 182,696 orders, mean 166.5 per day, range 73 to 354 | `data/raw/daily_orders.csv` |
| Dataset validation | 25 of 26 checks against the Swiggy report and survey pass; all five volume effects are recovered | `docs/validation.md`, `results/validation/` |
| Rain effect | rain raises weekday orders (about x1.56) and lowers weekend orders (about x0.53); weekend x2.45; heat x1.06; festival x0.87 | `results/validation/volume_effects.csv` |
| Prediction (56-day hold-out) | Random Forest MAE 14.6, Ridge 14.9, XGBoost 14.7 orders/day, R2 about 0.90 | `results/model_metrics_overall.csv` |
| Model selection (365 test days) | Ridge, SARIMAX, count models and Extra Trees are statistically tied (MAE 14.99 to 15.08); best possible score 14.77; XGBoost and the simple baselines are worse | `results/selection_metrics.csv`, `docs/p5_model_selection.md` |
| Value of weather | every model loses 18 to 22 orders/day of accuracy without weather and the rain x weekend term | `results/selection_weather_effect.csv` |
| Time-series forecast (917-day rolling backtest) | SARIMAX on log orders with weather: MAE 15.75, against 38.99 for the weekday mean and 48.20 for seasonal naive; weather saves 23.5 orders/day (95% CI 21.5 to 25.5) | `results/metrics_overall.csv`, `results/bootstrap_comparisons.csv` |
| Planning buffer | forecast + 25 orders (about 17%) covers 9 days in 10; 27 on rainy days, 24 on dry days | `results/planning_buffer.csv` |
| Association rules | five planted snack and drink pairs recovered; chai with pakora or samosa and coffee with vada are stronger on rainy days, ice cream with cold coffee on dry hot days | `results/rules_*.csv`, `slides/review2/` |

## Repository layout

```
data/
  raw/          weather.csv, weather_hourly.csv, weather_meta.json, menu.csv, orders.csv, daily_orders.csv
  real/         chinthamani_orders.csv, chinthamani_items.csv (Swiggy report), survey_responses.csv (survey)
  reference/    snack_drink_prices.csv
  processed/    daily_merged.csv, daily_features.csv
src/
  weather.py        Open-Meteo fetcher and validator (T02)
  real_orders.py    cleans the Swiggy order report
  survey.py         cleans the survey responses
  calibration.py    derives every generator parameter from the evidence
  generator.py      evidence-based order generator (menu, daily orders, baskets)
  validation.py     26 checks of the synthetic data against the evidence
  features.py       lag, rolling and weather features (T14)
  eda.py            exploratory analysis and charts
  predict.py        Ridge, Random Forest and XGBoost models (T25, T26)
  model_selection.py  11-model comparison on a 365-day backtest
  forecast.py       time-series preparation, models, backtest, evaluation (T04-T06)
  arm.py            association rule mining and rainy-vs-dry evaluation
  plotstyle.py      shared figure style
dashboard/          Streamlit app (app.py) with data, weather, forecasting and analytics helpers
notebooks/          03 preprocess, 04 EDA, 05 model, 05b model selection, 06 ARM, 07 time series, 08 validation
                    (each notebook is generated by notebooks/build_<name>.py)
results/            metric tables and figures written by the notebooks
slides/             deck builders (build_review1.py, build_review2_arm.py, build_review2_forecasting.py) and decks
docs/               dataset.md, generator.md, validation.md, p5_model_selection.md, p1_slides.md
tests/              pytest suite for every module
github/             board setup script and the task list
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the project

Run from the repository root. The data files are already committed, so you can start from any step.

```bash
# 1. Weather (needs internet; or use the offline export)
python src/weather.py --force
python src/weather.py --from-export data/raw/openmeteo_export_11.00N76.99E.csv

# 2. Synthetic orders (writes menu.csv, daily_orders.csv, orders.csv)
python src/generator.py

# 3. Validate the dataset against the Swiggy report and survey
python src/validation.py

# 4. Notebooks: regenerate the source, then execute
python notebooks/build_07_timeseries.py
jupyter nbconvert --to notebook --execute --inplace notebooks/07_timeseries.ipynb   # long: 131 rolling folds
# the same pattern works for build_03_preprocess, build_04_eda, build_05_model,
# build_05b_model_selection, build_06_arm and build_08_validation

# 5. Tests
pytest -q

# 6. Decks
python slides/build_review2_forecasting.py
```

If the weather is refetched, regenerate the orders (step 2) and rerun the notebooks, because the orders are generated from the weather file.
`src/real_orders.py` and `src/survey.py` clean the original Excel files and are only needed to rebuild `data/real/` (the originals are not in the repository).

## Dashboard

```bash
streamlit run dashboard/app.py
```

- **Forecast view:** upcoming forecasts from Open-Meteo's current 16-day weather forecast, plus a chronological hold-out evaluation. The order data ends on
  2026-09-24, so upcoming projections bridge a gap and are illustrative only. The hold-out view uses observed same-day weather, so it is an optimistic upper bound.
- **Rain rules and monsoon timeline:** two-item basket rules mined separately for rainy and dry days, with minimum support and lift sliders and a rolling 14-day lift
  across the south-west monsoon, north-east monsoon and dry season.

## Data

| Data | Source | Used for |
|---|---|---|
| Weather | Open-Meteo Historical Weather API, Coimbatore (11.00 N, 76.99 E), daily and hourly | all models |
| Swiggy order report | one Coimbatore restaurant, 464 orders (Aug to Sep 2026), no customer names or contact details | menu, prices, baskets, order hours, weekend and rain effects |
| Survey | 27 Coimbatore adults, no names or emails | snack and drink mix, festival effect, weather-linked pairs |
| Orders | generated by `src/generator.py` (fixed seed) | everything downstream |

Field definitions and the data profile are in `docs/dataset.md`; every generator parameter and its evidence are in `docs/generator.md`.

## Documentation index

| File | What it explains |
|---|---|
| `docs/dataset.md` | problem statement, sources, field dictionary and data profile |
| `docs/generator.md` | how the synthetic orders are generated and where each parameter comes from |
| `docs/validation.md` | the 26 checks of the dataset against the evidence |
| `docs/p5_model_selection.md` | why these models, and how the 365-day comparison was done |
| `docs/p1_slides.md` | slide text for the forecasting section |

## Limitations

- Orders are synthetic. The evidence behind them is one restaurant's Swiggy report (51 days overlap with the weather) and a 27-person survey.
- The survey majority says people order less in rain, but the Swiggy data shows weekday orders rising in rain. The generator follows the Swiggy data.
- Forecast and model results use the observed weather of each day, so they are best-case. A live system would use a weather forecast and add error.
- The forecasting models have no yearly term, because the generator has none. This would need testing on real data.
- Several top models are statistically tied, so the choice among them is tentative.
