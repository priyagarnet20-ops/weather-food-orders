# Weather-Driven Food Order Patterns (Coimbatore)

BA capstone, team of 5. Real Open-Meteo weather drives a documented synthetic order generator.
This folder currently holds **P1's work (Weather Data & Forecasting)**; other members add their modules to the same layout.

```
data/raw/            weather.csv (P1), weather_hourly.csv (P1, after API refresh), orders from P2
src/weather.py       Open-Meteo fetcher (T02)
src/forecast.py      time-series prep, models, backtest, evaluation (T04-T06)
src/plotstyle.py     shared figure style
notebooks/07_timeseries.ipynb   forecasting notebook (executed, outputs included)
docs/dataset.md      problem statement, sources, field dictionary (T03)
docs/p1_slides.md    slide content for the review decks
results/             figures and metric tables written by the notebook
tests/               offline tests for the fetcher
github/              board setup script + the 29-task issue list (T01)
```

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce P1
```
python src/weather.py --force            # refresh weather from Open-Meteo (needs internet)
python src/weather.py --from-export data/raw/openmeteo_export_11.00N76.99E.csv   # offline alternative
pytest -q                                # 7 offline tests
python notebooks/build_07_timeseries.py  # regenerate the notebook source
jupyter nbconvert --to notebook --execute --inplace notebooks/07_timeseries.ipynb   # ~2 minutes
```
The notebook reads the order counts from `ORDERS_CSV` (first code cell), which points at `data/raw/daily_orders.csv`. The full run takes about
20 minutes (131 rolling folds, six models).

## Data status
`data/raw/weather.csv` (1,097 days, 2023-09-24 to 2026-09-24) comes from the Open-Meteo Archive API and `data/raw/daily_orders.csv` was generated
from it by P2. If the weather is refetched, P2 must regenerate the orders and P1 re-runs the notebook. See `docs/dataset.md`.

## Review 2 Dashboard (T22-T24)

Install the requirements above, then start the interactive dashboard from the repository root:

```bash
streamlit run dashboard/app.py
```

The light-themed dashboard has five focused tabs: **Overview** introduces the capstone and summarizes order volume; **Dataset** describes coverage, sources, fields, generator assumptions, and limitations, with downloads for analysis CSVs, the real Chinthamani restaurant order and item data used as the generator's base, and supporting survey and reference-price data; **Demand patterns** compares weekday, weather, and monthly demand; **Forecasts** provides upcoming projections and chronological holdout evaluation; and **Menu insights** compares directed basket-rule confidence across rainy and dry days. Overview includes all-history, trailing-year, and trailing-90-day summaries. Menu insights defaults to the generator's three planted rain-effect pairs, uses 0.5% minimum support per weather segment, excludes 100%-confidence or lift-above-20 shortcuts, and sorts by rainy-minus-dry confidence. The table also includes rainy/dry lift, pair counts, and a weather-effect filter; its 14-day chart hides windows with fewer than 20 antecedent orders.

The forecast view supports upcoming Ridge projections, selected-model historical holdout evaluation, and a full-year model-selection comparison from saved rolling-origin results. The full-year view ranks seven models by MAE, marks models statistically tied with Ridge, and explains why Ridge remains selected under the pre-defined replacement rule. Upcoming predictions stay on Ridge and can be requested for any date after the last order through 2027-01-01. In historical holdout evaluation, users select which models to show alongside actual orders; the graph, ranked MAE chart, and conclusion reflect only those selections and the chosen dates/weather. This is a retrospective comparison using observed weather, not a future forecast; Random Forest and XGBoost use dashboard quick settings. Upcoming Ridge predictions are retrained on the latest available order history and recursively bridge from the last observed orders, so uncertainty increases with distance; weather values filled from historical monthly medians are marked in the table. This dashboard projection is distinct from the seven-day SARIMAX rolling backtest. The latest 90th-percentile planning buffer is about +25 orders overall (+27 rainy, +24 dry), and is not applied to dashboard predictions. Order baskets are synthetic, so menu patterns demonstrate generator behavior rather than actual customer buying behaviour.
