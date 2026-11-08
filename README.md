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

The forecast view supports a user-selected start and end date and a chronological holdout evaluation. Upcoming predictions default to Ridge, retrained on the latest available order history, and can be requested for any date after the last order through 2027-01-01. The dashboard graphs projected orders and shows daily weather and Tamil Nadu festival/occasion names only within the selected range. Open-Meteo weather is used where available; historical month-level medians fill dates beyond its forecast horizon. The current data ends on 2026-09-24, so projections bridge a gap and are illustrative only; update the order data before using them for staffing or inventory. The holdout view uses observed same-day weather, so it is an optimistic upper bound for live forecasting.

The **Rain rules & monsoon timeline** view compares two-item basket rules mined separately for rainy and dry days. Adjust the minimum support and lift sliders, then select a rule to chart its rolling 14-day lift across the SW monsoon (Jun-Sep), NE monsoon (Oct-Dec), and dry season (Jan-May).
