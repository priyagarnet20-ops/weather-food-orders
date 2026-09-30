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
The notebook reads the order counts from `ORDERS_CSV` (first code cell). It points at P2's draft file until the final
`data/raw/orders.csv` exists; switching is a one-line change.

## Important before anyone finalises results
`data/raw/weather.csv` currently comes from a manual Open-Meteo download (GMT days, no humidity). After running
`python src/weather.py --force` the daily values change slightly, so P2 must regenerate the orders from the refreshed weather and P1 re-runs the notebook.
See `docs/dataset.md` section 3.1.
