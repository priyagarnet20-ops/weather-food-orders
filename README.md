# Weather-Driven Food Order Patterns (Surabaya, Indonesia)

**Business Analytics Capstone Project · Team of 5**

This repository investigates the empirical relationship between local meteorological conditions and food-and-beverage (F&B) demand. Using **real reanalysis weather data from the Open-Meteo API** matched with **authentic Point-of-Sale (POS) transactions from a local F&B MSME in Surabaya, Indonesia** (3 retail outlets combined, 53,820 orders, 96 menu items), we evaluate whether meteorological indicators can forecast order volume surges and uncover weather-conditioned consumer purchasing patterns.

---

## Team & Task Division

| Role | Member | Focus Areas | Completed R1 Tasks |
|---|---|---|---|
| **P1** | Weather & Forecasting | Open-Meteo Surabaya weather fetcher, data quality, dataset documentation; Review 2 time-series forecasting (SARIMAX, Prophet, baselines, weather effect) | T01, T02, T03 (R1); T04, T05, T06 (R2) |
| **P2** | Menu Catalog & Preprocessing | 96-item Surabaya menu catalog, multi-outlet aggregation, basket parsing, meal slots, price imputation | T07, T08, T09, T10 |
| **P3** | Cleaning & Feature Engineering | Orders-weather merge, lag/rolling features, Indonesian holidays, Surabaya NW monsoon phases, VIF checks | T13, T14, T15 |
| **P4** | EDA & Visualizations | Exploratory data analysis, 10-figure presentation chart set, 14-slide Review 1 PowerPoint deck | T19, T20, T21 |
| **P5** | Predictive Modeling | Ridge, Random Forest, XGBoost, Seasonal Naive baseline, chronological hold-out validation, SHAP & feature importance | T25, T26 |

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── Products.csv             # Raw MSME product catalog (Universitas Ciputra, Surabaya)
│   │   ├── Transactions.csv         # Raw POS transactions across 3 outlets (53,820 orders)
│   │   ├── menu.csv                 # Cleaned 96-item menu catalog with weather affinity tags
│   │   ├── orders.csv               # Aggregated POS orders with meal slots and parsed baskets
│   │   ├── order_items.csv          # Normalized order-item pairs for category & ARM analysis
│   │   ├── daily_orders.csv         # Daily aggregated orders joined with weather metrics
│   │   ├── weather.csv              # Open-Meteo daily weather for Surabaya (-7.26°N, 112.75°E)
│   │   ├── weather_hourly.csv       # Open-Meteo hourly weather (26,328 observations)
│   │   └── weather_meta.json        # Open-Meteo API metadata and validation results
│   └── processed/
│       ├── daily_merged.csv         # Clean joined orders and weather dataset (272 days)
│       └── daily_features.csv       # Engineered 28-feature dataset for predictive modeling
├── docs/
│   ├── dataset.md                   # Problem statement, API parameters, dictionary, data profile
│   └── generator.md                 # POS preprocessing, multi-outlet aggregation & meal slot logic
├── github/
│   └── issues.csv                   # GitHub project tracking issues across all 29 tasks
├── notebooks/
│   ├── 03_preprocess.ipynb          # Data cleaning, feature engineering & VIF checks (T13–T15)
│   ├── 04_eda.ipynb                 # Exploratory data analysis across 5 dimensions (T19–T20)
│   ├── 05_model.ipynb               # Predictive modeling, evaluation & SHAP analysis (T25–T26)
│   └── 07_timeseries.ipynb          # Time-series forecasting on the real days, with/without weather (T04–T06)
├── results/
│   ├── eda/                         # 10 exported presentation figures (PNG)
│   ├── fig09_model_metrics.png      # Model comparison metrics chart
│   ├── fig10_actual_vs_pred.png     # Actual vs. predicted time-series plot
│   ├── fig11_residuals.png          # Residual diagnostics
│   ├── fig12_importance.png         # Feature importance chart
│   ├── fig13_shap.png               # SHAP summary plot
│   ├── model_metrics_overall.csv    # Final evaluation metrics (MAE, RMSE, MAPE)
│   └── model_feature_importance.csv # Permutation importance rankings
├── slides/
│   ├── build_review1.py             # Script to build Review 1 presentation deck
│   └── review1/
│       ├── figures/                 # Embedded presentation chart set
│       └── review1_deck.pptx        # 14-slide widescreen presentation deck
├── src/
│   ├── weather.py                   # Open-Meteo weather fetcher & caching
│   ├── preprocess.py                # Multi-outlet aggregation, menu catalog & meal slots
│   ├── features.py                  # Feature engineering & calendar enrichment
│   ├── eda.py                       # Exploratory data analysis & figure generator
│   ├── predict.py                   # Machine learning models & evaluation pipeline
│   ├── forecast.py                  # Time-series preparation, SARIMAX/Prophet/baselines, backtest & evaluation
│   └── plotstyle.py                 # Shared presentation visualization styling
└── tests/                           # 62 automated unit tests (100% passing)
```

---

## Setup & Reproduction

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/
# Output: 62 passed in ~1.7s
```

### 3. Run Preprocessing Pipeline
```bash
# 1. Fetch Surabaya weather data from Open-Meteo (or use cached CSVs)
python src/weather.py --force

# 2. Preprocess Surabaya MSME POS dataset and aggregate 3 outlets
python src/preprocess.py

# 3. Generate feature set with Indonesian calendar & monsoon phases
python src/features.py
```

### 4. Run EDA, Visualizations & Models
```bash
# Generate all 10 presentation charts
python src/eda.py

# Train predictive models (Ridge, XGBoost, Random Forest) and export metrics
python src/predict.py

# Build Review 1 presentation deck
python slides/build_review1.py
```

### 5. Re-execute Notebooks
```bash
python notebooks/build_03_preprocess.py
jupyter nbconvert --to notebook --execute --inplace notebooks/03_preprocess.ipynb

python notebooks/build_04_eda.py
jupyter nbconvert --to notebook --execute --inplace notebooks/04_eda.ipynb

python notebooks/build_05_model.py
jupyter nbconvert --to notebook --execute --inplace notebooks/05_model.ipynb
```

---

## Key Review 1 Findings

1. **Precipitation Moderates Walk-in Foot Traffic:**  
   Daily order volume drops by **-7.2%** on rainy days (189.8 orders/day) compared to dry days (204.5 orders/day), reaching 186.8 orders/day during heavy monsoon downpours ($\ge 15\text{ mm}$). This difference is statistically significant (Welch's $t = -3.79, p = 1.89 \times 10^{-4}$).
2. **Meal Slot Clustering:**  
   Dinner (18:00–23:00) is the dominant operational window accounting for **36.4%** of all orders, followed by Lunch (**23.7%**), Breakfast (**20.8%**), and Afternoon Snacks (**19.2%**).
3. **Menu Category Substitution:**  
   Coffee anchors **36.2%** of sales. During tropical downpours, consumer preference pivots toward hot comfort beverages (Hot Coffee, Artisan Teas) and warm pastries (`Donat Polos`), whereas iced refreshments dominate during hot dry afternoons.
4. **Predictive Performance:**  
   Ridge regression achieves **13.62 MAE** (7.36% MAPE) and XGBoost achieves **14.39 MAE** (7.75% MAPE) on the held-out 56-day test window, outperforming the Seasonal Naive baseline (17.34 MAE) by **21.5%**. Top predictive drivers include historical 7-day order lag, Sunday/Friday calendar flags, and heavy rainfall indicators.
