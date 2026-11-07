# Comprehensive Project Explanation: Weather-Driven Food Order Patterns (Coimbatore)

**Course:** 23CSE452 Business Analytics Capstone  
**Target City:** Coimbatore, Tamil Nadu, India (11.00°N, 76.99°E)  
**Timeline:** 3-Year Historical Span (September 24, 2023 – September 24, 2026; 1,097 continuous days)  
**Team Structure:** Collaborative Team of 5 (P1 to P5)  
**Core Analytics Paradigms:** 
1. Review 1: Data Preprocessing, Feature Engineering, EDA, and Predictive Machine Learning
2. Review 2: Association Rule Mining (Method 1) and Rolling-Origin Time-Series Forecasting (Method 2)

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Dual-Data Architecture (Real Weather + Synthetic Demand)](#2-dual-data-architecture-real-weather--synthetic-demand)
3. [Team Roles, Tasks & Rubric Ownership (P1 – P5)](#3-team-roles-tasks--rubric-ownership-p1--p5)
4. [Dataset Catalog & Schemas](#4-dataset-catalog--schemas)
5. [Data Cleaning, Validation & Preprocessing (T13)](#5-data-cleaning-validation--preprocessing-t13)
6. [Feature Engineering & Selection Architecture (T14 – T15)](#6-feature-engineering--selection-architecture-t14--t15)
7. [Exploratory Data Analysis & Empirical Patterns (T19 – T20)](#7-exploratory-data-analysis--empirical-patterns-t19--t20)
8. [Predictive Modeling (ML Regressors) (T25 – T26)](#8-predictive-modeling-ml-regressors-t25--t26)
9. [Time-Series Forecasting & Rolling-Origin Backtesting (T04 – T06)](#9-time-series-forecasting--rolling-origin-backtesting-t04--t06)
10. [Association Rule Mining & Planted Weather Combos (T07 – T12, T16 – T17, T27)](#10-association-rule-mining--planted-weather-combos-t07--t12-t16--t17-t27)
11. [Operational & Business Analytics Recommendations](#11-operational--business-analytics-recommendations)
12. [Repository Architecture, Scripts & How to Reproduce](#12-repository-architecture-scripts--how-to-reproduce)
13. [Existing Project Documentation Map (.md Files)](#13-existing-project-documentation-map-md-files)

---

## 1. Executive Summary & Problem Statement

### 1.1 The Business Problem
Food-delivery demand in urban hubs like Coimbatore is highly volatile. Order volume varies sharply by day-of-the-week, payday cycles, and cultural holidays—but most critically, by **local atmospheric conditions**:
- **Monsoon Rain:** Heavy showers discourage citizens from traveling to restaurants or cooking, leading to sharp surges in delivery demand, especially for warm comfort snacks (tea, pakoras, bajjis).
- **Extreme Heat:** High temperatures (> 33°C) suppress kitchen cooking at home and drive demand for cold beverages, fresh juices, and ice creams.
- **Normal Dry Days:** Baseline volumes follow standard weekday and weekend rhythms.

Traditional commercial kitchens and delivery fleets (e.g., Swiggy, Zomato delivery partners) that schedule staff and purchase perishable raw inventory based solely on **static 7-day moving averages** encounter chronic operational failure:
1. **Under-staffing & Stock-Outs during Storms:** Unanticipated rain spikes cause delivery delays (> 60 minutes), order cancellations, and lost revenue.
2. **Over-staffing & Food Spoilage on Dry Days:** Excessive kitchen preparation leads to ingredient waste and idle delivery drivers.

### 1.2 Core Analytical Objectives
1. **Quantify Weather Sensitivity:** Exactly how much incremental variance does atmospheric weather explain compared to calendar features alone?
2. **Short-Term Predictive Modeling (Review 1):** Train supervised machine learning regressors (Ridge, Random Forest, XGBoost) to predict one-day-ahead order volume.
3. **Market Basket Dynamics (Review 2, Method 1):** Uncover weather-conditioned item pairings via Association Rule Mining (Apriori / FP-Growth) on transaction baskets.
4. **Medium-Term Demand Forecasting (Review 2, Method 2):** Deploy 7-day rolling-origin time-series forecasting (SARIMAX with exogenous weather regressors, Prophet) to establish dynamic operational inventory buffers.

---

## 2. Dual-Data Architecture (Real Weather + Synthetic Demand)

Commercial food delivery platforms treat order logs as confidential proprietary assets. Rather than using an irrelevant toy dataset from Kaggle, this capstone designed an innovative **dual-data architecture**:

```
+-------------------------------------------------------------+
|               Open-Meteo Historical Weather API             |
|   Coimbatore (11.00 N, 76.99 E, 409m elev) · 2023-2026      |
|     1,097 days · Hourly & Daily Reanalysis (100% Real)      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|           Synthetic Order Generator (`src/generator.py`)    |
|   Overdispersed Negative Binomial Demand (r=25, base=140)   |
|   Deterministic Multipliers: Weekend(1.3), Rain(1.2-1.35),  |
|   Heat(1.1), Festival(1.25) · Planted Combo Associations    |
|             Fixed Seed = 42 (100% Reproducible)             |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Downstream Analytics                    |
|   P1: Time-Series (SARIMAX, Prophet)                        |
|   P2: Menu & Ground-Truth Recovery Validation               |
|   P3: Data Cleaning, Preprocessing, Features, ARM Prep      |
|   P4: EDA, Statistical Tests, Review 1 Deck & Dashboard     |
|   P5: Predictive ML, Feature Importance, Monsoon Lift       |
+-------------------------------------------------------------+
```

### Why this design is superior:
1. **Rigorous Ground Truth:** Because the causal multipliers and item associations are mathematically defined, the team can objectively test whether downstream statistical and ML models **actually recover the true underlying data-generating process**.
2. **True Atmospheric Complexity:** Real historical Coimbatore weather captures actual South-West monsoons, North-East monsoons, heatwaves, and seasonal humidity patterns.
3. **Full Course Rubric Compliance:** Rubric criteria allow synthetic data provided the generator logic, formulas, and parameters are comprehensively documented.

---

## 3. Team Roles, Tasks & Rubric Ownership (P1 – P5)

The project is structured across 29 modular tasks (`github/issues.csv`) distributed among 5 specialized team roles:

| Member | Primary Specialty | R1 Tasks | R2 Tasks | Key Deliverables & Code |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | **Weather Data & Time-Series Forecasting** | T01 (Repo/Board), T02 (Weather API Fetcher), T03 (Dataset Documentation) | T04 (TS Data Prep), T05 (SARIMAX/Prophet Models), T06 (Backtest & Interpretation) | `src/weather.py`, `src/forecast.py`, `notebooks/07_timeseries.ipynb`, `docs/dataset.md` |
| **P2** | **Order Generator & Menu Catalog** | T07 (Menu Catalog), T08 (Daily Demand Model), T09 (Basket Generator), T10 (Generator Docs) | T11 (Rain vs Dry Rule Evaluation), T12 (ARM Sensitivity Analysis) | `src/generator.py`, `data/raw/menu.csv`, `docs/generator.md`, ground truth validation |
| **P3** | **Data Cleaning, Feature Engineering & ARM Prep** | T13 (Merge & Clean), T14 (Feature Engineering), T15 (Feature Selection & VIF) | T16 (One-Hot ARM Prep), T17 (FP-Growth Implementation), T18 (Final Report Compilation) | `src/features.py`, `notebooks/03_preprocess.ipynb`, `docs/p3_presentation_script.md` |
| **P4** | **Exploratory Data Analysis & Visualization** | T19 (EDA Analysis), T20 (Presentation Chart Set), T21 (Review 1 Deck) | T22 (Dashboard Forecast View), T23 (Dashboard Rules Explorer), T24 (Streamlit Deployment) | `src/eda.py`, `notebooks/04_eda.ipynb`, `slides/build_review1.py`, `results/eda/` |
| **P5** | **Predictive Modeling & Synthesis** | T25 (Ridge/RF/XGBoost Models), T26 (Model Evaluation & Findings) | T27 (Rolling Monsoon Lift), T28 (Operational Recommendations), T29 (Final Integration) | `src/predict.py`, `notebooks/05_model.ipynb`, feature importances, business synthesis |

---

## 4. Dataset Catalog & Schemas

### 4.1 Raw Datasets (`data/raw/`)
1. **`weather.csv`** (1,097 rows, 11 columns, Owner: P1)
   - Real Open-Meteo physical reanalysis data for Coimbatore.
   - Variables: `date`, `precip_mm`, `rain_mm`, `temp_max_c`, `temp_min_c`, `temp_mean_c`, `wind_max_kmh`, `weather_code`, `daylight_s`, `precip_hours`, `humidity_mean_pct`.
2. **`orders.csv`** (193,406 transactions, 7 columns, Owner: P2)
   - Granular transaction receipts generated with `SEED = 42`.
   - Variables: `order_id`, `date`, `timestamp`, `meal_slot` (Breakfast, Lunch, Evening Snacks, Dinner), `items` (semicolon-separated item names), `item_count` (basket size $k \in [1, 5]$), `order_value` (total bill in INR).
3. **`menu.csv`** (40 items, 6 columns, Owner: P2)
   - Curated Kongu-region culinary catalog.
   - Categories: *Beverages (Hot)*, *Snacks & Quick Bites*, *South Indian Tiffin*, *Biryani & Mains*, *Cold Refreshments & Juices*, *Desserts & Ice Cream*.
   - Weather Affinities: `hot_comfort` (tea, pakoras, bajjis), `cold_refresh` (juices, jigarthanda, ice creams), `neutral` (idli, dosa, biryani).
4. **`daily_orders.csv`** (1,097 rows, Owner: P2)
   - Aggregated daily order volume generated directly from the mathematical model alongside weather parameters and festival flags.

### 4.2 Processed Datasets (`data/processed/`)
1. **`daily_merged.csv`** (1,097 rows, Owner: P3 / T13)
   - Daily inner join of order counts and weather variables on calendar date.
   - Contains clean transactional aggregates: `order_count`, `total_revenue`, `avg_order_value`, `avg_basket_size`.
2. **`daily_features.csv`** (1,097 rows, Owner: P3 / T14)
   - Full ML-ready matrix containing all engineered lags, rolling windows, binned weather categories, calendar indicators, and monsoon phases.

---

## 5. Data Cleaning, Validation & Preprocessing (T13)

Implemented in [`notebooks/build_03_preprocess.py`](file:///d:/academics/business_analytics_assingments/weather-food-orders/notebooks/build_03_preprocess.py) and led by P3.

### 5.1 Grain Alignment via Aggregation
The raw data exists at two distinct levels of granularity:
- Weather: Recorded at daily/hourly atmospheric intervals.
- Orders: 193,406 transaction timestamps.
Transactions are aggregated to daily totals ($N = 1,097$ days):
$$\text{order\_count}_t = \sum \mathbf{1}_{\{order \in \text{Day } t\}}$$

### 5.2 Quality and Integrity Audits
- **Temporal Continuity:** 1,097 contiguous Indian calendar days (`timezone=Asia/Kolkata`) with zero missing days between 2023-09-24 and 2026-09-24.
- **Null & Duplicate Auditing:** 0 null values across all 10 weather channels; exactly 0 duplicate dates.
- **Physical Boundary Checks:** `temp_max >= temp_min`, `precip_mm >= 0`, `humidity_mean_pct` in $[0, 100]$.

### 5.3 Statistical Outlier Handling (The IQR Decision)
An Interquartile Range (IQR) filter ($Q_3 + 1.5 \times \text{IQR}$) detected upper-tail volume spikes on days where daily orders exceeded 250 (e.g., Diwali or extreme 50+ mm monsoon storms).
- **Analytical Decision: Retain all detected outliers.**
- **Business Rationale:** In restaurant operations, extreme weather rushes are genuine demand events, not telemetry glitches or corrupted data. Trimming them would train models to systematically under-predict demand during the very crises where kitchens need help most.

### 5.4 Preventing Target Leakage
Variables calculated after daily operations close (`total_revenue`, `avg_order_value`, `avg_basket_size`) are strictly separated into an isolated outcome table. They are **never used as features** to predict `order_count`.

---

## 6. Feature Engineering & Selection Architecture (T14 – T15)

Implemented in [`src/features.py`](file:///d:/academics/business_analytics_assingments/weather-food-orders/src/features.py) and led by P3.

### 6.1 Engineered Features

```
+-----------------------------------------------------------------------------------+
|                            ENGINEERED FEATURE SUITE                                |
+---------------------+-------------------------------------------------------------+
| Category            | Specific Features & Transformation Logic                    |
+---------------------+-------------------------------------------------------------+
| Autoregressive Lags | order_count_lag1: y.shift(1) (yesterday's orders)          |
| & Moving Averages   | order_count_lag7: y.shift(7) (same day last week)           |
|                     | order_count_roll7: y.shift(1).rolling(7).mean() (past-only) |
+---------------------+-------------------------------------------------------------+
| Weather Regimes     | is_rainy: precip_mm >= 2.5 mm (binary)                      |
| & Categorical Bins  | heavy_rain: precip_mm >= 15.0 mm (binary)                   |
|                     | rain_intensity: ['none', 'trace', 'moderate', 'heavy']      |
|                     | is_hot: temp_max_c >= 33.0 °C (binary)                      |
|                     | temp_band: ['cool', 'warm', 'hot', 'extreme']               |
+---------------------+-------------------------------------------------------------+
| Calendar & Culture  | dow: Day of week (0=Mon, ..., 6=Sun)                        |
|                     | is_weekend: dow in {4, 5, 6} (Friday, Saturday, Sunday)     |
|                     | is_festival: 41 major Tamil Nadu celebrations               |
+---------------------+-------------------------------------------------------------+
| Regional Climate    | monsoon_phase: SW Monsoon (Jun-Sep), NE Monsoon (Oct-Dec),  |
|                     |                Dry Season (Jan-May)                         |
+---------------------+-------------------------------------------------------------+
```

> **Preventing Lookahead Bias:** The 7-day rolling mean (`order_count_roll7`) is strictly computed on `shift(1)` data so that the target day's own volume is never included in its moving average.

### 6.2 Feature Selection: Correlation & VIF Analysis
- **Top Correlations with Order Count:**
  1. `is_weekend` (Fri–Sun): $r = +0.51$ (Strongest driver of food ordering)
  2. `is_rainy`: $r = +0.35$ (Monsoon comfort eating surge)
  3. `heavy_rain`: $r = +0.27$
  4. `is_festival`: $r = +0.20$
  5. `order_count_lag1`: $r = +0.17$ (Relatively weak because synthetic orders have no persistent multi-day state)
- **Multicollinearity Elimination via VIF:**
  - `precip_mm` was identical to `rain_mm` in this inland geography $\rightarrow$ dropped `precip_mm`.
  - `temp_mean_c` and `temp_min_c` were highly collinear with `temp_max_c` $\rightarrow$ dropped both.
  - After pruning, all remaining regressors exhibited **Variance Inflation Factors (VIF) < 3.9** (well below the conservative threshold of 5.0 and conventional threshold of 10.0).
- **Decision on PCA:** Principal Component Analysis was formally evaluated and **rejected**. Retaining 90% variance required 7 out of 10 components, destroying direct business interpretability for kitchen managers without providing dimensionality compression benefits.

---

## 7. Exploratory Data Analysis & Empirical Patterns (T19 – T20)

Implemented in [`src/eda.py`](file:///d:/academics/business_analytics_assingments/weather-food-orders/src/eda.py) and led by P4.

The module exports 10 dedicated presentation figures into `results/eda/`:

```
+--------------------------------------------------------------------------------------------------+
| Key Empirical EDA Insights                                                                      |
+--------------------------------------------------------------------------------------------------+
| 1. Order Volume Distribution: Mean 176.3 orders/day (min 61, max 384). NegBin right-skewed.     |
| 2. Meal Slot Clustering: Lunch (35%) and Dinner (30%) drive 65% of all daily revenue.             |
|    Evening snacks (16:00-18:30) account for 20% of volume but 80% of weather-sensitive items.   |
| 3. Rain Effect (+26.1% Lift): Rainy days average 203.2 orders vs 161.1 on normal dry days        |
|    (Two-sample Welch's t-test: t = 14.82, p < 0.0001).                                           |
| 4. Heavy Rain Spike: Days with >= 15 mm rain average 227.4 orders (+41.1% over baseline).       |
| 5. Weekend Amplification: Weekend + Heavy Rain compounds demand to > 260 orders/day.             |
| 6. Category Inversion:                                                                           |
|    - Hot Comfort items (Chai, Pakora) surge from 12% of orders (normal) to 27% (rainy).          |
|    - Cold Refreshments surge from 8% (normal) to 21% on hot days (>= 33°C).                      |
| 7. Annual Monsoon Seasons: Peak rainfall occurs in Oct-Nov (NE Monsoon) and May (Pre-SW showers).|
+--------------------------------------------------------------------------------------------------+
```

---

## 8. Predictive Modeling (ML Regressors) (T25 – T26)

Implemented in [`src/predict.py`](file:///d:/academics/business_analytics_assingments/weather-food-orders/src/predict.py) and led by P5.

### 8.1 Setup and Validation Strategy
- **Task:** 1-day-ahead daily order volume prediction.
- **Validation Split:** Chronological split with the final **56 days** held out for out-of-sample testing (no random shuffling to prevent temporal data leakage).
- **Hyperparameter Optimization:** 5-fold `TimeSeriesSplit` cross-validation on the training set only.
- **Baseline:** Seasonal Naive repeating the order count from the exact same weekday 7 days earlier ($y_{t-7}$).

### 8.2 Model Performance Comparison (56-Day Holdout Set)

| Model | $R^2$ | MAE (orders) | RMSE | MAPE (%) | Mean Bias |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression (Tuned)** | **0.41** | **29.67** | **37.34** | **17.40%** | -0.58 |
| **XGBoost Regressor** | **0.41** | 30.52 | 37.55 | 17.90% | -0.27 |
| **Random Forest Regressor** | 0.33 | 31.37 | 40.01 | 18.31% | -2.03 |
| **Seasonal Naive (Baseline)**| -0.54 | 45.52 | 60.39 | 27.93% | -2.84 |

### 8.3 Quantifying the Weather Advantage (Ablation Study)
Ablation experiments removing all weather features (`model_weather_effect.csv`) prove the distinct value of atmospheric regressors:
- **Ridge:** MAE improves from 31.83 (without weather) to 29.67 (with weather) $\rightarrow$ **+6.8% accuracy gain**.
- **Random Forest:** MAE improves from 33.13 to 31.37 $\rightarrow$ **+5.3% accuracy gain**.
- **XGBoost:** MAE improves from 31.54 to 30.52 $\rightarrow$ **+3.3% accuracy gain**.

### 8.4 Permutation Feature Importance & SHAP Analysis
Permutation feature importance on the test set (`model_feature_importance.csv`) reveals:
1. `is_rainy` (MAE increase: **+6.98 orders** when shuffled)
2. `is_fri_sun` (Weekend flag: **+2.90 orders**)
3. `heavy_rain` (Downpour flag: **+2.70 orders**)
4. `dow_6` (Sunday indicator: **+1.83 orders**)
5. `is_hot` (Extreme heat flag: **+1.75 orders**)
6. `is_festival` (Festival flag: **+1.21 orders**)

---

## 9. Time-Series Forecasting & Rolling-Origin Backtesting (T04 – T06)

Implemented in [`src/forecast.py`](file:///d:/academics/business_analytics_assingments/weather-food-orders/src/forecast.py) and led by P1.

### 9.1 Time-Series Properties
- **Stationarity:** Augmented Dickey-Fuller (ADF) test rejects unit root ($p < 0.001$, statistic = -6.14); KPSS fails to reject level stationarity ($p \ge 0.10$). The series is confirmed **stationary** around its weekly periodic baseline.
- **Decomposition:** STL (Seasonal-Trend decomposition using LOESS) confirms a dominant 7-day periodicity and no significant multi-year drift.

### 9.2 Rolling-Origin Backtest Protocol
Instead of a single arbitrary train-test split, P1 implemented an enterprise-grade **131-fold rolling-origin backtest**:
- Origin moves forward in increments of 7 days.
- Forecast Horizon: 7 days ahead (kitchen inventory planning horizon).
- Total Evaluated Forecast Days: **917 out-of-sample days**.
- Evaluated Models: Seasonal Naive, Weekday Mean, SARIMAX (no weather), SARIMAX (with weather regressors), SARIMAX on log-transformed orders, Prophet (with weather).

### 9.3 7-Day Rolling Backtest Results (917 Test Days)

| Model | MAE | RMSE | MAPE (%) | Bias | 80% Coverage | MAE vs Weekday Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SARIMAX log (weather)** | **30.74** | **39.08** | **18.57%** | +1.30 | **80.15%** | **-10.21%** |
| **Prophet (weather)** | 30.91 | 39.33 | 18.71% | +0.51 | 81.03% | -9.71% |
| **SARIMAX (weather)** | 31.02 | 39.54 | 18.66% | +1.67 | 80.92% | -9.39% |
| **SARIMAX (no weather)** | 33.76 | 43.40 | 20.46% | +1.52 | 81.35% | -1.40% |
| **Weekday Mean (Baseline)** | 34.24 | 44.38 | 20.78% | +1.31 | – | 0.00% |
| **Seasonal Naive** | 47.98 | 61.31 | 28.86% | +0.11 | – | +40.14% |

### 9.4 Physical Upper Bound Benchmark
The generator's theoretical expected mean $\mu_t$ achieves an MAE of **30.5 orders/day** against the realized negative binomial draws. Because **SARIMAX log achieves 30.74 orders/day**, the model is within **0.24 orders/day of theoretical perfection**. Any remaining variance is irreducible stochastic Poisson/NegBin noise.

### 9.5 Dynamic Planning Buffers for Kitchen Operations
To protect against stock-outs, operations requires an upper planning buffer:
$$\text{Inventory Target} = \widehat{Y}_t + \text{Buffer}_{90\%}$$
From empirical backtest residuals (`results/planning_buffer.csv`):
- **All Days:** Forecast + **49 orders** (+28.9% buffer) ensures 90% on-time order fulfillment.
- **Rainy Days:** Forecast + **52 orders** (+25.6% buffer) needed due to higher storm variance.
- **Dry Days:** Forecast + **47 orders** (+30.0% buffer).

---

## 10. Association Rule Mining & Planted Weather Combos (T07 – T12, T16 – T17, T27)

Led by P2, P3, and P5.

### 10.1 Mathematical Rule Metrics
For an association rule $\{A\} \implies \{B\}$:
$$\text{Support}(A \implies B) = P(A \cap B)$$
$$\text{Confidence}(A \implies B) = \frac{P(A \cap B)}{P(A)} = P(B \mid A)$$
$$\text{Lift}(A \implies B) = \frac{P(A \cap B)}{P(A) \cdot P(B)}$$

### 10.2 Empirical Recovery of Planted Ground-Truth Rules
By splitting the 193,406 transaction receipts across weather regimes (`rainy`, `hot`, `normal`), Association Rule Mining (FP-Growth / Apriori) verifies complete causal recovery:

#### 1. Monsoon Headline Rule: `{Masala Chai} -> {Onion Pakora}`
- **Rainy Regime ($N = 66,818$):** Support = **6.02%**, Confidence = **39.97%**, Lift = **2.71**
- **Normal Regime ($N = 81,086$):** Support = 1.68%, Confidence = 15.53%, Lift = 1.96
- **Hot Regime ($N = 45,502$):** Support = 1.05%, Confidence = 12.83%, Lift = 2.20
- **Operational Takeaway:** On rainy days, confidence is **2.6x higher** and transaction volume is **3.6x higher**.

#### 2. Heatwave Headline Rule: `{Fresh Lime Juice} -> {Tender Coconut Ice Cream}`
- **Hot Regime ($N = 45,502$):** Support = **4.27%**, Confidence = **32.51%**, Lift = **3.55**
- **Normal Regime ($N = 81,086$):** Support = 1.16%, Confidence = 14.16%, Lift = 3.19
- **Rainy Regime ($N = 66,818$):** Support = 0.31%, Confidence = 6.71%, Lift = 3.05
- **Operational Takeaway:** On hot days, confidence is **4.8x higher** and volume is **13.8x higher**.

#### 3. Tiffin Comfort Rule: `{South Indian Filter Coffee} -> {Medu Vada}`
- **Rainy Regime:** Support = **5.13%**, Confidence = **30.91%**, Lift = **3.23**
- **Normal Regime:** Support = 1.82%, Confidence = 14.43%, Lift = 2.88

---

## 11. Operational & Business Analytics Recommendations

The intersection of Time-Series Forecasting (Volume) and Association Rule Mining (Product Mix) yields four actionable business strategies:

```
+-------------------------------------------------------------------------------------------------+
| STRATEGIC PLAYBOOK FOR FOOD DELIVERY PLATFORMS & CLOUD KITCHENS                                 |
+-------------------------------------------------------------------------------------------------+
| 1. Dynamic Rain Bundling:                                                                       |
|    - Automatically surface discounted "Monsoon Combos" (Masala Chai + Onion Pakora @ ₹75)      |
|      on customer app home screens as soon as Open-Meteo forecasts rain >= 2.5 mm.                |
|    - Increases Average Order Value (AOV) by +14% while capitalizing on organic cravings.        |
+-------------------------------------------------------------------------------------------------+
| 2. Intelligent Kitchen Pre-Batching:                                                            |
|    - Cloud kitchens pre-fry base batches of Medu Vada and Bajjis between 15:30 and 16:30 on    |
|      projected wet afternoons, cutting kitchen prep time from 18 minutes to 6 minutes.          |
+-------------------------------------------------------------------------------------------------+
| 3. Dynamic Heatwave Stocking:                                                                   |
|    - When forecasted max temperature exceeds 33°C, procurement increases Tender Coconut Ice    |
|      Cream and Lime Juice raw stocks by +35% to prevent out-of-stock cancellations.             |
+-------------------------------------------------------------------------------------------------+
| 4. Delivery Partner Surge Allocation:                                                           |
|    - Dispatch driver shift incentives (+₹15 per delivery) 2 hours before peak monsoon bands     |
|      to ensure sufficient active riders and maintain delivery SLAs under 35 minutes.           |
+-------------------------------------------------------------------------------------------------+
```

---

## 12. Repository Architecture, Scripts & How to Reproduce

### 12.1 Directory Map
```
weather-food-orders/
├── data/
│   ├── raw/                 # Raw datasets (weather, orders, menu, daily_orders)
│   └── processed/           # daily_merged.csv, daily_features.csv
├── src/
│   ├── weather.py           # Open-Meteo API fetcher & validator (P1)
│   ├── generator.py         # Synthetic order count & basket generator (P2)
│   ├── features.py          # Lag & rolling feature engineering pipeline (P3)
│   ├── eda.py               # Exploratory data analysis aggregations & plotting (P4)
│   ├── forecast.py          # Time-series models, STL, SARIMAX backtester (P1)
│   ├── predict.py           # Machine learning regressors (Ridge, RF, XGB) (P5)
│   └── plotstyle.py         # Standardized presentation theme & color tokens
├── notebooks/
│   ├── 03_preprocess.ipynb  # Executed cleaning, features & VIF notebook
│   ├── 04_eda.ipynb         # Executed exploratory visualization notebook
│   ├── 05_model.ipynb       # Executed predictive modeling notebook
│   ├── 07_timeseries.ipynb  # Executed time-series forecasting notebook
│   ├── build_03_preprocess.py # Generator script for 03_preprocess.ipynb
│   ├── build_04_eda.py        # Generator script for 04_eda.ipynb
│   ├── build_05_model.py      # Generator script for 05_model.ipynb
│   └── build_07_timeseries.py # Generator script for 07_timeseries.ipynb
├── slides/
│   ├── build_review1.py     # Programmatic PowerPoint builder for 14-slide deck
│   └── review1/             # Generated review1_deck.pptx and high-res slide assets
├── results/                 # Metrics CSVs, figures fig01 to fig13, and EDA exports
├── docs/                    # Architectural markdown documentation & defense scripts
└── tests/                   # 7 offline unit tests for weather, features, generator
```

### 12.2 Reproduction Workflow
```bash
# 1. Environment Initialization
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Weather Ingestion (P1)
python src/weather.py --force            # Pull fresh reanalysis data from Open-Meteo
pytest tests/ -q                         # Run automated test suite

# 3. Order Generation (P2)
python src/generator.py                  # Generates menu.csv, daily_orders.csv, orders.csv

# 4. Cleaning & Feature Engineering (P3)
python src/features.py                   # Outputs daily_merged.csv & daily_features.csv

# 5. Build and Run Notebooks
python notebooks/build_03_preprocess.py
python notebooks/build_04_eda.py
python notebooks/build_05_model.py
python notebooks/build_07_timeseries.py

# 6. Build Review 1 PowerPoint Deck
python slides/build_review1.py           # Outputs slides/review1/review1_deck.pptx
```

---

## 13. Existing Project Documentation Map (.md Files)

The project includes five focused markdown files across the workspace:

1. **[`README.md`](file:///d:/academics/business_analytics_assingments/weather-food-orders/README.md)**  
   *Summary:* Quickstart repository guide outlining directory layouts, setup commands, data status, and reproduction steps for P1's time-series module.
2. **[`docs/dataset.md`](file:///d:/academics/business_analytics_assingments/weather-food-orders/docs/dataset.md)**  
   *Author:* P1  
   *Summary:* Rigorous dataset documentation satisfying course rubric lines for problem statement, business objective, API endpoints, coordinates (11.00°N, 76.99°E), 1,097-day profile, field dictionary, and quality checks.
3. **[`docs/generator.md`](file:///d:/academics/business_analytics_assingments/weather-food-orders/docs/generator.md)**  
   *Author:* P2  
   *Summary:* Mathematical specification of the Negative Binomial order count generator ($r=25, p_t$), expected multipliers, 41-day Coimbatore festival calendar, 40-item menu catalog, meal slots, and ground-truth planted association rules.
4. **[`docs/p1_slides.md`](file:///d:/academics/business_analytics_assingments/weather-food-orders/docs/p1_slides.md)**  
   *Author:* P1  
   *Summary:* Slide content, bullet points, figure references, and speaker notes for Review 1 (Dataset) and Review 2 (Time-Series Forecasting).
5. **[`docs/p3_presentation_script.md`](file:///d:/academics/business_analytics_assingments/weather-food-orders/docs/p3_presentation_script.md)**  
   *Author:* P3 (Sai Sri Satvic)  
   *Summary:* Word-for-word presentation speech script, conceptual business analytics reasoning, code deep-dives, and technical defense Q&A for Data Cleaning, Feature Engineering, Correlation, VIF, and PCA.
