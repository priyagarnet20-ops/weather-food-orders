# Dataset Documentation

**Project:** Weather-Driven Food Order Patterns (Surabaya, Indonesia) · BA Capstone, Team of 5  
**Author:** P1 (Weather Data & Dataset Documentation) · **Rubric line:** *Problem statement, business objective and dataset description*  
**Tasks:** T02 (Open-Meteo Weather Fetcher for Surabaya), T03 (Dataset Documentation)

---

## 1. Problem Statement

Food and beverage (F&B) demand in Surabaya, Indonesia is subject to both temporal variations (day-of-week, meal hours) and tropical weather dynamics (extreme heat vs. tropical monsoon downpours).
Physical stores and kitchen operations that plan only around an average day experience staffing and prep misallocations: over-staffed on quiet rainy afternoons where walk-in foot traffic slows, yet unprepared for sudden demand shifts toward warm comfort items and hot specialty beverages.
This project analyzes how much of that demand variation local weather explains and determines whether meteorological forecasts can be leveraged to optimize daily order planning, kitchen prep, and inventory management.

## 2. Business Objectives

1. **Review 1:** Predict daily order volume from weather, calendar, and historical lag features, and evaluate which meteorological variables drive operational shifts.
2. **Review 2, Method 1 (Association Rule Mining):** Mine transaction itemsets to identify product affinities, evaluating whether purchasing behavior shifts between dry, hot, and monsoon rain conditions (e.g., hot coffee and warm pastries vs. iced refreshments).
3. **Review 2, Method 2 (Time Series Forecasting):** Forecast multi-step order demand to support kitchen inventory and staff scheduling, quantifying the empirical error reduction enabled by weather signals.
4. **Actionable Operations:** Translate analytical findings into concrete recommendations: rainy-day combo promotions, dynamic safety buffers, and weather-conditioned replenishment.

## 3. Data Sources and Provenance

| Part | Source | Type | Details |
|---|---|---|---|
| **Weather Data** (`weather.csv`, `weather_hourly.csv`) | Open-Meteo Historical Weather API (`open-meteo.com`) | **Real** (ERA5 reanalysis) | Surabaya, Indonesia (-7.2575°N, 112.7521°E), 3-year history (2023–2026), hourly & daily |
| **Menu Catalog** (`menu.csv`) | Extracted from MSME Product Catalog (`Products.csv`) | **Real** | 96 items across 6 categories (IDR prices, categorized, weather affinities) |
| **Food Orders** (`orders.csv`, `order_items.csv`, `daily_orders.csv`) | Real Point-of-Sale MSME POS Transactions (`Transactions.csv`) | **Real** POS Transactions | 53,820 orders across 272 days (Jan–Sep 2025), 3 outlets combined |

### 3.1 Dataset Attribution & Provenance
The food order dataset is an authentic transactional dataset from a local F&B Micro-Small-Medium Enterprise (MSME) in Surabaya, Indonesia, curated by researchers from Universitas Ciputra Surabaya:
- **Repository:** Mendeley Data (`kcgf45y24m/2`)
- **Published Paper:** *Anonymous transactional dataset in a local food and beverages (F&B) micro-small-medium enterprise (MSME) for recommender systems*, *Data in Brief*, 2026. DOI: `10.1016/j.dib.2026.112519`
- **Raw Files:** `data/raw/Products.csv` (item catalog and prices), `data/raw/Transactions.csv` (53,820 order records across 3 retail outlets).

### 3.2 Weather Collection Method (T02)

* **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
* **Target Coordinates:** Latitude -7.2575°N, Longitude 112.7521°E (Surabaya, East Java, Indonesia).
* **Grid Cell Matched:** -7.26°N, 112.75°E at 11.0 m elevation (see `data/raw/weather_meta.json`).
* **Observation Window:** 2023-09-24 to 2026-09-24 inclusive (1,097 calendar days; 3 years).
* **Time Zone:** `Asia/Jakarta` (WIB, UTC+7), ensuring daily summaries align with Indonesian calendar days.
* **Daily Variables:** `precipitation_sum` (mm), `rain_sum` (mm), `temperature_2m_max` (°C), `temperature_2m_min` (°C), `temperature_2m_mean` (°C), `wind_speed_10m_max` (km/h), `weather_code` (WMO code), `daylight_duration` (s).
* **Hourly Variables:** `relative_humidity_2m` (%), `temperature_2m` (°C), `precipitation` (mm). Daily mean humidity and precipitation duration are computed from the 24 hourly records per day.
* **Script:** `python src/weather.py --force` (cached and rerunnable; validated via `tests/test_weather.py`).

### 3.3 POS Preprocessing & Aggregation (T07, T08, T09)

1. **Multi-Outlet Combination:** The MSME operates three outlets (`SHOP001`, `SHOP002`, `SHOP003`). Because the capstone models weather-driven demand for a single restaurant entity, transactions across all 3 outlets are merged into a single chronological stream (53,820 orders).
2. **Item Standardization:** Semicolon-delimited basket strings (e.g., `Friendly Coffee (Ice); Donat Polos`) are standardized and parsed. All 94 unique transaction item strings matched the official 96-item menu catalog.
3. **Price Imputation:** Promotional zero-amount transactions (1,733 records) were imputed using official catalog prices, ensuring 100% positive order values.
4. **Operational Meal Windows:** Each transaction timestamp is assigned to an operational meal window:
   - **Breakfast:** 07:00–11:00 (20.8% of orders)
   - **Lunch:** 11:00–15:00 (23.7% of orders)
   - **Snacks:** 15:00–18:00 (19.2% of orders)
   - **Dinner:** 18:00–23:00 (36.4% of orders)

---

## 4. Data Files & Directory Layout

| File | Rows | Description | Owner |
|---|---|---|---|
| `data/raw/Products.csv` | 96 | Raw MSME item catalog with Indonesian Rupiah prices | External / P2 |
| `data/raw/Transactions.csv` | 53,820 | Raw POS transactions across 3 outlets | External / P2 |
| `data/raw/menu.csv` | 96 | Cleaned menu catalog with weather affinity tags | P2 |
| `data/raw/orders.csv` | 53,820 | Preprocessed transaction records with meal slots and baskets | P2 |
| `data/raw/order_items.csv` | 61,973 | Normalized order-item pairs for category & ARM analysis | P2 |
| `data/raw/daily_orders.csv` | 272 | Daily order counts, revenue, and mean basket values | P2 |
| `data/raw/weather.csv` | 1,097 | Surabaya daily weather metrics from Open-Meteo | P1 |
| `data/raw/weather_hourly.csv` | 26,328 | Surabaya hourly weather variables | P1 |
| `data/raw/weather_meta.json` | – | API query parameters, grid cell, and validation report | P1 |
| `data/processed/daily_merged.csv` | 272 | Joined POS order counts and weather variables | P3 |
| `data/processed/daily_features.csv` | 272 | Curated 28-feature dataset for predictive modeling | P3 |

---

## 5. Field Dictionary

### 5.1 Weather Metrics (`weather.csv`)

| Column | Unit | Description | Open-Meteo Variable |
|---|---|---|---|
| `date` | YYYY-MM-DD | Calendar day (Asia/Jakarta) | `time` |
| `precip_mm` | mm | Total daily precipitation | `precipitation_sum` |
| `rain_mm` | mm | Daily liquid rain | `rain_sum` |
| `temp_max_c` | °C | Maximum daily 2 m air temperature | `temperature_2m_max` |
| `temp_min_c` | °C | Minimum daily 2 m air temperature | `temperature_2m_min` |
| `temp_mean_c` | °C | Mean daily 2 m air temperature | `temperature_2m_mean` |
| `wind_max_kmh` | km/h | Maximum 10 m wind gust speed | `wind_speed_10m_max` |
| `weather_code` | WMO code | Most severe weather condition (WMO standard) | `weather_code` |
| `daylight_s` | seconds | Total daylight duration | `daylight_duration` |
| `precip_hours` | hours | Hours with precipitation > 0 mm | derived from hourly precipitation |
| `humidity_mean_pct` | % | Mean relative humidity | mean of hourly `relative_humidity_2m` |

### 5.2 Regime & Calendar Conventions

| Flag | Logic / Rule | Meaning |
|---|---|---|
| `is_rainy` | `precip_mm >= 2.5` | Meaningful tropical rain day |
| `heavy_rain` | `precip_mm >= 15.0` | Heavy tropical downpour |
| `is_hot` | `temp_max_c >= 33.0` | Hot tropical afternoon |
| `regime` | `'rainy'` if rain $\ge 2.5\text{mm}$, else `'hot'` if temp $\ge 33^\circ\text{C}$, else `'normal'` | Dominant weather regime |
| `monsoon_phase` | `'NW_Monsoon'` (Nov–Apr; wet) vs `'SE_Monsoon'` (May–Oct; dry) | Surabaya tropical monsoon season |
| `is_weekend` | `dow in [4, 5, 6]` (Friday, Saturday, Sunday) | Weekend leisure / dining window |
| `is_festival` | Indonesian National Holiday or cultural festival | Verified via `holidays.Indonesia()` |

---

## 6. Dataset Profile: Surabaya (2025 Observation Window)

| Measure | Value | Notes |
|---|---|---|
| **Observation Days** | 272 days | Continuous daily coverage: 2025-01-01 to 2025-09-29 |
| **Total POS Transactions** | 53,820 orders | Aggregated across SHOP001, SHOP002, SHOP003 |
| **Mean Daily Orders** | 197.9 orders/day | Standard deviation: 33.0 |
| **Order Volume Range** | 126 – 282 orders/day | Median: 194.0; IQR: 43.2 |
| **Average Order Value** | Rp 25,261 | Median: Rp 23,000 |
| **Precipitation Distribution** | Dry days: 149 (54.8%) \| Rainy days: 123 (45.2%) | Heavy rain days (≥15mm): 30 days |
| **Mean Orders by Weather** | Dry days: 204.5 orders/day \| Rainy days: 189.8 orders/day | Rain suppresses foot traffic by -7.2% ($p < 0.001$) |
| **Meal Slot Split** | Dinner: 36.4% \| Lunch: 23.7% \| Breakfast: 20.8% \| Snacks: 19.2% | Aligns with store operating hours (07:00–23:00 WIB) |

---

## 7. Data Quality Controls

All datasets pass automated validation gates before entering the modeling pipeline (`pytest tests/`):
- **Contiguity:** Contiguous daily date indices with zero missing calendar dates.
- **Null & Duplicate Checks:** Zero duplicate records, zero NaN values across weather and feature tables.
- **Physical Boundaries:** `temp_max >= temp_min` holds on 100% of days; humidity $\in [0, 100]\%$.
- **Catalog Integrity:** 100% of 94 unique transaction item strings match catalog SKUs.
- **Positive Spend:** 100% of orders have positive order value after price imputation.

## 8. Limitations & Scope

- **Single Year POS Coverage:** Transactional data spans 272 days (January–September 2025). Models cannot test inter-annual trend components.
- **Aggregated Outlets:** Three physical outlets are combined to represent total enterprise demand for a unified kitchen/inventory model.
- **Weather Grid Snapping:** Open-Meteo reanalysis operates on an 11 km grid centered at -7.26°N, 112.75°E, representing city-level meteorological conditions rather than localized hyper-local rain sensors.
