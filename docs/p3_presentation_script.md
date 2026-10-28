# Review 1 Presentation Script & Technical Defense (P3)
**Course:** 23CSE452 Business Analytics · Review 1  
**Project:** Weather-Driven Food Order Patterns (Surabaya, Indonesia)  
**Section Owner:** P3 — Sai Sri Satvic (Data Cleaning, Feature Engineering & ARM Preparation)  
**Tasks Covered:** T13 (Merge & Clean), T14 (Feature Engineering), T15 (Feature Selection), T16–T17 (ARM Roadmap)  
**Slide Deck Source:** `slides/review1/review1_deck.pptx` (Slide 6)  
**Associated Code Files:**  
- [`src/features.py`](file:///d:/academics/business_analytics_assingments/weather-food-orders/src/features.py)  
- [`notebooks/build_03_preprocess.py`](file:///d:/academics/business_analytics_assingments/weather-food-orders/notebooks/build_03_preprocess.py)  
- [`notebooks/03_preprocess.ipynb`](file:///d:/academics/business_analytics_assingments/weather-food-orders/notebooks/03_preprocess.ipynb)  
- [`data/processed/daily_merged.csv`](file:///d:/academics/business_analytics_assingments/weather-food-orders/data/processed/daily_merged.csv)  
- [`data/processed/daily_features.csv`](file:///d:/academics/business_analytics_assingments/weather-food-orders/data/processed/daily_features.csv)  
- [`tests/test_features.py`](file:///d:/academics/business_analytics_assingments/weather-food-orders/tests/test_features.py)  

---

## Table of Contents
1. [Overview & Quick Reference Cheat Sheet](#1-overview--quick-reference-cheat-sheet)
2. [Review 1 Deck Layout (Slide 6 Breakdown)](#2-review-1-deck-layout-slide-6-breakdown)
3. [Word-for-Word Spoken Presentation Script (~90–120 Seconds)](#3-word-for-word-spoken-presentation-script-90120-seconds)
4. [Task-by-Task Technical Deep Dive](#4-task-by-task-technical-deep-dive)
   - [4.1 T13: Data Cleaning, Quality Auditing & Aggregation](#41-t13-data-cleaning-quality-auditing--aggregation)
   - [4.2 T14: Feature Engineering & Leakage Prevention](#42-t14-feature-engineering--leakage-prevention)
   - [4.3 T15: Feature Selection, Correlation Analysis & VIF Diagnostics](#43-t15-feature-selection-correlation-analysis--vif-diagnostics)
   - [4.4 T16–T17: Pipeline Reproducibility & ARM Roadmap (Review 2)](#44-t16t17-pipeline-reproducibility--arm-roadmap-review-2)
5. [Code Architecture Walkthrough](#5-code-architecture-walkthrough)
6. [Comprehensive Panel Q&A Defense & Trap Questions](#6-comprehensive-panel-qa-defense--trap-questions)
7. [Master Glossary of Keywords & Statistical Concepts](#7-master-glossary-of-keywords--statistical-concepts)

---

## 1. Overview & Quick Reference Cheat Sheet

| Task | Core Technical Contribution | Key Data Points & Verified Numbers | Critical In-Flow Buzzwords |
| :--- | :--- | :--- | :--- |
| **T13: Cleaning & Merging** | Aggregated multi-outlet transactions to daily grain; joined with Surabaya weather; zero missing data. | **53,820 orders** across 3 outlets $\rightarrow$ **272 continuous days** (Jan 2 – Sep 30, 2025); 0 nulls; 0 duplicates. Currency: **IDR (Rp)**. | **Grain Alignment**, **Inner Join**, **IQR (Interquartile Range)**: Weather anomalies retained because tropical rain downpours reflect true operational reality. |
| **T14: Feature Engineering** | Past-only lags prevent lookahead bias; rain and temperature thresholds; Indonesian calendar; monsoon phase. | Lags (1, 7) and 7-day rolling mean; Rain ($\ge 2.5$ mm, $\ge 15.0$ mm); Heat ($\ge 33^\circ\text{C}$); **15 Indonesian public holidays**; **NW Monsoon (Nov–Apr)** vs **Dry Season (May–Oct)**. | **Data Leakage Prevention**: Rolling window calculated strictly after `.shift(1)`. Left-closed bins $[a, b)$ ensure boundary integrity. |
| **T15: Feature Selection** | Multicollinearity eliminated; dropped 3 redundant columns; all VIF $< 10$; PCA rejected for interpretability. | Dropped `precip_mm`, `temp_mean_c`, `temp_min_c`. **All VIFs $< 10$** (Max VIF: **9.12** on `roll7`, lag-7: **4.34**, lag-1: **4.28**). | **VIF (Variance Inflation Factor)**, **Autocorrelation** ($r = +0.769$), **Walk-in Friction** ($r = -0.213$ for rain), **Interpretability over PCA**. |
| **T16–T17: Review 2 Roadmap** | Transaction-level itemset preprocessing for Association Rule Mining (ARM). | Semicolon-delimited basket strings prepared across 96 menu catalog items; ready for FP-Growth and Apriori. | **Itemsets**, **TransactionEncoder**, **Weather Regime Segmentation**. |

---

## 2. Review 1 Deck Layout (Slide 6 Breakdown)

```
=============================================================================================================================
                          P3 · Data Cleaning, Feature Engineering & Pipeline Architecture
                 Weather-orders merge · lag features · Indonesian calendar · NW monsoon phases
                                                   P3 · T13, T14, T15, T16, T17
-----------------------------------------------------------------------------------------------------------------------------
  [LEFT BLOCK: Preprocessing & Feature Engineering]           [RIGHT BLOCK: Pipeline Architecture & ARM Preparation]
  
  DATA CLEANING & MERGING (T13):                              PIPELINE REPRODUCIBILITY & TEST SUITE:
  • Joined daily POS orders to Surabaya weather on date       • Automated unit test suite passing in tests/ directory
    (272 days, Jan–Sep 2025)                                  • Modular architecture in src/ (weather, preprocess, features,
  • Parsed timestamps, verified 0 missing dates, zero           eda, predict)
    duplicates, zero IQR errors                               • Notebooks execute cleanly end-to-end (03_preprocess,
  • Clean reproducible pipeline saved to                        04_eda, 05_model)
    data/processed/daily_merged.csv                           
                                                              ARM PREPARATION (T16, T17 – REVIEW 2 ROADMAP):
  FEATURE ENGINEERING (T14):                                  • Order transactions preprocessed into basket itemsets
  • Lags (1, 7) and 7-day rolling mean (strictly              • Ready for TransactionEncoder and weather regime segmentation
    chronological, zero data leakage)                         • FP-Growth and Apriori implementations scheduled for
  • Rain flag (>=2.5mm) + Heavy rain flag (>=15mm) +            Review 2
    Intensity tiers
  • Temperature bands: Cool / Warm / Hot (>=33°C) / Extreme
  • Indonesian Public Holidays (holidays.Indonesia()) +
    Nyepi, Waisak, Idul Fitri (15 holidays)
  • Surabaya Monsoon Phase: NW Monsoon (wet, Nov–Apr) vs
    Dry Season (May–Oct)

  FEATURE SELECTION (T15):
  • Correlation matrix + Variance Inflation Factor (VIF)
    checked
  • Multicollinearity controlled; 27 curated modeling
    features (all VIF < 10)
=============================================================================================================================
```

---

## 3. Word-for-Word Spoken Presentation Script (~90–120 Seconds)

> *"Good morning respected evaluators. I am Sai Sri Satvic, presenting the data preparation, feature engineering pipeline, and feature selection architecture for our Surabaya F&B analytics project (Tasks T13 through T15).*
>
> *Starting with **Data Cleaning and Merging (Task T13)**: our raw inputs consisted of over 53,800 Point-of-Sale order transactions across three restaurant outlets in Surabaya, and historical weather records from the Open-Meteo API. We aggregated transactional orders into a homogeneous daily grain, calculating daily order counts, total revenue, average order value, and basket size in Indonesian Rupiah. We then performed an inner join with daily meteorological records on calendar date, yielding 272 fully continuous days from January 2nd through September 30th, 2025. We audited data types, verified zero missing dates and zero duplicates, and evaluated IQR outliers. All statistical outliers were deliberately retained because tropical rain downpours reflect genuine operational surges and walk-in friction rather than sensor errors.*
>
> *Moving to **Feature Engineering (Task T14)**: to capture temporal dynamics without lookahead bias, we engineered Lag-1, Lag-7, and 7-day rolling means strictly on shifted history—meaning rolling windows are applied only after `.shift(1)`. This completely eliminates data leakage. We mapped weather variables into behavioral thresholds: rainfall into binary rain and heavy rain flags ($\ge 2.5$ mm and $\ge 15.0$ mm) alongside four intensity bins; temperature into thermal comfort bands with a hot threshold at $33^\circ\text{C}$; weekend indicators for Friday through Sunday; fifteen Indonesian national public holidays via `holidays.Indonesia()`; and the tropical monsoon regime distinguishing the wet Northwest Monsoon from the dry season.*
>
> *For **Feature Selection (Task T15)**: we analyzed the correlation structure and diagnosed multicollinearity using Variance Inflation Factors (VIF). Unlike synthetic datasets, our real Surabaya transactions display authentic day-to-day autocorrelation—our 7-day rolling mean and Lag-1 correlate at $+0.81$ and $+0.77$ with order volume, while rainfall acts as physical travel friction, moderating store walk-in orders with a negative correlation of $-0.21$. To eliminate severe multicollinearity, we dropped redundant variables—specifically precipitation millimeters, mean temperature, and minimum temperature. Every single retained predictor achieved a VIF safely below the threshold of 10, with our rolling feature peaking at 9.12. Because multicollinearity is fully controlled, PCA was rejected to preserve complete business interpretability for restaurant operators.*
>
> *Finally, our pipeline is fully automated and modular, verified by our unit test suite. In Task T16 and T17, we have formatted all transactions into cleaned itemsets, priming the codebase for FP-Growth and Apriori association rule mining in Review 2. Thank you."*

---

## 4. Task-by-Task Technical Deep Dive

### 4.1 T13: Data Cleaning, Quality Auditing & Aggregation
* **Grain Alignment:** POS transactions occur at irregular minute intervals; atmospheric weather changes across days. We aggregate 53,820 order transactions into daily summary metrics ($n = 272$ days).
* **Target Leakage Quarantine:** We explicitly quarantine same-day financial outcomes:
  - `total_revenue`
  - `avg_order_value`
  - `avg_basket_size`  
  Because revenue is only known *after* daily orders conclude, using them as inputs to predict `order_count` would constitute target leakage. They are retained in `daily_merged.csv` purely for descriptive EDA, not feature modeling.
* **IQR Outlier Retention Decision:** Traditional statistical cleansing automatically trims data outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$. However, in restaurant operational analytics, demand spikes during extreme weather (e.g., severe storms $> 40$ mm) or major holidays (Idul Fitri) represent critical capacity stress tests. Trimming them artificially depresses model forecasts during severe storms.

### 4.2 T14: Feature Engineering & Leakage Prevention
* **Leakage-Free Temporal Features:**
  - `order_count_lag1`: Yesterday's volume ($y_{t-1}$)
  - `order_count_lag7`: Same weekday last week ($y_{t-7}$)
  - `order_count_roll7`: Moving average of past 7 days ($\frac{1}{7} \sum_{i=1}^{7} y_{t-i}$)
  - **Mathematical Guarantee:** Applied after `.shift(1)`. Row $t$ sees zero data from day $t$.
* **Weather Regimes:**
  - `is_rainy`: $\text{rain\_mm} \ge 2.5$
  - `heavy_rain`: $\text{rain\_mm} \ge 15.0$
  - `rain_intensity`: Categorical bins `[none, trace, moderate, heavy)` with `right=False` to ensure exact boundary alignment with binary flags.
  - `is_hot`: $\text{temp\_max\_c} \ge 33.0^\circ\text{C}$
  - `temp_band`: Categorical bins `[cool, warm, hot, extreme)`
* **Indonesian Calendar & Monsoon Dynamics:**
  - Weekend: Friday, Saturday, Sunday (matching retail dining habits).
  - Indonesian Public Holidays: 15 recognized national holidays in 2025 (Nyepi, Eid al-Fitr, Vesak Day, Independence Day, etc.).
  - Monsoon Phase: Northwest Monsoon (wet season, November through April) vs. Southeast / Dry Season (May through October), capturing East Java tropical macro-climatic shifts.

### 4.3 T15: Feature Selection, Correlation Analysis & VIF Diagnostics
* **Correlation Profile with `order_count`:**
  - `order_count_roll7`: $r = +0.806$ (strong local trend)
  - `order_count_lag1`: $r = +0.769$ (real-world business autocorrelation)
  - `order_count_lag7`: $r = +0.645$ (strong weekly cyclical pattern)
  - `is_rainy`: $r = -0.213$ (walk-in travel friction in tropical Surabaya)
  - `humidity_mean_pct`: $r = -0.191$
  - `rain_mm`: $r = -0.167$
  - `is_hot`: $r = -0.166$
  - `heavy_rain`: $r = -0.120$
* **Variance Inflation Factor (VIF) Diagnostic:**
  Computed with an explicit intercept constant ($\text{VIF}_j = \frac{1}{1 - R_j^2}$):
  
  | Feature | VIF | Status |
  | :--- | :---: | :--- |
  | `order_count_roll7` | **9.12** | Controlled ($< 10$) |
  | `order_count_lag7` | **4.34** | Healthy |
  | `order_count_lag1` | **4.28** | Healthy |
  | `humidity_mean_pct` | **3.40** | Healthy |
  | `is_rainy` | **3.06** | Healthy |
  | `rain_mm` | **2.57** | Healthy |
  | `temp_max_c` | **1.95** | Healthy |
  | `is_hot` | **1.34** | Healthy |
  | `is_festival` | **1.04** | Independent |
  | `is_weekend` | **1.02** | Independent |

* **Redundant Variables Dropped:**
  - `precip_mm`: 100% duplicate of `rain_mm` in this dataset ($\text{VIF} = \infty$).
  - `temp_mean_c`: Exact mathematical midpoint of daily min and max ($\text{VIF} > 60$).
  - `temp_min_c`: Highly collinear with `temp_max_c`.
* **The PCA Decision:** Rubric specifies: *"PCA only if multicollinearity is a problem."* Because all post-selection VIFs are under 10, multicollinearity is solved. Applying PCA would project intuitive business drivers (millimeters of rain, holiday flags) onto abstract eigenvectors, destroying operational utility.

### 4.4 T16–T17: Pipeline Reproducibility & ARM Roadmap (Review 2)
* **Modular Codebase:** `src/preprocess.py`, `src/features.py`, `src/eda.py`, and `src/predict.py`.
* **Automated Testing:** `tests/test_features.py` enforces zero-leakage assertions, threshold boundary checks, and holiday flag consistency.
* **ARM Preparation:** Multi-item receipts are converted into standard semicolon-delimited transactional strings (`order_items.csv` and `orders.csv`), ready for `mlxtend`'s `TransactionEncoder`, Apriori, and FP-Growth segmented by rainy vs. dry weather regimes in Review 2.

---

## 5. Code Architecture Walkthrough

### 5.1 Zero-Leakage Feature Pipeline (`src/features.py`)
```python
def add_lag_features(df: pd.DataFrame, target: str = "order_count") -> pd.DataFrame:
    """Lag-1, lag-7, and 7-day rolling mean of target.
    
    Rolling window is applied strictly AFTER shift(1) so that only
    information available prior to the predicted day is used.
    """
    df = df.sort_values("date").copy()
    df[f"{target}_lag1"] = df[target].shift(1)
    df[f"{target}_lag7"] = df[target].shift(7)
    df[f"{target}_roll7"] = df[target].shift(1).rolling(7).mean()
    return df
```

### 5.2 Behavioral Weather Thresholds (`src/features.py`)
```python
RAIN_THRESHOLD = 2.5        # mm -> is_rainy
HEAVY_RAIN_THRESHOLD = 15.0 # mm -> heavy_rain
HOT_THRESHOLD = 33.0        # °C -> is_hot
WEEKEND_DAYS = {4, 5, 6}    # Friday, Saturday, Sunday

def add_rain_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_rainy"] = (df["rain_mm"] >= RAIN_THRESHOLD).astype(int)
    df["heavy_rain"] = (df["rain_mm"] >= HEAVY_RAIN_THRESHOLD).astype(int)
    # Left-closed bins [a, b) ensure boundary consistency
    df["rain_intensity"] = pd.cut(
        df["rain_mm"],
        bins=[-float("inf"), 1e-9, RAIN_THRESHOLD, HEAVY_RAIN_THRESHOLD, float("inf")],
        labels=["none", "trace", "moderate", "heavy"],
        right=False,
    )
    return df
```

### 5.3 Surabaya Tropical Monsoon Tagging (`src/features.py`)
```python
def add_monsoon_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Tag Surabaya monsoon season: NW monsoon (wet season, Nov-Apr), dry season (May-Oct)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    month = df["date"].dt.month
    df["monsoon_phase"] = np.where(
        month.isin([11, 12, 1, 2, 3, 4]),
        "nw_monsoon",
        "dry_season",
    )
    return df
```

### 5.4 Multicollinearity & VIF Calculation (`notebooks/build_03_preprocess.py`)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Selected 10 features after dropping redundant collinear channels
selected_features = [
    "order_count_lag1", "order_count_lag7", "order_count_roll7",
    "rain_mm", "temp_max_c", "humidity_mean_pct",
    "is_weekend", "is_hot", "is_rainy", "is_festival",
]

# Adding constant is mandatory for centered, valid VIF
vif_data = sm.add_constant(features[selected_features].dropna())
vif_df = pd.DataFrame({
    "Feature": selected_features,
    "VIF": [variance_inflation_factor(vif_data.values, i + 1)
            for i in range(len(selected_features))],
}).sort_values("VIF", ascending=False)
```

---

## 6. Comprehensive Panel Q&A Defense & Trap Questions

### Q1: Why does rainfall have a negative correlation ($r = -0.213$) with orders in Surabaya? Doesn't bad weather increase delivery demand?
* **Defense:**  
  *"In Western delivery-app case studies or cold regions, rain often boosts delivery orders. However, in tropical Southeast Asian MSME F&B environments—such as the Surabaya physical outlets in our dataset—severe tropical monsoon rain causes localized street flooding, waterlogged roads, and transit friction. Because this POS dataset captures walk-in and physical restaurant transactions across three outlets, torrential downpours suppress foot traffic and cause customers to postpone dining out, resulting in a negative correlation ($r = -0.213$). Furthermore, our item-level analysis in Task T19 demonstrates that while total volume dips, demand shifts internally toward hot comfort beverages and warm snacks."*

### Q2: Why is the VIF for `order_count_roll7` at 9.12? Is that dangerously close to the threshold of 10?
* **Defense:**  
  *"A VIF of 9.12 is completely natural and mathematically expected for a rolling 7-day average of order volume, because it is inherently correlated with Lag-1 ($r = +0.77$) and Lag-7 ($r = +0.65$). The critical insight is that it remains strictly below the accepted diagnostic threshold of 10. Keeping `order_count_roll7` is crucial because it provides the model with baseline trend momentum, enabling our ridge and tree models to beat naive seasonal baselines by over 21% in MAE."*

### Q3: How do you mathematically guarantee that your lag and rolling features do not leak future information?
* **Defense:**  
  *"We guarantee zero data leakage by enforcing a strict temporal shift before applying any rolling statistics: `df[target].shift(1).rolling(7).mean()`. This ensures that on day $t$, the rolling calculation only includes observations up to day $t-1$. The value of day $t$ is never incorporated into its own input vector. We have unit tests in `tests/test_features.py` that explicitly verify row $t$ contains `NaN` on the first row and matches historical slices precisely."*

### Q4: Why did you drop `precip_mm`, `temp_mean_c`, and `temp_min_c`?
* **Defense:**  
  *"We dropped them based on structural redundancy. In Open-Meteo Surabaya weather, `precip_mm` and `rain_mm` are identical, creating perfect collinearity ($R^2 = 1.0$) which produces an infinite VIF. `temp_mean_c` is an exact mathematical linear combination of maximum and minimum temperature ($\text{VIF} > 60$). Dropping the redundant columns restored full rank to our design matrix while preserving maximum temperature and rain millimeters as intuitive, actionable predictors."*

### Q5: Why did you reject PCA for dimensionality reduction?
* **Defense:**  
  *"We followed the project rubric: 'PCA only if multicollinearity is a problem.' Our post-selection VIF analysis confirmed that all 10 features are below 10, meaning multicollinearity is already resolved. In addition, when features represent distinct real-world units—millimeters of rain, degrees Celsius, holiday flags, and past order counts—PCA would project them into abstract linear combinations. A restaurant manager cannot take operational inventory actions based on an abstract eigenvector, whereas they can immediately act on 'heavy rain over 15 mm'."*

### Q6: Why did you group Friday, Saturday, and Sunday into `is_weekend`?
* **Defense:**  
  *"In Indonesian urban retail and dining culture, Friday evening marks the structural onset of leisure dining, social gatherings, and weekend spending patterns. Analyzing daily volume confirms that Friday through Sunday behaves as a unified high-traffic block compared to the Monday through Thursday utilitarian baseline."*

---

## 7. Master Glossary of Keywords & Statistical Concepts

* **Daily Aggregation:** The process of condensing granular, multi-outlet order receipts into one single observation per calendar day.
* **Grain Alignment:** Ensuring both independent variables (daily weather) and target variables (daily orders) exist at the identical temporal unit of analysis.
* **Data Leakage (Lookahead Bias):** An analytical error where information from the forecast period is inadvertently used as a predictor, producing artificially inflated model accuracy that fails in production.
* **Autoregressive Memory ($AR(1)$):** The degree to which yesterday’s demand level persists into today’s transaction volume.
* **Interquartile Range (IQR):** The distance between the 25th percentile ($Q_1$) and 75th percentile ($Q_3$); statistical threshold for detecting outliers without making Gaussian assumptions.
* **Variance Inflation Factor (VIF):** A regression diagnostic index ($\frac{1}{1 - R_j^2}$) measuring how much the variance of an estimated regression coefficient increases due to collinearity with other features. Values $> 10$ signal problematic multicollinearity.
* **Multicollinearity:** High linear correlation among two or more predictor variables, destabilizing matrix inversion and regression coefficient estimates.
* **Principal Component Analysis (PCA):** An orthogonal linear transformation that converts correlated features into uncorrelated principal components ranked by variance explained.
* **Left-Closed Interval $[a, b)$:** A range definition inclusive of the lower bound $a$ and exclusive of the upper bound $b$; critical for aligning categorical bin edges with boolean inequality thresholds.
* **Northwest Monsoon (`nw_monsoon`):** The maritime wet season in East Java / Surabaya running from November through April, characterized by heavy convective tropical precipitation.
* **Southeast Dry Season (`dry_season`):** The dry season in East Java running from May through October, characterized by lower precipitation and higher solar irradiance.
* **Itemset:** A discrete set of items purchased together in a single POS transaction basket, formatted for association rule mining algorithms.
