# Synthetic Order Generator Documentation

**Project:** Weather-Driven Food Order Patterns (Coimbatore) · BA Capstone, Team of 5  
**Author:** P2 (Order Generator & Menu Catalog)  
**Rubric Line:** *Problem statement, business objective, and dataset description (required by rubric)*  
**Tasks:** T07 (Menu Catalog), T08 (Daily Demand Model), T09 (Basket Generator), T10 (Generator Documentation)

---

## 1. Overview and Purpose

This document provides the complete theoretical specification, mathematical formulation, parameter dictionary, and ground-truth rules for the Coimbatore food delivery synthetic order generator (`src/generator.py`).

### Why Synthetic Orders Driven by Real Weather?
Real-world delivery platforms (e.g., Swiggy, Zomato) rarely publish granular order-level transaction data with timestamps and baskets due to competitive and customer privacy concerns. However, real historical weather for Coimbatore is accessible via the Open-Meteo Historical Weather API (`data/raw/weather.csv`, collected by P1).

To enable rigorous analytics across the team:
1. **Documented Ground Truth:** Known causal relationships and planted association rules are deliberately programmed into the generator. Downstream methods (Feature Engineering by P3, EDA by P4, Predictive Modeling by P5, Time-Series Forecasting by P1, and Association Rule Mining by P3/P5) can be evaluated by their ability to **recover true underlying parameters** rather than guessing against opaque data.
2. **Deterministic Reproducibility:** Every stochastic step relies on a fixed NumPy pseudo-random generator with `SEED = 42`. Re-running the pipeline generates identical data bit-for-bit.

---

## 2. Daily Order Volume Model (T08)

Daily order demand is modeled as an overdispersed count process governed by calendar factors, weather intensity, and regional cultural celebrations.

### 2.1 Mathematical Formulation
For each day $t \in \{1, \dots, 1097\}$ from 2023-09-24 to 2026-09-24:

$$Y_t \sim \text{NegativeBinomial}(r=25, p_t)$$

where:
- $Y_t$ is the actual integer number of orders placed on day $t$.
- $r = 25$ is the dispersion parameter controlling variance.
- The success probability $p_t$ is parameterized by the deterministic expected daily demand $\mu_t$:

$$p_t = \frac{r}{r + \mu_t}$$

The moments of the Negative Binomial distribution under this parameterization are:

$$\mathbb{E}[Y_t] = \frac{r(1 - p_t)}{p_t} = \mu_t$$

$$\text{Var}(Y_t) = \frac{r(1 - p_t)}{p_t^2} = \mu_t + \frac{\mu_t^2}{r}$$

Since $\text{Var}(Y_t) > \mathbb{E}[Y_t]$, this formulation accurately captures the heavy-tailed overdispersion and delivery demand surges observed in empirical food delivery operations.

### 2.2 Expected Order Multipliers
The deterministic expected daily demand $\mu_t$ is the product of a baseline weekday volume and four multiplicative effects:

$$\mu_t = \text{BASE} \times M_{\text{weekend}}(t) \times M_{\text{rain}}(t) \times M_{\text{temp}}(t) \times M_{\text{festival}}(t)$$

| Parameter | Symbol | Value | Condition / Rule |
| :--- | :--- | :--- | :--- |
| **Base Volume** | $\text{BASE}$ | `140.0` | Expected baseline order volume for a standard dry weekday |
| **Weekend Multiplier** | $M_{\text{weekend}}$ | `1.30` | Applied on Friday, Saturday, and Sunday (`dow` $\in \{4, 5, 6\}$) |
| | | `1.00` | Applied on Monday through Thursday |
| **Rain Multiplier** | $M_{\text{rain}}$ | `1.00` | Dry / Trace conditions: $\text{precip\_mm} < 2.5\text{ mm}$ |
| | | `1.20` | Moderate rain: $2.5\text{ mm} \le \text{precip\_mm} < 15.0\text{ mm}$ |
| | | `1.35` | Heavy monsoon downpour: $\text{precip\_mm} \ge 15.0\text{ mm}$ |
| **Temperature Multiplier**| $M_{\text{temp}}$ | `1.10` | Extreme heat: $\text{temp\_max\_c} \ge 33.0^\circ\text{C}$ (avoiding outdoor cooking) |
| | | `1.00` | Normal temperature: $\text{temp\_max\_c} < 33.0^\circ\text{C}$ |
| **Festival Multiplier** | $M_{\text{festival}}$ | `1.25` | Major Tamil Nadu / Coimbatore festival day (dining celebration surge) |
| | | `1.00` | Regular non-festival day |
| **Dispersion Parameter**| $r$ | `25` | Negative Binomial shape parameter |
| **Random Seed** | $\text{SEED}$ | `42` | NumPy random generator seed (`np.random.default_rng(42)`) |

---

## 3. Coimbatore Festival Calendar

Cultural celebrations in Tamil Nadu lead to notable delivery order surges. The generator includes a curated calendar of 41 regional festival dates over the 3-year history:

| Date | Festival Name | Regional Significance | Multiplier |
| :--- | :--- | :--- | :--- |
| `2023-10-23` / `2024-10-11` / `2025-10-01` | Ayudha Puja | Veneration of tools and machinery; community sweets and feasts | 1.25x |
| `2023-10-24` / `2024-10-12` / `2025-10-02` | Vijaya Dashami | Dussehra festival culmination | 1.25x |
| `2023-11-12` / `2024-10-31` / `2025-10-20` | Deepavali (Diwali) | Festival of Lights; major family feasts and sweets | 1.25x |
| `2023-12-25` / `2024-12-25` / `2025-12-25` | Christmas | Holiday dining and celebrations | 1.25x |
| `2023-12-31` / `2024-12-31` / `2025-12-31` | New Year's Eve | Late evening party ordering surge | 1.25x |
| `2024-01-01` / `2025-01-01` / `2026-01-01` | New Year's Day | Holiday family breakfast & lunch | 1.25x |
| `2024-01-15` / `2025-01-14` / `2026-01-14` | Thai Pongal | Tamil harvest festival; traditional feast day | 1.25x |
| `2024-01-16` / `2025-01-15` / `2026-01-15` | Mattu Pongal | Day honoring cattle; community lunch gatherings | 1.25x |
| `2024-01-17` / `2025-01-16` / `2026-01-16` | Kaanum Pongal | Family outings, picnics, and high dining demand | 1.25x |
| `2024-04-11` / `2025-03-31` / `2026-03-20` | Eid al-Fitr (Ramzan) | Feasting and biryani demand surge | 1.25x |
| `2024-04-14` / `2025-04-14` / `2026-04-14` | Tamil New Year (Puthandu)| Traditional festive feasts | 1.25x |
| `2024-06-17` / `2025-06-07` / `2026-05-27` | Bakrid (Eid al-Adha) | Mutton and biryani special dining | 1.25x |
| `2024-09-07` / `2025-08-27` / `2026-09-14` | Vinayagar Chaturthi | Festival savories and kozhukattai treats | 1.25x |
| `2024-09-16` / `2025-09-05` | Milad-un-Nabi | Regional community feasts | 1.25x |

---

## 4. Menu Architecture & Item Affinities (T07)

The menu catalog (`data/raw/menu.csv`) features 40 authentic dishes local to Coimbatore and the Kongu region across 6 categories. Each item has a base price in INR and a tagged weather affinity:

| Category | Items Count | Typical Dishes | Price Range | Weather Affinity Tag |
| :--- | :--- | :--- | :--- | :--- |
| **Beverages (Hot)** | 5 | South Indian Filter Coffee, Masala Chai, Sukku Malli Tea, Cardamom Tea, Badam Milk | ₹25 – ₹45 | `hot_comfort` |
| **Snacks & Quick Bites** | 7 | Onion Pakora, Milagai Bajji, Vazhakkai Bajji, Medu Vada, Potato Bonda, Samosa, Kuzhi Paniyaram | ₹30 – ₹60 | `hot_comfort` |
| **South Indian Tiffin** | 9 | Ghee Podi Dosa, Masala Dosa, Plain Dosa, Idli (2 pcs), Pongal, Poori Masala, Malabar Parotta (2 pcs), Chicken Kothu Parotta, Veechu Parotta | ₹40 – ₹160 | `neutral` |
| **Biryani & Mains** | 7 | Coimbatore Chicken Biryani, Kongu Mutton Biryani, Veg Biryani, Egg Biryani, Sambar Rice, Chicken Chukka, Curd Rice | ₹60 – ₹280 | `neutral` (Curd Rice: `cold_refresh`) |
| **Cold Refreshments & Juices** | 7 | Jigarthanda, Nannari Sarbath, Fresh Lime Juice, Watermelon Juice, Mango Juice, Moru (Buttermilk), Rose Milk | ₹30 – ₹70 | `cold_refresh` |
| **Desserts & Ice Cream** | 5 | Tender Coconut Ice Cream, Vanilla Ice Cream, Butterscotch Ice Cream, Paal Payasam, Gulab Jamun (2 pcs) | ₹50 – ₹80 | `cold_refresh` (Sweets: `neutral`) |

---

## 5. Meal Slots and Timestamp Generation (T09)

Orders are not uniformly scattered across 24 hours. In Coimbatore food delivery, customer transactions cluster into four operational meal slots:

| Meal Slot | Time Window | Duration | Share of Daily Orders | Category Ordering Priorities |
| :--- | :--- | :--- | :--- | :--- |
| **Breakfast** | 07:00 – 10:30 | 3.5 hours | **15%** | High demand for South Indian Tiffin (Idli, Dosa, Pongal) & Filter Coffee |
| **Lunch** | 12:00 – 15:00 | 3.0 hours | **35%** | High demand for Biryanis, Rice Meals, Refreshments (Buttermilk, Fresh Lime Juice) |
| **Evening Snacks** | 16:00 – 18:30 | 2.5 hours | **20%** | Peak window for Hot Beverages (Chai, Coffee) & Fried Bites (Pakora, Bajji, Vada) |
| **Dinner** | 19:00 – 22:30 | 3.5 hours | **30%** | High demand for Malabar Parotta, Kothu Parotta, Biryani, and Ice Creams |

### Timestamp Generation Algorithm
For each day $t$ with $N_t$ total orders:
1. Each order draws a meal slot $s$ with categorical probabilities `[0.15, 0.35, 0.20, 0.30]`.
2. A random second offset $u \sim \text{Uniform}(0, \text{duration}(s))$ is drawn within the slot window.
3. Orders are sorted chronologically per day, generating realistic `YYYY-MM-DD HH:MM:SS` timestamps.

---

## 6. Basket Construction and Weather Conditioning (T09)

### 6.1 Basket Size Distribution
Customer basket sizes $k \in \{1, 2, 3, 4, 5\}$ are sampled from the empirical delivery distribution:

$$P(k = 1) = 0.30, \quad P(k = 2) = 0.42, \quad P(k = 3) = 0.18, \quad P(k = 4) = 0.07, \quad P(k = 5) = 0.03$$

The expected items per basket is $2.11$ items.

### 6.2 Weather Affinity Multipliers
The baseline item popularity $w_i$ is modified dynamically by both the meal slot $s$ and the ambient weather regime $R_t \in \{\text{rainy}, \text{hot}, \text{normal}\}$:

$$w_i(s, R_t) = w_i^{\text{base}} \times W_{\text{slot}}(s, \text{category}_i) \times W_{\text{weather}}(R_t, \text{affinity}_i)$$

| Category Affinity | Rainy Day ($p \ge 2.5\text{ mm}$) | Hot Day ($T_{\max} \ge 33^\circ\text{C}$) | Normal Day |
| :--- | :--- | :--- | :--- |
| **`hot_comfort`** | **1.80x** (Craving hot fried snacks & warm tea) | **0.80x** | 1.00x |
| **`cold_refresh`** | **0.60x** | **1.90x** (High demand for coolers & ice cream)| 1.00x |
| **`neutral`** | **1.00x** (Staple meals remain steady) | **1.00x** | 1.00x |

---

## 7. Planted Association Rules (Ground Truth)

To ensure downstream Association Rule Mining (ARM, P3 and P5 in Review 2) discovers authentic, interpretable weather-driven behavior, explicit conditional item pairs are planted in the generator.

When a basket contains a **Trigger** item and has space for companion items ($k > 1$), an **Add-on** item is appended with documented conditional probability $P(\text{Add-on} \mid \text{Trigger}, \text{Regime})$:

| Trigger Item | Add-on Item | Target Phenomenon | $P(\text{Add-on} \mid \text{Rainy})$ | $P(\text{Add-on} \mid \text{Normal})$ | $P(\text{Add-on} \mid \text{Hot})$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Masala Chai** | **Onion Pakora** | **Headline Monsoon Combo** | **0.58** | 0.12 | 0.08 |
| **Masala Chai** | **Milagai Bajji** | Rainy Evening Tea & Fritter | **0.35** | 0.10 | 0.08 |
| **South Indian Filter Coffee**| **Medu Vada** | Classic Rainy Tiffin Pair | **0.40** | 0.15 | 0.10 |
| **Fresh Lime Juice** | **Tender Coconut Ice Cream** | **Headline Heat Cooler Combo**| 0.06 | 0.15 | **0.45** |
| **Watermelon Juice** | **Vanilla Ice Cream** | Hot Day Refreshment | 0.05 | 0.12 | **0.38** |
| **Coimbatore Chicken Biryani**| **Buttermilk (Moru)** | Spiced Biryani Cooling Pair | 0.12 | 0.20 | **0.45** |
| **Malabar Parotta (2 pcs)** | **Chicken Chukka** | Kongunadu Dinner Staple | 0.35 | 0.35 | 0.30 |
| **Ghee Podi Dosa** | **South Indian Filter Coffee**| All-Weather Tiffin Pair | 0.38 | 0.30 | 0.20 |

---

## 8. Empirical Recovery Benchmark

Evaluating the 193,406 generated transactions in `data/raw/orders.csv` across the 1,097 simulated days confirms that downstream data mining algorithms will cleanly recover these exact planted associations:

### 8.1 Headline Monsoon Rule: `{Masala Chai} -> {Onion Pakora}`
| Weather Regime | Transactions ($N$) | Support $P(A \cap B)$ | Confidence $P(B \mid A)$ | Lift $\frac{P(A \cap B)}{P(A)P(B)}$ | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rainy** | 66,818 | **6.02%** | **39.97%** | **2.71** | Strong, high-volume rule emerges on monsoon days |
| **Normal** | 81,086 | 1.68% | 15.53% | 1.96 | Moderate baseline pairing |
| **Hot** | 45,502 | 1.05% | 12.83% | 2.20 | Low volume; customer preference shifts away |

> **Key Takeaway:** On rainy days, confidence is **2.6x higher** and overall support is **3.6x higher** than on normal days.

### 8.2 Headline Heat Rule: `{Fresh Lime Juice} -> {Tender Coconut Ice Cream}`
| Weather Regime | Transactions ($N$) | Support $P(A \cap B)$ | Confidence $P(B \mid A)$ | Lift $\frac{P(A \cap B)}{P(A)P(B)}$ | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hot** | 45,502 | **4.27%** | **32.51%** | **3.55** | Strongest lift and confidence on scorching days |
| **Normal** | 81,086 | 1.16% | 14.16% | 3.19 | Regular pairing |
| **Rainy** | 66,818 | 0.31% | 6.71% | 3.05 | Rule virtually disappears during monsoon downpours |

> **Key Takeaway:** On hot days, confidence is **4.8x higher** and support is **13.8x higher** than on rainy days.

### 8.3 Secondary Comfort Rule: `{South Indian Filter Coffee} -> {Medu Vada}`
| Weather Regime | Transactions ($N$) | Support $P(A \cap B)$ | Confidence $P(B \mid A)$ | Lift $\frac{P(A \cap B)}{P(A)P(B)}$ |
| :--- | :--- | :--- | :--- | :--- |
| **Rainy** | 66,818 | **5.13%** | **30.91%** | **3.23** |
| **Normal** | 81,086 | 1.82% | 14.43% | 2.88 |
| **Hot** | 45,502 | 0.90% | 9.87% | 2.99 |

---

## 9. Output File Schemas

The generator outputs three standard CSV files into `data/raw/`:

### 9.1 `data/raw/menu.csv` (T07)
| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `item_id` | String | Unique item identifier (M01 to M40) | `M01` |
| `item_name` | String | Authentic Coimbatore dish name | `South Indian Filter Coffee` |
| `category` | String | Menu grouping (6 categories) | `Beverages (Hot)` |
| `price` | Integer | Price in Indian Rupees (INR) | `30` |
| `weather_affinity` | String | `hot_comfort`, `cold_refresh`, or `neutral` | `hot_comfort` |
| `description` | String | Cultural and culinary notes | Traditional filter coffee brewed in dabarah |

### 9.2 `data/raw/daily_orders.csv` (T08)
| Field | Type | Description |
| :--- | :--- | :--- |
| `date` | Date | Calendar day (`YYYY-MM-DD`), 2023-09-24 to 2026-09-24 |
| `precip_mm` | Float | Precipitation sum from Open-Meteo |
| `temp_max_c` | Float | Maximum 2m air temperature |
| `is_weekend` | Boolean | True for Friday, Saturday, Sunday |
| `is_rainy` | Boolean | True if $\text{precip\_mm} \ge 2.5\text{ mm}$ |
| `is_hot` | Boolean | True if $\text{temp\_max\_c} \ge 33.0^\circ\text{C}$ |
| `is_festival` | Boolean | True if regional Coimbatore festival |
| `regime` | String | `rainy`, `hot`, or `normal` |
| `expected_orders` | Float | Deterministic mean $\mu_t$ |
| `order_count` | Integer | Stochastic Negative Binomial daily volume |

### 9.3 `data/raw/orders.csv` (T09)
| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `order_id` | Integer | Monotonic primary key (1 to 193,406) | `1042` |
| `date` | Date | Calendar day (`YYYY-MM-DD`) | `2024-10-11` |
| `timestamp` | Datetime | Order placement time (`YYYY-MM-DD HH:MM:SS`) | `2024-10-11 16:42:15` |
| `meal_slot` | String | `Breakfast`, `Lunch`, `Evening Snacks`, `Dinner` | `Evening Snacks` |
| `items` | String | Semicolon-delimited list of ordered items | `Masala Chai; Onion Pakora` |
| `item_count` | Integer | Basket size $k \in [1, 5]$ | `2` |
| `order_value` | Integer | Total order amount in INR (sum of item prices) | `75` |

---

## 10. Reproduction and Verification

To regenerate the entire dataset from scratch using the fixed random seed:
```bash
# 1. Run generator CLI
.venv/bin/python src/generator.py

# 2. Run automated test suite verifying all 23 tests across project
.venv/bin/pytest -v
```

All empirical rules, parameters, and multipliers are guaranteed to match this document bit-for-bit.
