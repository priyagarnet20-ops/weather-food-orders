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
