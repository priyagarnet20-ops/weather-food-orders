# Empirically Grounded Food Order Generator (Surabaya, Indonesia)

**Project:** Weather-Driven Food Order Patterns (Surabaya, Indonesia) · BA Capstone, Team of 5  
**Author:** P2 (Order Generator & Menu Catalog)  
**Rubric Line:** *Problem statement, business objective, and dataset description (required by rubric)*  
**Tasks:** T07 (Menu Catalog & Weather Affinities), T08 (Daily Demand Modeling), T09 (Basket Generation & Intraday Dynamics), T10 (Generator Documentation)

---

## 1. Executive Summary & Review 1 Pivot

During Review 1, the faculty committee rejected purely synthetic food order generation, clarifying the academic standard:
> *Synthetic transactional datasets are acceptable and methodologically valid only when mathematically derived from, calibrated by, and grounded in authentic, real-world Point-of-Sale (POS) base observations.*

To comply strictly with this requirement, the project pivoted from synthetic Coimbatore data to an **authentic transactional dataset from an active F&B Micro-Small-Medium Enterprise (MSME) in Surabaya, Indonesia**. We then mathematically estimated the underlying generative process of the restaurant operations from the real 272-day POS base dataset and used that empirical model to synthesize the missing periods:
1. **Real Ground-Truth Base Dataset:** 53,820 authentic POS orders spanning 272 days (January 2, 2025 to September 30, 2025) across 3 physical outlets.
2. **Empirical Calibration:** Maximum Likelihood Estimation (MLE) and Negative Binomial generalized linear modeling fitted against real weather and calendar variables.
3. **Synthetic Extension:** Generation of the preceding 12 months of 2024 (366 days, leap year) and the closing quarter of 2025 (93 days), creating a **complete 2-year (731 continuous calendar days) dataset** from `2024-01-01` to `2025-12-31`.
4. **Data Transparency & Provenance:** Every order and daily summary contains an explicit `is_synthetic` flag (`0` for real POS transactions, `1` for grounded synthetic generations), guaranteeing 100% academic integrity.

---

## 2. Dataset Attribution & Base Data Characteristics

### 2.1 Research Citation
- **Source:** Universitas Ciputra Surabaya research repository, Mendeley Data (`kcgf45y24m/2`).
- **Publication:** *Anonymous transactional dataset in a local food and beverages (F&B) micro-small-medium enterprise (MSME) for recommender systems*, *Data in Brief*, 2026. DOI: `10.1016/j.dib.2026.112519`.
- **Raw Artifacts:**
  - `data/raw/Products.csv`: 96 products across 6 categories with authentic retail prices in Indonesian Rupiah (IDR / Rp).
  - `data/raw/Transactions.csv`: 53,820 transaction records from 3 retail outlets (`SHOP001`: 23,059; `SHOP002`: 16,843; `SHOP003`: 13,918).

### 2.2 Outlet Combination Justification
The 3 outlets represent physical storefronts of the same MSME in Surabaya. In retail operations analytics, city-wide weather events (monsoon downpours, extreme tropical heat index) affect foot traffic systematically across the metropolitan area. Aggregating the outlets produces an enterprise-level view:
- **Mean Daily Volume:** 197.9 orders/day (standard deviation: 33.0).
- **Variance-to-Mean Ratio:** $\sigma^2 / \mu = 5.51$, indicating substantial overdispersion requiring a Negative Binomial formulation rather than a standard Poisson distribution.

---

## 3. Mathematical & Probabilistic Demand Formulations (T08)

### 3.1 Daily Count Distribution: Negative Binomial Model
Daily order demand $Y_t$ on calendar day $t$ is modeled as a Negative Binomial random variable parameterized by dispersion parameter $r$ and success probability $p_t$:

$$Y_t \sim \text{NegativeBinomial}\left(r, p_t\right), \quad p_t = \frac{r}{r + \mu_t}$$

$$\mathbb{E}[Y_t] = \mu_t, \quad \text{Var}(Y_t) = \mu_t + \frac{\mu_t^2}{r}$$

Setting $r = 50.0$ matches the empirical dispersion ratio $\sigma_Y^2 / \bar{Y} \approx 5.51$ observed in the authentic 2025 POS data.

### 3.2 Expected Demand Specification
Expected demand $\mu_t$ is structured as a log-linear multiplicative process conditioned on day-of-week, precipitation regimes, thermal extremes, and national holidays:

$$\mu_t = \text{BASE} \times M_{\text{dow}}(t) \times M_{\text{rain}}(t) \times M_{\text{temp}}(t) \times M_{\text{festival}}(t)$$

Parameters were estimated via Poisson/Negative Binomial GLM regression on the 272-day real base dataset:

| Parameter | Value | 95% Conf. Interval / p-value | Empirical Interpretation |
|---|---|---|---|
| **$\text{BASE}$** | **205.8** | [199.4, 212.3], $p < 0.001$ | Baseline weekday volume under dry, temperate conditions |
| **$M_{\text{rain}}$ (Moderate: 2.5–15 mm)** | **0.933** ($-6.7\%$) | [0.898, 0.969], $p = 0.0002$ | Physical walk-in foot traffic dip during standard tropical showers |
| **$M_{\text{rain}}$ (Heavy: $\ge 15$ mm)** | **0.901** ($-9.9\%$) | [0.847, 0.958], $p = 0.0012$ | Significant foot traffic reduction during heavy monsoon downpours |
| **$M_{\text{temp}}$ (Heat: $\ge 33.0^\circ\text{C}$)** | **0.844** ($-15.6\%$) | [0.772, 0.923], $p = 0.0002$ | Reduced walk-in transit during extreme tropical afternoon heat |
| **$M_{\text{festival}}$ (Holidays)** | **0.982** ($-1.8\%$) | [0.931, 1.036], $p = 0.518$ | Commercial MSME holiday variation across Indonesian national holidays |

### 3.3 Day-of-Week Multipliers $M_{\text{dow}}$
Fitted directly to empirical day-of-week volume distributions in Surabaya:
- **Monday ($t=0$):** $0.9939$
- **Tuesday ($t=1$):** $1.0003$
- **Wednesday ($t=2$):** $1.0050$
- **Thursday ($t=3$):** $1.0008$
- **Friday ($t=4$):** $1.0208$
- **Saturday ($t=5$):** $0.9890$
- **Sunday ($t=6$):** $1.0263$

---

## 4. Intraday Temporal Clustering & Basket Architecture (T09)

### 4.1 Operational Meal Windows
Order arrival timestamps $T_i$ are assigned according to four empirical operational meal windows:

$$\text{MealSlot}(T_i) = \begin{cases} 
\text{Breakfast} & 07:00 \le T_i < 11:00 \quad (20.75\% \text{ of daily volume}) \\
\text{Lunch} & 11:00 \le T_i < 15:00 \quad (23.69\% \text{ of daily volume}) \\
\text{Afternoon Snacks} & 15:00 \le T_i < 18:00 \quad (19.15\% \text{ of daily volume}) \\
\text{Dinner} & 18:00 \le T_i < 23:00 \quad (36.40\% \text{ of daily volume}) \\
\text{Late Night} & 23:00 \le T_i < 07:00 \quad (0.00\% \text{ store closed})
\end{cases}$$

### 4.2 Intra-Slot Hourly Distribution
Within each meal slot, order arrival hours are sampled from empirical discrete probability masses derived from the POS transactions:
- **Breakfast:** Peak at 09:00–10:00 WIB (morning coffee & pastries).
- **Lunch:** Sharp peak at 12:00–13:00 WIB (rice bowls & lunch combos).
- **Snacks:** Peak at 16:00–17:00 WIB (tea & doughnut afternoon break).
- **Dinner:** Prolonged peak at 19:00–21:00 WIB (evening dining & social coffee).
- Minute $m \sim \text{Uniform}(0, 59)$ and second $s \sim \text{Uniform}(0, 59)$.

### 4.3 Basket Size Distribution
In the authentic POS data, transactions consist almost entirely of single items and pairs:
- **1-item baskets:** $84.85\%$ ($n=45,667$ in base dataset)
- **2-item baskets:** $15.15\%$ ($n=8,153$ in base dataset)

### 4.4 Item Selection & Weather-Conditioned Affinities
Each item $j \in \{1, \dots, 96\}$ has a base empirical sales frequency $w_j^0$ calculated from the 272-day POS transactions. Primary items are selected with weather-adjusted probabilities:

$$w_j(t) = w_j^0 \times A(j, \text{weather}_t)$$

Where affinity multiplier $A(j, \text{weather}_t)$ operates on item category tags (`hot_comfort`, `cold_refresh`, `neutral`):
- **Rainy Days ($\text{precip} \ge 2.5\text{ mm}$):**
  - $\text{hot\_comfort}$ items receive $+30\%$ boost ($A = 1.30$).
  - $\text{cold\_refresh}$ items receive $-25\%$ suppression ($A = 0.75$).
- **Hot Days ($\text{temp\_max} \ge 33.0^\circ\text{C}$):**
  - $\text{cold\_refresh}$ items receive $+30\%$ boost ($A = 1.30$).
  - $\text{hot\_comfort}$ items receive $-25\%$ suppression ($A = 0.75$).
- **Normal Days:** $A = 1.00$ for all items.

### 4.5 Empirical Co-Occurrence Transition Matrix for Multi-Item Baskets
For 2-item orders, the secondary item is **not** chosen independently. Instead, we built a $96 \times 96$ empirical transition matrix $P(\text{Item}_2 \mid \text{Item}_1)$ directly from all 8,153 authentic multi-item orders in the base dataset:

$$P(\text{Item}_2 = k \mid \text{Item}_1 = j) = \frac{C(j, k) + \epsilon}{\sum_{m \neq j} (C(j, m) + \epsilon)}$$

Where $C(j, k)$ is the observed co-occurrence count between item $j$ and item $k$, and $\epsilon = 0.05$ provides Laplace-style smoothing. This mathematical design guarantees that realistic product associations (e.g., `Friendly Coffee (Ice)` paired with `Donat Polos`, or `Menu Gedhe` rice bowls paired with artisan iced tea) are naturally preserved across all synthetic periods.

### 4.6 Pricing & Order Valuation
100% of item prices are deterministic lookups against `data/raw/menu.csv` in Indonesian Rupiah:

$$\text{OrderValue}_i = \sum_{j \in \text{Basket}_i} \text{Price}(j)$$

---

## 5. Summary of the Combined 2-Year Dataset (2024–2025)

The generated dataset seamlessly integrates real and grounded synthetic data across exactly 731 calendar days:

| Metric | Real Base Data (Jan–Sep 2025) | Grounded Synthetic (2024 + Q4 2025) | Unified 2-Year Dataset |
|---|---|---|---|
| **Calendar Coverage** | 2025-01-02 to 2025-09-30 | 2024-01-01 to 2024-12-31 & 2025-10-01 to 2025-12-31 | **2024-01-01 to 2025-12-31** |
| **Total Calendar Days** | 272 days | 459 days | **731 continuous days** |
| **Total Orders** | 53,820 orders | 88,798 orders | **142,618 orders** |
| **Mean Daily Orders** | 197.9 orders/day | 193.5 orders/day | **195.1 orders/day** |
| **Mean Basket Size** | 1.15 items | 1.15 items | **1.15 items** |
| **Mean Order Value** | Rp 25,261 | Rp 24,844 | **Rp 25,001** |
| **Median Order Value** | Rp 23,000 | Rp 23,000 | **Rp 23,000** |
| **Rainy Day Lift** | $-6.5\%$ | $-5.8\%$ | **$-6.0\%$ ($p < 0.001$)** |
| **Data Provenance Tag** | `is_synthetic = 0` | `is_synthetic = 1` | Verified in all tables |

---

## 6. Pipeline Architecture & Reproduction Commands

The entire pipeline is deterministic, fully reproducible with seed `42`, and executed with:

```bash
# 1. Generate 2-year grounded dataset (orders.csv, order_items.csv, daily_orders.csv, daily_merged.csv)
.venv/bin/python src/generator.py --seed 42

# 2. Engineer predictive features (daily_features.csv: 731 rows, 29 columns)
.venv/bin/python src/features.py

# 3. Execute unit test suite (62/62 passing)
.venv/bin/pytest tests/

# 4. Refresh EDA charts (results/eda/ & slides/review1/figures/)
.venv/bin/python src/eda.py
```
