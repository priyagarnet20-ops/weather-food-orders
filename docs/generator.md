# Food Orders Dataset Preprocessing & Aggregation Pipeline

**Project:** Weather-Driven Food Order Patterns (Surabaya, Indonesia) · BA Capstone, Team of 5  
**Author:** P2 (Menu Catalog & Data Preprocessing)  
**Rubric Line:** *Problem statement, business objective, and dataset description (required by rubric)*  
**Tasks:** T07 (Menu Catalog), T08 (Daily Demand Aggregation), T09 (Multi-Outlet Preprocessing & Meal Slots), T10 (Preprocessing Pipeline Documentation)

---

## 1. Overview and Migration from Synthetic to Real POS Data

In Review 1, initial feedback emphasized that completely synthetic food order generation would not satisfy academic requirements for real business analytics. Consequently, the project pivoted from synthetic orders for Coimbatore to **authentic Point-of-Sale (POS) transaction records from an active F&B Micro-Small-Medium Enterprise (MSME) in Surabaya, Indonesia**.

### Dataset Provenance
- **Source:** Universitas Ciputra Surabaya research repository on Mendeley Data (`kcgf45y24m/2`).
- **Research Citation:** *Anonymous transactional dataset in a local food and beverages (F&B) micro-small-medium enterprise (MSME) for recommender systems*, *Data in Brief*, 2026. DOI: `10.1016/j.dib.2026.112519`.
- **Raw Files:**
  - `data/raw/Products.csv`: Product catalog containing product codes, product names, categories, and retail prices in Indonesian Rupiah (IDR / Rp).
  - `data/raw/Transactions.csv`: 53,820 transaction records from January 1, 2025 to September 29, 2025 across 3 physical outlets.

---

## 2. Multi-Outlet Aggregation (T08 & T09)

The MSME operated across three retail outlet locations:
- `SHOP001`: 23,059 orders
- `SHOP002`: 16,843 orders
- `SHOP003`: 13,918 orders
- **Total:** 53,820 orders across 272 days.

### Why Outlets Were Combined
Our business analytics objective is **weather-driven food order prediction and inventory planning for a unified restaurant operation**. Weather effects (monsoon downpours, extreme tropical heat, humidity) act at the city level across Surabaya. Having separate low-volume outlets would fragment demand and dilute meteorological signal. Aggregating all three outlets produces:
- A stable, realistic enterprise-level daily demand volume (mean: 197.9 orders/day, standard deviation: 33.0).
- Robust statistical power for machine learning models (Ridge, XGBoost, Random Forest) and time-series forecasting.
- Sufficient transaction density per hour across all operational meal windows.

---

## 3. Data Cleaning & Normalization Steps

### 3.1 Item String Standardisation & Parsing
In `Transactions.csv`, multi-item orders are stored as semicolon-delimited strings (e.g., `Friendly Coffee (Ice); Donat Polos`).
- Strings were parsed and stripped of whitespace.
- 94 unique item strings were extracted from transactions.
- All 94 unique transaction strings mapped directly to the official 96-item product catalog (`Products.csv` / `menu.csv`), achieving **100% SKU match rate**.

### 3.2 Price Imputation for Promotional Zero-Amount Transactions
A total of 1,733 transactions in `Transactions.csv` had `order_value == 0` or missing amounts due to promotional vouchers, complimentary items, or system recording anomalies.
- For each zero-amount transaction, item components were resolved against the catalog price list.
- Imputed order values were calculated as the sum of catalog item prices.
- As a result, 100% of the 53,820 transactions have strictly positive order values ($V_i > 0$).
- Post-imputation distribution:
  - Mean order value: **Rp 25,261**
  - Median order value: **Rp 23,000**
  - Modal single item: **Rp 20,000**

### 3.3 Meal Slot Allocation
To analyze intraday operational dynamics, each transaction's local time (`Asia/Jakarta`, WIB UTC+7) was assigned to an operational meal window:
- **Breakfast (07:00 – 11:00):** 11,169 orders (20.8%) — Morning coffee and quick bakery breakfast.
- **Lunch (11:00 – 15:00):** 12,752 orders (23.7%) — Main rice bowls (`Menu Gedhe`), lunch dining.
- **Afternoon Snacks (15:00 – 18:00):** 10,307 orders (19.2%) — Pastries (`Donat Polos`, croissants), teas, and cold refreshments.
- **Dinner (18:00 – 23:00):** 19,592 orders (36.4%) — Evening social visits, dining, and late coffee.
- **Late Night (23:00 – 07:00):** 0 orders (0.0%) — Verified kitchen closing and restocking hours.

---

## 4. Menu Architecture & Weather Affinities (T07)

The menu catalog (`data/raw/menu.csv`) encompasses 96 distinct products categorized into 6 operational categories with weather affinity metadata:

| Category | Item Count | Share of Items Sold | Typical Menu Items | Price Range (IDR) | Weather Affinity Tag |
|---|---|---|---|---|---|
| **Coffee** | 35 | 36.2% | Friendly Coffee (Ice/Hot), Espresso Houseblend, Americano, Latte, Mocha | Rp 15,000 – Rp 28,000 | `hot_comfort` (hot brews) / `cold_refresh` (iced brews) |
| **Menu Gedhe** | 18 | 20.3% | Signature Rice Bowls, Savory Chicken & Beef Mains, Teriyaki, Katsu | Rp 22,000 – Rp 40,000 | `neutral` |
| **Other** | 16 | 16.1% | Artisan Drinks, Black Rice, Syrups, Mineral Water, House Sodas | Rp 5,000 – Rp 25,000 | `neutral` / `cold_refresh` |
| **Tea** | 13 | 13.7% | Artisan Tea Blends, Moody, Lemon Tea, Lychee Tea, Green Tea | Rp 12,000 – Rp 22,000 | `hot_comfort` / `cold_refresh` |
| **Snacks** | 9 | 9.4% | Donat Polos, Croissant, French Fries, Fried Platters | Rp 10,000 – Rp 22,000 | `hot_comfort` (warm baked goods) |
| **Add on** | 5 | 4.3% | Extra Shot Espresso, Flavor Syrups, Oat Milk, Toppings | Rp 3,000 – Rp 8,000 | `neutral` |

---

## 5. Output Datasets Generated

The preprocessing module (`src/preprocess.py`) compiles the raw data into standardized tables:

1. `data/raw/menu.csv`:
   Cleaned 96-item menu catalog with SKU codes, categories, IDR prices, and weather affinity tags.
2. `data/raw/orders.csv`:
   53,820 order-level transactions with fields:
   `order_id`, `date`, `time`, `datetime`, `outlet_id`, `items`, `item_count`, `order_value`, `meal_slot`.
3. `data/raw/order_items.csv`:
   61,973 individual item rows linking `order_id`, `date`, `item_name`, `category`, `price`, `weather_affinity`.
4. `data/raw/daily_orders.csv`:
   272 daily rows aggregated from `orders.csv` joined with daily weather metrics:
   `date`, `order_count`, `total_revenue`, `mean_order_val`, and all 10 Open-Meteo weather variables.

---

## 6. Verification and Offline Tests

The pipeline includes comprehensive offline unit tests in `tests/`:
- `tests/test_menu.py`: Validates schema, pricing boundaries, non-null categories, and unique product identifiers (5/5 passed).
- `tests/test_weather.py`: Validates Surabaya weather bounds, date continuity, and hourly aggregations (7/7 passed).
- `tests/test_features.py`: Validates Indonesian holiday integration, lag computation without data leakage, and NW monsoon phase assignment (26/26 passed).
- `tests/test_eda.py`: Validates statistical summaries, category aggregations, and presentation figure outputs (7/7 passed).
- `tests/test_predict.py`: Validates chronological train/test splitting, baseline models, Ridge, and XGBoost evaluations (7/7 passed).

All 62 tests run and pass cleanly via `pytest tests/`.
