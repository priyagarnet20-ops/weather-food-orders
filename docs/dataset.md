# Dataset documentation

Project: Weather-Driven Food Order Patterns (Coimbatore) · BA capstone, team of 5
Author: P1 (Weather Data & Forecasting) · Rubric line: *Problem statement, business objective and dataset description*

---

## 1. Problem statement

Food-delivery demand in Coimbatore is not steady: it moves with the day of the week and, we expect, with the weather.
Kitchens and delivery fleets that plan around an average day are over-staffed on quiet dry days and short on rainy ones.
This project asks how much of that variation weather explains and whether it can be used to plan ahead.

## 2. Business objective

1. **Review 1:** predict the number of orders per day from weather and calendar features, and report which factors matter.
2. **Review 2, method 1 (association rules):** find which menu items are ordered together and whether those combinations
   differ between rainy and dry days (for example chai with pakora).
3. **Review 2, method 2 (time series):** forecast the next week's order volume so that kitchen staffing and stock can be
   planned, and quantify how much knowing the weather helps.
4. Turn the results into operational recommendations: rainy-day promotions, staffing buffers, and inventory for weather-linked items.

## 3. Data sources and how they were collected

| Part | Source | Real or synthetic |
|---|---|---|
| Weather (all columns in `weather.csv`, `weather_hourly.csv`) | Open-Meteo Historical Weather API (open-meteo.com) | **Real** (reanalysis data) |
| Daily order counts (`daily_orders.csv`) | Team's order generator (`src/generator.py`, documented in `docs/generator.md`) | **Synthetic** |
| Order baskets / items | Same generator | **Synthetic** |

No pre-built dataset from Kaggle, UCI, GitHub or a similar repository is used. Weather comes from an API and orders are produced by a
documented synthetic-data generator, which the course rubric permits provided the method is documented.

### 3.1 Weather collection method

* **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
* **Location:** latitude 11.00, longitude 76.99 (Coimbatore). Open-Meteo snaps this to its nearest grid cell; the cell used for the
  data in the repository is 11.0018 N, 76.9947 E at 409 m elevation (see `data/raw/weather_meta.json`).
* **Period:** 2023-09-24 to 2026-09-24 inclusive, 1,097 days (three years).
* **Daily variables:** `precipitation_sum`, `rain_sum`, `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`,
  `wind_speed_10m_max`, `weather_code`, `daylight_duration`.
* **Hourly variables:** `relative_humidity_2m`, `temperature_2m`, `precipitation`. Daily humidity is the mean of the 24 hourly values.
* **Time zone:** the fetcher requests `timezone=Asia/Kolkata`, so a day is an Indian calendar day.
* **Script:** `python src/weather.py` (cached and rerunnable; `--force` refetches). Every run records the exact request URL,
  retrieval time, grid point and validation result in `data/raw/weather_meta.json`.

> **Status of the data.** `data/raw/weather.csv` is the output of `python src/weather.py --force` run on 2026-09-30 (request URL and retrieval time
> are in `data/raw/weather_meta.json`): 1,097 daily rows aggregated over Indian calendar days, with `humidity_mean_pct` and `precip_hours` derived from
> 26,328 hourly values, and no validation problems. P2's `daily_orders.csv` was generated from this file, and its ten weather columns are identical to
> `weather.csv` (checked in `tests/test_weather.py`). If the weather is ever refetched, the orders must be regenerated from it.

### 3.2 Order generation (summary)

Orders are synthetic, but every generator parameter comes from evidence:

| Evidence | Used for | File |
|---|---|---|
| Swiggy order report, Hotel Chinthamani, Coimbatore (464 orders, Aug-Sep 2026) | food menu, prices, baskets, order hours, weekend and rain effects | `data/real/chinthamani_*.csv` |
| Team survey, 27 adult Coimbatore respondents | festival effect, snack/drink share and mix, weather-linked pairs | `data/real/survey_responses.csv` |
| Coimbatore outlet menus (Zomato, Swiggy, magicpin) | snack and drink prices | `data/reference/snack_drink_prices.csv` |

Daily volume is negative-binomial (seed 42). The multipliers come from a GLM on the real Swiggy days: weekend (Fri–Sun) x2.47, rain x1.56
on weekdays and x0.54 on weekends, heat x1.07. The festival multiplier (x0.89) comes from the survey. Food baskets are resampled real Swiggy
baskets. Snack and drink orders follow the survey's weather mix, with planted pairs (chai with pakora or samosa and coffee with vada are more
likely on rainy days; ice cream with cold coffee on hot days). Full parameters: `docs/generator.md`. Validation against the evidence (25 of
26 checks pass): `docs/validation.md`.

## 4. Data files

| File | Rows | Description | Owner |
|---|---|---|---|
| `data/raw/weather.csv` | 1,097 | one row per day, all weather variables | P1 |
| `data/raw/weather_hourly.csv` | 26,328 | hourly humidity, temperature, precipitation (written by the fetcher; not committed) | P1 |
| `data/raw/weather_meta.json` | – | request, grid cell, units, validation | P1 |
| `data/raw/openmeteo_export_11.00N76.99E.csv` | 366 | unmodified manual download from the first load (one year, GMT days); kept for the offline test only | P1 |
| `data/raw/daily_orders.csv` | 1,097 | final daily order counts with the weather columns, regime, festival fields, multipliers and expected orders | P2 |

## 5. Field dictionary: `weather.csv`

| Column | Unit | Meaning | Open-Meteo variable |
|---|---|---|---|
| `date` | date | calendar day | `time` |
| `precip_mm` | mm | total precipitation of the day (rain + showers + snow water equivalent) | `precipitation_sum` |
| `rain_mm` | mm | rain only | `rain_sum` |
| `temp_max_c` | °C | highest 2 m air temperature | `temperature_2m_max` |
| `temp_min_c` | °C | lowest 2 m air temperature | `temperature_2m_min` |
| `wind_max_kmh` | km/h | highest 10 m wind speed | `wind_speed_10m_max` |
| `weather_code` | WMO code | most severe condition of the day (0 clear; 1–3 mainly clear to overcast; 51–55 drizzle; 61–65 rain) | `weather_code` |
| `temp_mean_c` | °C | mean 2 m air temperature | `temperature_2m_mean` |
| `daylight_s` | s | seconds of daylight | `daylight_duration` |
| `precip_hours` | hours | number of hours of the day with measurable rain (> 0 mm): how long it rained  | derived from hourly `precipitation` |
| `humidity_mean_pct` | % | mean relative humidity  | mean of hourly `relative_humidity_2m` |

Derived flags used across the project (defined once so every module agrees):

| Flag | Rule |
|---|---|
| `is_rainy` | `precip_mm >= 2.5` (meaningful rain, not trace drizzle) |
| `heavy_rain` | `precip_mm >= 15` |
| `is_hot` | `temp_max_c >= 33` |
| `regime` | `rainy` if `is_rainy`, else `hot` if `is_hot`, else `normal` |

## 6. Profile of the data (2023-09-24 to 2026-09-24)

| Measure | Value |
|---|---|
| Days | 1,097, no gaps, no missing values |
| Total precipitation | 3,599 mm (518 mm in the 2023 part, 1,299 mm in 2024, 1,143 mm in 2025, 639 mm in the 2026 part); 387 days with none |
| Rainy days (≥ 2.5 mm) | 329; heavy-rain days (≥ 15 mm) 67 |
| Max temperature | 24.0 – 40.8 °C, mean 31.5 °C; 300 days ≥ 33 °C |
| Daily orders (current dashboard generator output) | 182,696 orders over 1,097 days, mean 166.5 per day, range 73 – 354 |
| Regime split | 329 rainy, 265 hot (and not rainy), 503 normal; 41 festival days |
| Mean orders by regime | rainy 153, hot 177, normal 169 |

The wettest months in the window are November 2023 (253 mm), May 2024 (244 mm), October 2025 (231 mm) and October 2024 (218 mm).

## 7. Quality checks

`src/weather.py` validates every load: contiguous dates, no duplicates, no nulls, non-negative rainfall, `temp_max >= temp_min`,
requested start and end dates present, and 24 hourly values per day when hourly data is fetched. `tests/test_weather.py` covers the
parser, the validator, the cache and the manual-export path, and checks that `weather.csv` and `daily_orders.csv` cover the same 1,097 days
with identical weather columns. Run `pytest -q`.

## 8. Limitations

* Three years of history, but the generator has no yearly effect beyond weather and festivals, so the forecasting models do not include a yearly term. On real data this would need to be tested.
* Festival days are treated as known in advance. Their dates are fixed in the generator's calendar.
* Open-Meteo archive values are model reanalysis on a grid, not readings from a station in the city.
* Orders are synthetic; conclusions describe the method's ability to recover the generator's rules, not real customer behaviour.
* The 2.5 mm, 15 mm and 33 °C thresholds are project conventions, chosen for the generator, not meteorological standards.
