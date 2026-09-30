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
| Daily order counts | Team's order generator (`src/generator.py`, documented in `docs/generator.md`) | **Synthetic** |
| Order baskets / items | Same generator | **Synthetic** |

No pre-built dataset from Kaggle, UCI, GitHub or a similar repository is used. Weather comes from an API and orders are produced by a
documented synthetic-data generator, which the course rubric permits provided the method is documented.

### 3.1 Weather collection method

* **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
* **Location:** latitude 11.00, longitude 76.99 (Coimbatore). Open-Meteo snaps this to its nearest grid cell; the cell used for the
  data currently in the repository is 11.0018 N, 76.9947 E at 422 m elevation (see `data/raw/weather_meta.json`).
* **Period:** 2025-09-24 to 2026-09-24 inclusive, 366 days.
* **Daily variables:** `precipitation_sum`, `rain_sum`, `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`,
  `wind_speed_10m_max`, `weather_code`, `daylight_duration`.
* **Hourly variables:** `relative_humidity_2m`, `temperature_2m`, `precipitation`. Daily humidity is the mean of the 24 hourly values.
* **Time zone:** the fetcher requests `timezone=Asia/Kolkata`, so a day is an Indian calendar day.
* **Script:** `python src/weather.py` (cached and rerunnable; `--force` refetches). Every run records the exact request URL,
  retrieval time, grid point and validation result in `data/raw/weather_meta.json`.

> **Status of the data currently in the repository.** The API could not be reached from the environment where the fetcher was written,
> so `data/raw/weather.csv` was produced by normalising the CSV downloaded by hand from open-meteo.com for the same location and dates
> (`python src/weather.py --from-export data/raw/openmeteo_export_11.00N76.99E.csv`). That export is aggregated over **GMT** days and has
> **no humidity or rain duration**. Running `python src/weather.py --force` on a networked machine replaces it with Indian-day aggregates and adds
> `humidity_mean_pct`, `precip_hours` and the hourly file. Rainfall totals per day will shift slightly when this is done, so **regenerate the synthetic
> orders after the refresh and before anyone finalises results**; otherwise orders will be tied to the old day boundaries.

### 3.2 Order generation (summary)

Orders are synthetic. Daily volume is negative-binomial around a base level with multipliers for weekend days, rain and heat; item
baskets are drawn with weather-dependent category weights and planted item pairs (for example chai and pakora are more likely to be
ordered together on rainy days). The full parameter list and random seed are in `docs/generator.md` (owner: P2). Because the rules are
known, the notebooks can check whether each method recovers them.

## 4. Data files

| File | Rows | Description | Owner |
|---|---|---|---|
| `data/raw/weather.csv` | 366 | one row per day, all weather variables | P1 |
| `data/raw/weather_hourly.csv` | 8,784 (after API refresh) | hourly humidity, temperature, precipitation | P1 |
| `data/raw/weather_meta.json` | – | request, grid cell, units, validation | P1 |
| `data/raw/openmeteo_export_11.00N76.99E.csv` | 366 | unmodified manual download used for the first load | P1 |
| `data/raw/orders.csv` | – | synthetic orders (final file from P2) | P2 |
| `data/raw/daily_orders_p2_draft.csv` | 366 | draft daily order counts with weather columns, used until P2's final file lands | P2 |

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
| `precip_hours` | hours | number of hours of the day with measurable rain (> 0 mm): how long it rained (present after the API refresh) | derived from hourly `precipitation` |
| `humidity_mean_pct` | % | mean relative humidity (present after the API refresh) | mean of hourly `relative_humidity_2m` |

Derived flags used across the project (defined once so every module agrees):

| Flag | Rule |
|---|---|
| `is_rainy` | `precip_mm >= 2.5` (meaningful rain, not trace drizzle) |
| `heavy_rain` | `precip_mm >= 15` |
| `is_hot` | `temp_max_c >= 33` |
| `regime` | `rainy` if `is_rainy`, else `hot` if `is_hot`, else `normal` |

## 6. Profile of the weather data (2025-09-24 to 2026-09-24)

| Measure | Value |
|---|---|
| Days | 366, no gaps, no missing values |
| Total precipitation | 999.9 mm; 125 days with none |
| Rainy days (≥ 2.5 mm) | 108; heavy-rain days (≥ 15 mm) 15 |
| Max temperature | 24.0 – 39.3 °C, mean 31.7 °C; 106 days ≥ 33 °C |
| Daily orders (draft generator run) | 63,458 orders over 366 days, mean 173.4 per day, range 82 – 366 |
| Regime split | 108 rainy, 87 hot (and not rainy), 171 normal |

Wettest months in the window are October (231 mm) and August (162 mm); February to April are nearly dry.

## 7. Quality checks

`src/weather.py` validates every load: contiguous dates, no duplicates, no nulls, non-negative rainfall, `temp_max >= temp_min`,
requested start and end dates present, and 24 hourly values per day when hourly data is fetched. `tests/test_weather.py` covers the
parser, the validator, the cache and the manual-export path, and checks that the weather columns match those in the generator's draft
output exactly. Run `pytest -q`.

## 8. Limitations

* One year of history: a yearly cycle cannot be estimated, only weekly patterns and one monsoon season.
* Open-Meteo archive values are model reanalysis on a grid, not readings from a station in the city.
* Orders are synthetic; conclusions describe the method's ability to recover the generator's rules, not real customer behaviour.
* The 2.5 mm, 15 mm and 33 °C thresholds are project conventions, chosen for the generator, not meteorological standards.
