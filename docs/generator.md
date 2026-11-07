# Synthetic Order Generator Documentation

**Project:** Weather-Driven Food Order Patterns (Coimbatore) · BA Capstone, Team of 5
**Rubric line:** *Problem statement, business objective, and dataset description*
**Tasks:** T07 (menu), T08 (daily demand), T09 (baskets), T10 (documentation). Rebuilt so that every
parameter traces back to evidence. Validation: `docs/validation.md`.

---

## 1. Why a generator, and what it is based on

No public dataset gives three years of Coimbatore food orders with timestamps and baskets, and the course rules out pre-built
public datasets. So the team generates the orders, but the generator does **not** use hand-picked numbers. Every rule is taken
from one of four evidence sources:

| Source | What it is | File in the repo |
|---|---|---|
| **Swiggy order report, Hotel Chinthamani** (Gopalapuram, near the railway station) | 464 delivered orders, 727 item lines, 5 Aug - 30 Sep 2026, exported from the restaurant's Swiggy partner dashboard | `data/real/chinthamani_orders.csv`, `data/real/chinthamani_items.csv` |
| **Team survey** "Weather & Food Ordering Habits - Coimbatore" (Google Form, 21 questions) | 28 responses; 27 adults kept (one under-18 removed) | `data/real/survey_responses.csv` |
| **Coimbatore outlet menus** (Zomato, Swiggy, magicpin) | Snack and drink prices with outlet, URL and date checked | `data/reference/snack_drink_prices.csv` |
| **Open-Meteo archive** | Real daily Coimbatore weather, 2023-09-24 to 2026-09-24 | `data/raw/weather.csv` |

The original Excel exports are not committed. The cleaning scripts (`src/real_orders.py`, `src/survey.py`) replace Swiggy order
IDs with a running number and drop the restaurant address, taxes and discounts.

All parameters are computed by `src/calibration.py` when the generator runs. Nothing is typed in by hand except the festival
calendar, the weather thresholds, the seed and the outlet scale.

## 2. Daily order volume (T08)

$$Y_t \sim \text{NegativeBinomial}(r = 20 \times r_{\text{outlet}},\ \text{mean } \mu_t)$$

$$\mu_t = 20 \times \text{BASE} \times M_{\text{weekend}} \times M_{\text{rain}} \times M_{\text{heat}} \times M_{\text{festival}}$$

| Parameter | Value | Where it comes from |
|---|---|---|
| BASE (orders per outlet on a dry weekday) | 5.28 | Swiggy: quasi-Poisson GLM `n ~ weekend * rainy + hot` on the 51 days that have weather (95% CI 3.65-7.63) |
| Weekend (Fri-Sun) | x2.47 | Same GLM (CI 1.61-3.78). Survey Q8 agrees: 17 order more at weekends, 3 less |
| Rain (>= 2.5 mm) on a weekday | x1.56 | Same GLM (CI 1.00-2.41) |
| Rain on a weekend | x0.54 | Same GLM, rain x weekend interaction (CI 0.35-0.82). Survey Q5 agrees on the direction (18 of 27 order less in rain) |
| Heat (max >= 33 °C) | x1.07 | Same GLM (CI 0.73-1.57). Survey Q7: 7 more, 5 less |
| Festival day | x0.89 | Survey Q19: 1 + (8 more - 11 less) / 27. The Swiggy window has only one festival day |
| r per outlet (over-dispersion) | 6.9 | Pearson over-dispersion of the Swiggy GLM. 20 independent outlets add up to r = 20 x 6.9 |
| Outlets | 20 | Scale choice: one Chinthamani-sized outlet x 20 keeps volume near the old dataset (about 167 orders a day) |

Rain is treated per day type (team decision, "option 1"): rain raises orders on weekdays and lowers them on weekends. Pooled over
all days, rainy days have about 11% fewer orders than dry days, as in the real Swiggy data (7.8 vs 9.0 a day), and rain flattens
the weekly cycle. Heavy rain (>= 15 mm) is kept as a flag only: the Swiggy window has three heavy-rain days, too few for its own
effect.

Weather regime per day: `rainy` if precipitation >= 2.5 mm, else `hot` if max temperature >= 33 °C, else `normal`.

### Festival calendar

41 Tamil Nadu / Coimbatore festival dates (Ayudha Puja, Vijaya Dashami, Deepavali, Christmas, New Year's Eve and Day, Thai /
Mattu / Kaanum Pongal, Eid al-Fitr, Tamil New Year, Bakrid, Vinayagar Chaturthi, Milad-un-Nabi) are listed in
`COIMBATORE_FESTIVALS` in `src/generator.py`.

## 3. Orders and baskets (T09)

Each order is either a **food order** or a **snack/drink order**.

| Step | Rule | Where it comes from |
|---|---|---|
| Order time | Hour drawn from the real hourly profile (11:00-23:00, lunch peak at 12-13), minute and second uniform | Swiggy order times |
| Meal slot | Lunch 11-15, Evening Snacks 16-18, Dinner 19-23 (no breakfast: the real outlet opens at 11) | Derived from the hour |
| Snack/drink share | 35% of orders on rainy days, 54% on hot days, 18% on normal days | Survey Q9-Q11: share of snack/drink items among all items named per weather |
| Food basket | One of the 464 real Swiggy baskets, resampled: real items, quantities and line prices | Swiggy item lines |
| Rainy-day soup tilt | Baskets with a soup or rasam are x1.11 more likely on rainy days | Swiggy: soup share on rainy vs dry days |
| Snack/drink item mix | Weights per weather from survey mentions (+1 smoothing). Rainy: coffee, samosa, chai, bajji lead. Hot: ice cream, lime juice, cold coffee | Survey Q9-Q11 |
| Snack/drink basket size | 1 item 33%, 2 items 48%, 3 items 11%, 4+ items 7% | Survey Q18 |
| Prices | Food: median real unit price. Snacks/drinks: median of the outlet listings | Swiggy; `snack_drink_prices.csv` |

### Planted pairs (ground truth for association rules)

When a trigger is in a basket, its add-on is present with exactly the probability below, so `P(add-on | trigger, weather)`
equals the evidence. The probability is the mean survey frequency score (Never 0 ... Always 4) divided by 4. Questions asked as
"rainy vs dry" use the dry answer for both normal and hot days.

| Trigger -> Add-on | Rainy | Normal | Hot | Survey question |
|---|---|---|---|---|
| Chai -> Pakora | **0.30** | 0.22 | 0.22 | Q12 |
| Chai -> Samosa | **0.32** | 0.25 | 0.25 | Q13 |
| Coffee -> Vada | **0.31** | 0.27 | 0.27 | Q14 |
| Ice Cream -> Cold Coffee | 0.23 | 0.28 | **0.31** | Q16 |
| Any biryani -> Buttermilk | 0.30 | 0.30 | 0.30 | Q15 (the survey shows no weather effect, so none is planted) |

The weather shift in item mix adds to these: chai, samosa and pakora are far more common on rainy days and ice cream and cold
drinks on hot days, so rainy-day support for the chai pairs and hot-day support for ice cream -> cold coffee are several times
higher than on other days.

### Menu (`data/raw/menu.csv`, 80 items)

70 real Chinthamani dishes (categories: Starters & Gravies, Biryani & Rice, Breads & Tiffin, Chinese, Soups, Egg, Meals) and 10
snacks and drinks (Chai, Coffee, Samosa, Pakora, Bajji, Vada, Ice Cream, Cold Coffee, Fresh Lime Juice, Buttermilk). Soups and hot
snacks/drinks are tagged `hot_comfort`, cold drinks and ice cream `cold_refresh`, everything else `neutral`.

Two snack prices use a stand-in because no Coimbatore listing could be found: Pakora uses Onion Bajji (₹36, Shree Anandhaas) and
Buttermilk uses Salted Lassi (₹48, Sree Annapoorna). Both are marked `STAND-IN` in the price file.

## 4. Output files (same columns as before)

| File | Rows | Columns |
|---|---|---|
| `data/raw/daily_orders.csv` | 1,097 | weather columns, `dow, is_weekend, is_rainy, heavy_rain, is_hot, regime, festival_name, is_festival, mult_weekend, mult_rain, mult_temp, mult_festival, expected_orders, order_count` |
| `data/raw/orders.csv` | 182,696 | `order_id, date, timestamp, meal_slot, items, item_count, order_value` |
| `data/raw/order_items.csv` | 315,829 | one row per item line (not committed; regenerate) |
| `data/raw/menu.csv` | 80 | `item_id, item_name, category, price, weather_affinity, description` |

A snack/drink order holds only snack/drink items; a food order always contains a real dish (`generator.is_snack_order`).

## 5. Reproduce

```
python src/real_orders.py --report "../report chinthamani.xlsx"        # only if the raw export changes
python src/survey.py --responses "../Weather & Food Ordering Habits - Coimbatore (Responses).xlsx"
python src/generator.py                                                 # seed 42, about 5 seconds
python src/validation.py                                                # checks, results/validation/
```

## 6. Limits

* The Swiggy data is one restaurant over 57 days in the monsoon months, with no snacks or drinks. Effects estimated from it have
  wide intervals (see the CIs above).
* The survey is small (27 adults, mostly students who order rarely). It sets directions and pair strengths, not exact sizes.
* Snack/drink orders reuse the restaurant's hour profile because no snack-specific timing evidence exists.
* Volume is a scale-up of one outlet, not a measured city total.
