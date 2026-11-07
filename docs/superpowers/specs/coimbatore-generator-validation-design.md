# Design: evidence-based Coimbatore order generator and its validation

Branch: `coimbatore/review1-validation` (from `db44213`, the Review 1 state before the Surabaya migration)
Owner: P5 (Deepak)

## Why

Faculty asked what the synthetic generator is based on. The old generator (`src/generator.py` at `db44213`) used hand-picked
multipliers. This rebuild ties every parameter to evidence and then shows the data reproduces that evidence.

## Evidence (inputs)

| Source | What it is | Cleaned file |
|---|---|---|
| Swiggy order report, Hotel Chinthamani, Gopalapuram | 464 delivered orders, 727 item lines, 2026-08-05 to 2026-09-30 | `data/real/chinthamani_orders.csv`, `data/real/chinthamani_items.csv` |
| Google Form "Weather & Food Ordering Habits - Coimbatore" | 28 responses, 27 adults kept (one under-18 excluded) | `data/real/survey_responses.csv` |
| Open-Meteo archive | Real Coimbatore weather, 2023-09-24 to 2026-09-24 | `data/raw/weather.csv` (unchanged) |
| Tamil Nadu festival calendar | 41 dates | `COIMBATORE_FESTIVALS` in `src/generator.py` (unchanged) |
| Coimbatore outlet menus (Swiggy, Zomato, magicpin) | Snack and drink prices with outlet, URL, date checked | `data/reference/snack_drink_prices.csv` |

The original `.xlsx` files stay outside git. Swiggy order IDs are replaced by a running number.

## Generator (same output files and columns as before)

Period: 2023-09-24 to 2026-09-24, 1,097 days. Seed 42.

**Daily volume.** `order_count ~ NegativeBinomial(mean = mu_t, r)` with
`mu_t = BASE x weekend x rain(weekday/weekend) x heat x festival`.

| Parameter | Source |
|---|---|
| Weekend lift (Fri-Sun) | Poisson GLM on Swiggy daily counts (51 days with weather) |
| Rain on weekdays, rain on weekends (separate) | Same GLM, rain x weekend interaction (decision: option 1) |
| Heat (Tmax >= 33 C) | Same GLM; survey Q7 as direction check |
| Festival | Survey Q19 net share (Swiggy window has only one festival day) |
| Dispersion r | Swiggy residual over-dispersion per outlet, times number of outlets |
| BASE | Swiggy weekday-dry mean x `OUTLETS` scale, so volume stays near the old ~150-180/day |

**Order content.** Each order is either a food order or a snack/drink order.

| Part | Source |
|---|---|
| P(snack/drink order \| regime) | Survey Q9-Q11: share of snack/drink mentions per weather |
| Food basket | Bootstrap of the real Swiggy baskets (real items, quantities, prices, pairings) |
| Soup/rasam tilt on rainy days | Swiggy soup share rainy vs dry |
| Snack/drink item weights per regime | Survey Q9-Q11 mentions (+1 smoothing) |
| Snack/drink basket size | Survey Q18 |
| Planted pairs and P(add-on \| trigger, regime) | Survey Q12-Q16 mean frequency score / 4 |
| Snack/drink prices | `data/reference/snack_drink_prices.csv` (real outlets) |
| Order hour | Swiggy hourly profile (11:00-23:00); meal slot derived from hour |

Planted pairs: chai -> pakora, chai -> samosa, coffee -> vada (rainy > dry), ice cream -> cold coffee (hot > normal > rainy),
biryani -> buttermilk (flat across weather, as the survey shows).

## Validation (`src/validation.py`, `notebooks/08_validation.ipynb`, `docs/validation.md`)

1. **Grounding** on the 51 overlap days: hour profile, food basket size, food bill, food item shares, weekday/weekend ratio vs Swiggy
   (chi-square / KS). Pass = no significant difference at 5%, or a stated reason.
2. **Survey direction**: every volume effect and every pair moves the way the survey says. Pass = same sign.
3. **Recovery**: refit the GLM and the pair confidences on the synthetic data; planted values fall inside their 95% intervals.

Output: `results/validation/validation_summary.csv` with one pass/fail row per check, plus figures.

## Out of scope

Re-running notebooks 03/04/05/07 and the Review 1 deck. Those follow once the validation is reviewed.

## Known limits

The survey is small (27 adults, mostly students). The Swiggy data is one restaurant over 57 days, and it has no snacks or drinks.
The volume is a scale-up of that one outlet. Snack orders use the restaurant's hour profile.
