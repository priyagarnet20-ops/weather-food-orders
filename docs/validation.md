# Validation of the synthetic order dataset

**Question (faculty):** *What is the synthetic generator based on, and how do we know the data is right?*
**Short answer:** every generator rule is taken from real Swiggy orders, the team survey, real outlet prices or real weather
(`docs/generator.md`, section 1). 26 checks then test the generated data against that evidence. **25 pass.** The one that fails
is a known disagreement between the two evidence sources, explained below.

Run: `python src/validation.py` (table only) or `notebooks/08_validation.ipynb` (table and figures). Outputs are in
`results/validation/`: `validation_summary.csv`, `volume_effects.csv`, `pair_rules_by_regime.csv` and `fig_v1`-`fig_v5`.

## 1. Grounding: does the synthetic data look like the real orders?

Food orders on the 51 days covered by both the Swiggy report and the weather file (5 Aug - 24 Sep 2026).

| Check | Result | Test | Pass |
|---|---|---|---|
| Order hour profile | chi² = 1.2, p = 1.00 | chi-square goodness of fit | ✅ |
| Items per food order | chi² = 0.3, p = 0.96 | chi-square goodness of fit | ✅ |
| Food order value | D = 0.015, p = 1.00 | two-sample Kolmogorov-Smirnov | ✅ |
| Food item mix | chi² = 3.4, p = 1.00 | chi-square goodness of fit | ✅ |
| Day-to-day volume | r = 0.50, p < 0.001 | Pearson correlation, synthetic / 20 vs real | ✅ |

The first four agree closely because real baskets are resampled. The checks confirm nothing was lost on the way. The volume
check is the real test: the synthetic daily counts follow the outlet's real ups and downs. Both peak at weekends and both
dip on rainy weekends.

## 2. Survey: does every effect point the way respondents say?

| Effect | Survey (less / same / more) | Synthetic | Pass |
|---|---|---|---|
| Weekend | 3 / 7 / 17 -> more | x2.45 | ✅ |
| Rain on weekends | 18 / 5 / 4 -> less | x0.53 | ✅ |
| Rain on weekdays | 18 / 5 / 4 -> less | x1.56 | ❌ by design |
| Heat | 5 / 15 / 7 -> more | x1.06 | ✅ |
| Festival | 11 / 8 / 8 -> less | x0.87 | ✅ |
| Snack/drink share by weather | hot 0.54 > rainy 0.35 > normal 0.18 | hot 0.54 > rainy 0.34 > normal 0.18 | ✅ |
| Chai -> Pakora | rainy 0.30 > dry 0.22 | rainy 0.31 > normal 0.22 | ✅ |
| Chai -> Samosa | rainy 0.32 > dry 0.25 | rainy 0.32 > normal 0.23 | ✅ |
| Coffee -> Vada | rainy 0.31 > dry 0.27 | rainy 0.31 > normal 0.27 | ✅ |
| Ice Cream -> Cold Coffee | hot 0.31 > normal 0.28 > rainy 0.23 | hot 0.32 > normal 0.28 > rainy 0.24 | ✅ |
| Biryani -> Buttermilk | equal in all weather (0.30) | 0.28-0.30 (within 0.03) | ✅ |

**Why rain on weekdays fails.** The survey asked about rain in general, and most respondents said they order less. The real
Swiggy orders show more orders on rainy weekdays (x1.56) and fewer on rainy weekends (x0.54). The generator follows the
observed orders (team decision, option 1), so the weekend half agrees with the survey and the weekday half does not. A
plausible reading: on rainy working days people stay in and order, while at weekends rain cancels outings that would have
ended in a takeaway. With 17 rainy weekdays in the Swiggy window this is evidence, not proof.

## 3. Recovery: can a model find what was planted?

| Planted | Value | Recovered (95% CI) | Pass |
|---|---|---|---|
| Weekend | x2.47 | x2.45 (2.41-2.49) | ✅ |
| Rain, weekday | x1.55 | x1.56 (1.53-1.60) | ✅ |
| Rain, weekend | x0.54 | x0.53 (0.52-0.54) | ✅ |
| Heat | x1.07 | x1.06 (1.05-1.08) | ✅ |
| Festival | x0.89 | x0.87 (0.84-0.91) | ✅ |

The volume effects come from a quasi-Poisson GLM on the 1,097 synthetic days.

For all five planted pairs, the association-rule confidence matches the planted value in every weather type (within 2.94
standard errors, Bonferroni-corrected over the 15 pair x weather cells), and lift is above 1 everywhere (1.7 to 11). So
association-rule mining in Review 2 has real, documented patterns to find.

## 4. Limits

* **Swiggy data:** one restaurant, 57 days, monsoon months only, no snacks or drinks. Its effect estimates have wide
  intervals (weekend 1.6-3.8, rain on weekdays 1.0-2.4).
* **Survey:** 27 adults, mostly students who order rarely. It fixes directions and pair strengths, not sizes.
* **Prices:** pakora and buttermilk use the nearest real menu item as a stand-in.
* **Timing:** snack/drink orders use the restaurant's hourly profile.
* **Scale:** volume is one outlet scaled to 20 similar outlets, not a measured city total.
* **Circularity:** grounding and recovery show the generator is faithful to its evidence. They cannot show the evidence
  represents all of Coimbatore. That would need more restaurants and a larger survey.
