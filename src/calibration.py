"""Derive every generator parameter from the real evidence.

Sources (see docs/generator.md, section "What the generator is based on"):
    Swiggy report (Hotel Chinthamani)  -> volume effects, dispersion, hours, food baskets, soup tilt
    Survey (27 adult respondents)      -> festival effect, snack/drink share and mix, pairs, snack basket size
    Coimbatore outlet menus            -> snack/drink prices

`calibrate()` returns one frozen `Calibration` that src/generator.py consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import real_orders
import survey as survey_mod

ROOT = Path(__file__).resolve().parents[1]
WEATHER_CSV = ROOT / "data" / "raw" / "weather.csv"
PRICES_CSV = ROOT / "data" / "reference" / "snack_drink_prices.csv"

RAIN_MM = 2.5  # same thresholds as the old generator and docs/dataset.md
HOT_C = 33.0
WEEKEND_DAYS = (4, 5, 6)  # Fri-Sun, as asked in survey Q8
REGIMES = ("rainy", "hot", "normal")

SNACK_ITEMS = (
    "Chai", "Coffee", "Samosa", "Pakora", "Bajji", "Vada",
    "Ice Cream", "Cold Coffee", "Fresh Lime Juice", "Buttermilk",
)
REGIME_QUESTIONS = {"rainy": "rainy_items", "hot": "hot_items", "normal": "normal_items"}

# (trigger, add-on) -> survey question prefix (Q12-Q16)
PAIR_QUESTIONS: dict[tuple[str, str], str] = {
    ("Chai", "Pakora"): "chai_pakora",
    ("Chai", "Samosa"): "chai_samosa",
    ("Coffee", "Vada"): "coffee_vada",
    ("Ice Cream", "Cold Coffee"): "icecream_coldcoffee",
    ("Biryani", "Buttermilk"): "biryani_buttermilk",
}
LIKERT_MAX = max(survey_mod.LIKERT.values())
FORMULA = "n ~ is_weekend * is_rainy + is_hot"
NO_OVERDISPERSION_R = 1e6


@dataclass(frozen=True)
class Calibration:
    volume: dict[str, dict[str, float]]
    dispersion_r: float
    festival: float
    hour_profile: dict[int, float]
    food_baskets: tuple[tuple[tuple[str, float], ...], ...]
    soup_tilt: float
    snack_share: dict[str, float]
    snack_weights: dict[str, dict[str, float]]
    snack_basket_probs: dict[int, float]
    pair_probs: dict[tuple[str, str], dict[str, float]]
    snack_prices: dict[str, float]


# -----------------------------------------------------------------------------
# Volume (Swiggy daily counts)
# -----------------------------------------------------------------------------
def real_daily(orders: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Daily order counts joined to weather, for days covered by both sources."""
    start, end = orders["date"].min(), orders["date"].max()
    w = weather[(weather["date"] >= start) & (weather["date"] <= end)].copy()
    counts = orders.groupby("date").size()
    dow = pd.to_datetime(w["date"]).dt.day_of_week
    return pd.DataFrame({
        "date": w["date"].to_numpy(),
        "n": w["date"].map(counts).fillna(0).astype(int).to_numpy(),
        "is_weekend": dow.isin(WEEKEND_DAYS).astype(int).to_numpy(),
        "is_rainy": (w["precip_mm"] >= RAIN_MM).astype(int).to_numpy(),
        "is_hot": (w["temp_max_c"] >= HOT_C).astype(int).to_numpy(),
    })


def _formula(days: pd.DataFrame) -> str:
    return FORMULA if days["is_hot"].nunique() > 1 else FORMULA.replace(" + is_hot", "")


def glm_effect(fit, terms: list[str]) -> dict[str, float]:
    """exp of a sum of coefficients, with a 95% interval from the covariance matrix."""
    weights = np.array([1.0 if name in terms else 0.0 for name in fit.params.index])
    est = float(weights @ fit.params.to_numpy())
    se = float(np.sqrt(weights @ fit.cov_params().to_numpy() @ weights))
    return {"value": np.exp(est), "low": np.exp(est - 1.96 * se), "high": np.exp(est + 1.96 * se)}


def fit_volume_effects(days: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Quasi-Poisson GLM of daily counts; returns multipliers with 95% intervals."""
    fit = smf.glm(_formula(days), data=days, family=sm.families.Poisson()).fit(scale="X2")
    no_heat = {"value": 1.0, "low": 1.0, "high": 1.0}
    return {
        "base_weekday_dry": glm_effect(fit, ["Intercept"]),
        "weekend": glm_effect(fit, ["is_weekend"]),
        "rain_weekday": glm_effect(fit, ["is_rainy"]),
        "rain_weekend": glm_effect(fit, ["is_rainy", "is_weekend:is_rainy"]),
        "heat": glm_effect(fit, ["is_hot"]) if "is_hot" in fit.params else no_heat,
    }


def outlet_dispersion(days: pd.DataFrame) -> float:
    """Negative-binomial r for one outlet, from the GLM's Pearson over-dispersion."""
    terms = [t for t in ("is_weekend", "is_rainy", "is_hot") if days[t].nunique() > 1]
    formula = "n ~ " + (" * ".join(terms[:2]) + "".join(f" + {t}" for t in terms[2:]) if terms else "1")
    fit = smf.glm(formula, data=days, family=sm.families.Poisson()).fit()
    phi = fit.pearson_chi2 / fit.df_resid
    if phi <= 1:
        return NO_OVERDISPERSION_R
    return float(fit.fittedvalues.mean() / (phi - 1))


# -----------------------------------------------------------------------------
# Survey-derived parameters
# -----------------------------------------------------------------------------
def festival_effect(survey: pd.DataFrame) -> float:
    """1 + (share ordering more - share ordering less) on festival days (Q19)."""
    d = survey_mod.volume_direction(survey, "festival_volume")
    n = sum(d.values())
    return 1 + (d["more"] - d["less"]) / n


def snack_share(survey: pd.DataFrame) -> dict[str, float]:
    """Share of item mentions that are snacks/drinks, per weather (Q9-Q11)."""
    share = {}
    for regime, col in REGIME_QUESTIONS.items():
        mentions = survey_mod.item_mentions(survey, col)
        share[regime] = float(mentions.reindex(SNACK_ITEMS).fillna(0).sum() / mentions.sum())
    return share


def snack_weights(survey: pd.DataFrame) -> dict[str, dict[str, float]]:
    """P(snack/drink item | regime) from Q9-Q11 mentions, with +1 smoothing."""
    weights = {}
    for regime, col in REGIME_QUESTIONS.items():
        counts = survey_mod.item_mentions(survey, col).reindex(SNACK_ITEMS).fillna(0) + 1
        weights[regime] = (counts / counts.sum()).to_dict()
    return weights


def pair_probabilities(survey: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """P(add-on | trigger, regime) = mean frequency score / 4 (Q12-Q16).

    Questions asked as rainy vs dry apply the dry answer to both normal and hot days.
    """
    scores = survey_mod.combo_scores(survey)
    probs = {}
    for pair, question in PAIR_QUESTIONS.items():
        rows = scores[scores["combo"] == question].set_index("condition")["mean_score"] / LIKERT_MAX
        if rows.empty:
            continue
        if "dry" in rows.index:
            probs[pair] = {"rainy": rows["rainy"], "normal": rows["dry"], "hot": rows["dry"]}
        else:
            probs[pair] = {r: rows[r] for r in REGIMES}
    return probs


def snack_basket_probs(survey: pd.DataFrame) -> dict[int, float]:
    """Items per snack/drink order from Q18 ('4 or more' counted as 4)."""
    sizes = survey["basket_size"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    return {int(k): float(v) for k, v in sizes.value_counts(normalize=True).sort_index().items()}


# -----------------------------------------------------------------------------
# Swiggy order content
# -----------------------------------------------------------------------------
def hour_profile(orders: pd.DataFrame) -> dict[int, float]:
    """Share of orders placed in each clock hour."""
    shares = orders["hour"].value_counts(normalize=True).sort_index()
    return {int(h): float(s) for h, s in shares.items()}


def food_baskets(items: pd.DataFrame) -> list[tuple[tuple[str, float], ...]]:
    """Each real order as a tuple of (item name, line price)."""
    return [
        tuple(zip(group["item_name"], group["line_price"].astype(float)))
        for _, group in items.groupby("order_no", sort=True)
    ]


def is_soup(name: str) -> bool:
    return "soup" in name.lower() or "rasam" in name.lower()


def soup_tilt(orders: pd.DataFrame, items: pd.DataFrame, weather: pd.DataFrame) -> float:
    """Share of orders with a soup/rasam on rainy days divided by the share on dry days."""
    soup_orders = set(items.loc[items["item_name"].map(is_soup), "order_no"])
    rainy_dates = set(weather.loc[weather["precip_mm"] >= RAIN_MM, "date"])
    known = orders[orders["date"].isin(set(weather["date"]))]
    has_soup = known["order_no"].isin(soup_orders)
    on_rain = known["date"].isin(rainy_dates)
    return float(has_soup[on_rain].mean() / has_soup[~on_rain].mean())


def median_prices(prices: pd.DataFrame) -> dict[str, float]:
    return {k: float(v) for k, v in prices.groupby("item_name")["price"].median().items()}


# -----------------------------------------------------------------------------
# Everything together
# -----------------------------------------------------------------------------
def calibrate(
    weather_csv: str | Path = WEATHER_CSV,
    prices_csv: str | Path = PRICES_CSV,
) -> Calibration:
    orders, items = real_orders.load_clean()
    responses = survey_mod.load_clean()
    weather = pd.read_csv(weather_csv)
    prices = pd.read_csv(prices_csv)

    missing = set(SNACK_ITEMS) - set(prices["item_name"])
    if missing:
        raise ValueError(f"No sourced price for: {sorted(missing)}")

    days = real_daily(orders, weather)
    return Calibration(
        volume=fit_volume_effects(days),
        dispersion_r=outlet_dispersion(days),
        festival=festival_effect(responses),
        hour_profile=hour_profile(orders),
        food_baskets=tuple(food_baskets(items)),
        soup_tilt=soup_tilt(orders, items, weather),
        snack_share=snack_share(responses),
        snack_weights=snack_weights(responses),
        snack_basket_probs=snack_basket_probs(responses),
        pair_probs=pair_probabilities(responses),
        snack_prices=median_prices(prices),
    )
