"""Evidence-based weather-driven food order generator for Coimbatore (P2 / T08 & T09, rebuilt).

Every parameter comes from src/calibration.py, which derives it from:
    - the real Swiggy order report of Hotel Chinthamani, Coimbatore (data/real/)
    - the team's ordering-habits survey (data/real/survey_responses.csv)
    - snack/drink prices listed by Coimbatore outlets (data/reference/snack_drink_prices.csv)
The weather is real (Open-Meteo) and the festival calendar is the Tamil Nadu calendar below.

Models:
    1. Daily demand (T08)
       Y_t ~ NegativeBinomial(r = OUTLETS * r_outlet, mean mu_t)
       mu_t = OUTLETS * base * weekend * rain(weekday or weekend) * heat * festival

    2. Orders (T09)
       - hour from the real Swiggy hourly profile; meal slot derived from the hour
       - food order: a real Swiggy basket (soup baskets tilted up on rainy days),
         plus Buttermilk after Biryani with the survey probability
       - snack/drink order (share per weather from the survey): survey item mix and
         basket size, with the survey's weather-dependent pairs planted

Author: P2 (Order Generator & Menu Catalog); rebuilt by P5 for the evidence-based validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import calibration as C

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
OUTLETS = 20  # volume = one Chinthamani-like outlet scaled to 20 similar outlets
RAIN_TRACE_MM = C.RAIN_MM
RAIN_HEAVY_MM = 15.0  # flag only; the Swiggy window has too few heavy-rain days for its own effect
TEMP_HOT_C = C.HOT_C
WEEKEND_DAYS = C.WEEKEND_DAYS
SOURCE_REAL = "Hotel Chinthamani, Coimbatore (Swiggy, Aug-Sep 2026)"

COIMBATORE_FESTIVALS: dict[str, str] = {
    # 2023
    "2023-10-23": "Ayudha Puja",
    "2023-10-24": "Vijaya Dashami",
    "2023-11-12": "Deepavali",
    "2023-12-25": "Christmas",
    "2023-12-31": "New Year's Eve",
    # 2024
    "2024-01-01": "New Year's Day",
    "2024-01-15": "Thai Pongal",
    "2024-01-16": "Mattu Pongal",
    "2024-01-17": "Kaanum Pongal",
    "2024-04-11": "Eid al-Fitr (Ramzan)",
    "2024-04-14": "Tamil New Year (Puthandu)",
    "2024-06-17": "Bakrid (Eid al-Adha)",
    "2024-09-07": "Vinayagar Chaturthi",
    "2024-09-16": "Milad-un-Nabi",
    "2024-10-11": "Ayudha Puja",
    "2024-10-12": "Vijaya Dashami",
    "2024-10-31": "Deepavali",
    "2024-12-25": "Christmas",
    "2024-12-31": "New Year's Eve",
    # 2025
    "2025-01-01": "New Year's Day",
    "2025-01-14": "Thai Pongal",
    "2025-01-15": "Mattu Pongal",
    "2025-01-16": "Kaanum Pongal",
    "2025-03-31": "Eid al-Fitr (Ramzan)",
    "2025-04-14": "Tamil New Year (Puthandu)",
    "2025-06-07": "Bakrid",
    "2025-08-27": "Vinayagar Chaturthi",
    "2025-09-05": "Milad-un-Nabi",
    "2025-10-01": "Ayudha Puja",
    "2025-10-02": "Vijaya Dashami",
    "2025-10-20": "Deepavali",
    "2025-12-25": "Christmas",
    "2025-12-31": "New Year's Eve",
    # 2026
    "2026-01-01": "New Year's Day",
    "2026-01-14": "Thai Pongal",
    "2026-01-15": "Mattu Pongal",
    "2026-01-16": "Kaanum Pongal",
    "2026-03-20": "Eid al-Fitr",
    "2026-04-14": "Tamil New Year (Puthandu)",
    "2026-05-27": "Bakrid",
    "2026-09-14": "Vinayagar Chaturthi",
}

# Meal slot by clock hour (the real orders run 11:00-23:00, so there is no breakfast slot)
SLOT_BY_HOUR: dict[str, range] = {
    "Lunch": range(11, 16),
    "Evening Snacks": range(16, 19),
    "Dinner": range(19, 24),
}

SNACK_CATEGORIES: dict[str, tuple[str, str]] = {
    "Chai": ("Hot Beverages", "hot_comfort"),
    "Coffee": ("Hot Beverages", "hot_comfort"),
    "Samosa": ("Snacks", "hot_comfort"),
    "Pakora": ("Snacks", "hot_comfort"),
    "Bajji": ("Snacks", "hot_comfort"),
    "Vada": ("Snacks", "hot_comfort"),
    "Ice Cream": ("Cold Drinks & Desserts", "cold_refresh"),
    "Cold Coffee": ("Cold Drinks & Desserts", "cold_refresh"),
    "Fresh Lime Juice": ("Cold Drinks & Desserts", "cold_refresh"),
    "Buttermilk": ("Cold Drinks & Desserts", "cold_refresh"),
}

# Food categories by keyword, checked in this order (first match wins)
FOOD_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Soups", ("soup", "rasam")),
    ("Meals", ("meals",)),
    ("Biryani & Rice", ("biryani", "rice")),
    ("Breads & Tiffin", ("parotta", "naan", "roti", "chapati", "dosa", "dosai", "roast")),
    ("Chinese", ("noodles", "manchurian", "schezwan", "chilli", "dragon", "lollipop", "dynamite")),
    ("Egg", ("egg", "omelette", "kalakki", "podimas")),
)
DEFAULT_FOOD_CATEGORY = "Starters & Gravies"
FOOD_TRIGGERS = {"Biryani"}


# -----------------------------------------------------------------------------
# Menu
# -----------------------------------------------------------------------------
def food_category(name: str) -> str:
    lower = name.lower()
    for category, keywords in FOOD_CATEGORY_RULES:
        if any(k in lower for k in keywords):
            return category
    return DEFAULT_FOOD_CATEGORY


def build_menu(real_items: pd.DataFrame, snack_prices: dict[str, float]) -> pd.DataFrame:
    """Real Chinthamani food items (median unit price) plus the sourced snacks and drinks."""
    food = real_items.groupby("item_name")["unit_price"].median().round().astype(int)
    rows = [
        (name, food_category(name), price,
         "hot_comfort" if C.is_soup(name) else "neutral", f"Real menu item: {SOURCE_REAL}")
        for name, price in food.items()
    ]
    rows += [
        (name, cat, int(round(snack_prices[name])), affinity,
         "Price: median of Coimbatore outlet listings (data/reference/snack_drink_prices.csv)")
        for name, (cat, affinity) in SNACK_CATEGORIES.items()
    ]
    menu = pd.DataFrame(rows, columns=["item_name", "category", "price", "weather_affinity", "description"])
    menu.insert(0, "item_id", [f"M{i:02d}" for i in range(1, len(menu) + 1)])
    return menu


# -----------------------------------------------------------------------------
# Daily demand (T08)
# -----------------------------------------------------------------------------
def rain_multiplier(is_rainy: pd.Series, is_weekend: pd.Series, cal: C.Calibration) -> np.ndarray:
    """Rain lifts weekday orders and lowers weekend orders (Swiggy GLM, decision option 1)."""
    rain_effect = np.where(
        is_weekend, cal.volume["rain_weekend"]["value"], cal.volume["rain_weekday"]["value"]
    )
    return np.where(is_rainy, rain_effect, 1.0)


def generate_daily_demand(
    weather_df: pd.DataFrame,
    cal: C.Calibration,
    seed: int = SEED,
    outlets: int = OUTLETS,
) -> pd.DataFrame:
    """Daily order counts and regime flags for every day of the weather table."""
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["dow"] = pd.to_datetime(df["date"]).dt.day_of_week
    df["is_weekend"] = df["dow"].isin(WEEKEND_DAYS)
    df["is_rainy"] = df["precip_mm"] >= RAIN_TRACE_MM
    df["heavy_rain"] = df["precip_mm"] >= RAIN_HEAVY_MM
    df["is_hot"] = df["temp_max_c"] >= TEMP_HOT_C
    # Regime hierarchy: rainy takes precedence, then hot, then normal
    df["regime"] = np.select([df["is_rainy"], df["is_hot"]], ["rainy", "hot"], default="normal")
    df["festival_name"] = df["date"].map(COIMBATORE_FESTIVALS).fillna("")
    df["is_festival"] = df["festival_name"] != ""

    df["mult_weekend"] = np.where(df["is_weekend"], cal.volume["weekend"]["value"], 1.0)
    df["mult_rain"] = rain_multiplier(df["is_rainy"], df["is_weekend"], cal)
    df["mult_temp"] = np.where(df["is_hot"], cal.volume["heat"]["value"], 1.0)
    df["mult_festival"] = np.where(df["is_festival"], cal.festival, 1.0)
    df["expected_orders"] = (
        outlets * cal.volume["base_weekday_dry"]["value"]
        * df["mult_weekend"] * df["mult_rain"] * df["mult_temp"] * df["mult_festival"]
    )

    r = outlets * cal.dispersion_r
    rng = np.random.default_rng(seed)
    df["order_count"] = rng.negative_binomial(r, r / (r + df["expected_orders"].to_numpy()))
    return df


# -----------------------------------------------------------------------------
# Orders (T09)
# -----------------------------------------------------------------------------
def meal_slot(hour: int) -> str:
    for slot, hours in SLOT_BY_HOUR.items():
        if hour in hours:
            return slot
    raise ValueError(f"Hour {hour} is outside the trading hours")


def food_basket_weights(cal: C.Calibration) -> dict[str, np.ndarray]:
    """Bootstrap weights for the real baskets; soup baskets get the rainy-day tilt."""
    has_soup = np.array([any(C.is_soup(n) for n, _ in b) for b in cal.food_baskets])
    flat = np.ones(len(has_soup))
    rainy = np.where(has_soup, cal.soup_tilt, 1.0)
    return {"rainy": rainy / rainy.sum(), "hot": flat / flat.sum(), "normal": flat / flat.sum()}


def triggers(item: str, trigger: str) -> bool:
    """Snack triggers match exactly; food triggers (Biryani) match any dish that names them."""
    return trigger in item if trigger in FOOD_TRIGGERS else item == trigger


def set_pairs(basket: list[str], regime: str, cal: C.Calibration, rng) -> list[str]:
    """Planted pairs: when a trigger is in the basket, its add-on is present with exactly the
    survey probability (added or removed), so P(add-on | trigger, regime) equals the evidence."""
    result = list(basket)
    for (trigger, addon), probs in cal.pair_probs.items():
        if not any(triggers(item, trigger) for item in result):
            continue
        wanted = rng.random() < probs[regime]
        if wanted and addon not in result:
            result.append(addon)
        elif not wanted and addon in result:
            result.remove(addon)
    return result


def snack_basket(regime: str, cal: C.Calibration, rng) -> list[str]:
    names = list(cal.snack_weights[regime])
    weights = np.array([cal.snack_weights[regime][n] for n in names])
    sizes = list(cal.snack_basket_probs)
    k = rng.choice(sizes, p=[cal.snack_basket_probs[s] for s in sizes])
    drawn = rng.choice(len(names), size=k, replace=False, p=weights)
    return set_pairs([names[i] for i in drawn], regime, cal, rng)


def food_basket(regime: str, cal: C.Calibration, weights: dict[str, np.ndarray], rng) -> list[tuple[str, float]]:
    lines = list(cal.food_baskets[rng.choice(len(cal.food_baskets), p=weights[regime])])
    names = set_pairs([n for n, _ in lines], regime, cal, rng)
    return lines + [(n, None) for n in names[len(lines):]]


def generate_order_baskets(
    daily_df: pd.DataFrame,
    cal: C.Calibration,
    menu_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per order (orders) and one row per item line (order_items)."""
    rng = np.random.default_rng(seed)
    prices = dict(zip(menu_df["item_name"], menu_df["price"].astype(float)))
    ids = dict(zip(menu_df["item_name"], menu_df["item_id"]))
    cats = dict(zip(menu_df["item_name"], menu_df["category"]))
    hours = list(cal.hour_profile)
    hour_p = [cal.hour_profile[h] for h in hours]
    basket_w = food_basket_weights(cal)

    orders, lines = [], []
    order_id = 1
    for day in daily_df[["date", "regime", "order_count"]].itertuples(index=False):
        n = int(day.order_count)
        if n <= 0:
            continue
        secs = np.sort(rng.choice(hours, p=hour_p, size=n) * 3600 + rng.integers(0, 3600, size=n))
        snack_flags = rng.random(n) < cal.snack_share[day.regime]
        for sec, is_snack in zip(secs, snack_flags):
            if is_snack:
                basket = [(name, prices[name]) for name in snack_basket(day.regime, cal, rng)]
            else:
                basket = [
                    (name, price if price is not None else prices[name])
                    for name, price in food_basket(day.regime, cal, basket_w, rng)
                ]
            hour = int(sec // 3600)
            ts = f"{day.date} {hour:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
            slot = meal_slot(hour)
            names = [name for name, _ in basket]
            orders.append((order_id, day.date, ts, slot, "; ".join(names), len(names),
                           round(sum(p for _, p in basket), 2)))
            lines += [(order_id, day.date, ts, slot, ids[name], name, cats[name], p) for name, p in basket]
            order_id += 1

    orders_df = pd.DataFrame(orders, columns=[
        "order_id", "date", "timestamp", "meal_slot", "items", "item_count", "order_value"])
    items_df = pd.DataFrame(lines, columns=[
        "order_id", "date", "timestamp", "meal_slot", "item_id", "item_name", "category", "price"])
    return orders_df, items_df


def is_snack_order(items: str) -> bool:
    """A snack/drink order contains only snack/drink items (food orders always have a real dish)."""
    return all(name in SNACK_CATEGORIES for name in items.split("; "))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-based Coimbatore orders.")
    parser.add_argument("--weather", default=str(ROOT / "data/raw/weather.csv"))
    parser.add_argument("--out-menu", default=str(ROOT / "data/raw/menu.csv"))
    parser.add_argument("--out-daily", default=str(ROOT / "data/raw/daily_orders.csv"))
    parser.add_argument("--out-orders", default=str(ROOT / "data/raw/orders.csv"))
    parser.add_argument("--out-items", default=str(ROOT / "data/raw/order_items.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    weather_path = Path(args.weather)
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather dataset not found at {weather_path}")

    print("Calibrating from the Swiggy report, survey and outlet prices...")
    cal = C.calibrate(weather_path)
    _, real_items = C.real_orders.load_clean()
    menu = build_menu(real_items, cal.snack_prices)

    print("Generating daily demand and orders...")
    daily = generate_daily_demand(pd.read_csv(weather_path), cal, seed=args.seed)
    orders, order_items = generate_order_baskets(daily, cal, menu, seed=args.seed)

    for frame, path in ((menu, args.out_menu), (daily, args.out_daily),
                        (orders, args.out_orders), (order_items, args.out_items)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(f"Menu {len(menu)} items, {len(daily)} days, {len(orders):,} orders, "
          f"{len(order_items):,} item lines")


if __name__ == "__main__":
    main()
