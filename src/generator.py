"""Weather-Driven Food Order Generator for Coimbatore (P2 / T08 & T09).

Generates synthetic daily demand volumes and customer order baskets conditioned
on real Coimbatore weather, day of week, and local festivals.

Theoretical Models:
    1. Daily Demand (T08):
       Y_t ~ NegativeBinomial(r=25, p=r / (r + mu_t))
       mu_t = BASE * M_weekend(t) * M_rain(t) * M_temp(t) * M_festival(t)

    2. Basket & Item Generation (T09):
       - Meal slots: Breakfast, Lunch, Evening Snacks, Dinner
       - Weather affinity multipliers for hot comfort vs cold refresh categories
       - Planted association rules (Chai -> Pakora, Juice -> Ice Cream, etc.)
       - Order timestamps clustered into meal slot windows

Author: P2 (Order Generator & Menu Catalog)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# 1. Project Global Parameters & Seed
# -----------------------------------------------------------------------------
SEED = 42
BASE_WEEKDAY = 140.0
WEEKEND_MULT = 1.30  # Friday, Saturday, Sunday
RAIN_MID_MULT = 1.20  # 2.5mm <= precip < 15mm
RAIN_HEAVY_MULT = 1.35  # precip >= 15mm
HEAT_MULT = 1.10  # temp_max >= 33.0 C
FESTIVAL_MULT = 1.25  # Local Coimbatore / Tamil Nadu festival days
NB_DISPERSION = 25  # Negative Binomial parameter 'r' (variance control)

# Meteorological thresholds matching project conventions (docs/dataset.md)
RAIN_TRACE_MM = 2.5
RAIN_HEAVY_MM = 15.0
TEMP_HOT_C = 33.0

# -----------------------------------------------------------------------------
# 2. Coimbatore & Tamil Nadu Major Festivals Calendar (2023 - 2026)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. Meal Slot Windows and Probabilities (T09)
# -----------------------------------------------------------------------------
# Window defined by (start_seconds_from_midnight, duration_seconds)
SLOT_WINDOWS: dict[str, tuple[int, int]] = {
    "Breakfast": (7 * 3600, int(3.5 * 3600)),       # 07:00 to 10:30 (3.5h)
    "Lunch": (12 * 3600, 3 * 3600),                  # 12:00 to 15:00 (3.0h)
    "Evening Snacks": (16 * 3600, int(2.5 * 3600)), # 16:00 to 18:30 (2.5h)
    "Dinner": (19 * 3600, int(3.5 * 3600)),         # 19:00 to 22:30 (3.5h)
}

SLOT_PROBS: dict[str, float] = {
    "Breakfast": 0.15,
    "Lunch": 0.35,
    "Evening Snacks": 0.20,
    "Dinner": 0.30,
}

# Basket size probability distribution: k in {1, 2, 3, 4, 5}
BASKET_SIZE_PROBS: dict[int, float] = {
    1: 0.30,
    2: 0.42,
    3: 0.18,
    4: 0.07,
    5: 0.03,
}

# Category affinity multipliers conditioned on weather regime
CATEGORY_WEATHER_AFFINITY: dict[str, dict[str, float]] = {
    "rainy": {"hot_comfort": 1.80, "cold_refresh": 0.60, "neutral": 1.00},
    "hot": {"hot_comfort": 0.80, "cold_refresh": 1.90, "neutral": 1.00},
    "normal": {"hot_comfort": 1.00, "cold_refresh": 1.00, "neutral": 1.00},
}

# Meal slot multiplier per menu category
SLOT_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "Breakfast": {
        "Beverages (Hot)": 2.5,
        "South Indian Tiffin": 3.0,
        "Snacks & Quick Bites": 1.0,
        "Biryani & Mains": 0.2,
        "Cold Refreshments & Juices": 0.8,
        "Desserts & Ice Cream": 0.2,
    },
    "Lunch": {
        "Beverages (Hot)": 0.5,
        "South Indian Tiffin": 0.8,
        "Snacks & Quick Bites": 0.8,
        "Biryani & Mains": 3.5,
        "Cold Refreshments & Juices": 2.0,
        "Desserts & Ice Cream": 1.5,
    },
    "Evening Snacks": {
        "Beverages (Hot)": 3.5,
        "South Indian Tiffin": 0.8,
        "Snacks & Quick Bites": 4.0,
        "Biryani & Mains": 0.2,
        "Cold Refreshments & Juices": 1.5,
        "Desserts & Ice Cream": 1.0,
    },
    "Dinner": {
        "Beverages (Hot)": 0.8,
        "South Indian Tiffin": 2.5,
        "Snacks & Quick Bites": 0.8,
        "Biryani & Mains": 3.0,
        "Cold Refreshments & Juices": 1.5,
        "Desserts & Ice Cream": 2.0,
    },
}

# Planted conditional pair rules (Trigger -> Addon -> {regime: probability})
# Evaluated when basket size > 1 to inject ground truth associations
PAIR_RULES: dict[str, dict[str, dict[str, float]]] = {
    # Headline rainy rule: Chai -> Pakora
    "Masala Chai": {
        "Onion Pakora": {"rainy": 0.58, "normal": 0.12, "hot": 0.08},
        "Milagai Bajji": {"rainy": 0.35, "normal": 0.10, "hot": 0.08},
    },
    # Secondary rainy comfort combo: Coffee -> Vada
    "South Indian Filter Coffee": {
        "Medu Vada": {"rainy": 0.40, "normal": 0.15, "hot": 0.10},
    },
    # Headline heat rule: Juice -> Ice cream
    "Fresh Lime Juice": {
        "Tender Coconut Ice Cream": {"rainy": 0.06, "normal": 0.15, "hot": 0.45},
    },
    # Secondary heat refresher: Juice -> Vanilla Ice cream
    "Watermelon Juice": {
        "Vanilla Ice Cream": {"rainy": 0.05, "normal": 0.12, "hot": 0.38},
    },
    # Heat biryani combo: Biryani -> Moru (Buttermilk)
    "Coimbatore Chicken Biryani": {
        "Buttermilk (Moru)": {"rainy": 0.12, "normal": 0.20, "hot": 0.45},
    },
    # Dinner staple combo: Parotta -> Chicken Chukka
    "Malabar Parotta (2 pcs)": {
        "Chicken Chukka": {"rainy": 0.35, "normal": 0.35, "hot": 0.30},
    },
    # Tiffin staple combo: Dosa -> Filter Coffee
    "Ghee Podi Dosa": {
        "South Indian Filter Coffee": {"rainy": 0.38, "normal": 0.30, "hot": 0.20},
    },
}


# -----------------------------------------------------------------------------
# 4. Daily Demand Multipliers (T08)
# -----------------------------------------------------------------------------
def get_rain_multiplier(precip_mm: float) -> float:
    """Return rain demand multiplier based on daily precipitation in mm."""
    if precip_mm >= RAIN_HEAVY_MM:
        return RAIN_HEAVY_MULT
    elif precip_mm >= RAIN_TRACE_MM:
        return RAIN_MID_MULT
    return 1.00


def get_temp_multiplier(temp_max_c: float) -> float:
    """Return temperature demand multiplier based on maximum daily temperature."""
    return HEAT_MULT if temp_max_c >= TEMP_HOT_C else 1.00


def get_festival_multiplier(date_str: str) -> float:
    """Return festival demand multiplier if the date is a recognized festival."""
    return FESTIVAL_MULT if date_str in COIMBATORE_FESTIVALS else 1.00


def compute_expected_orders(
    weather_df: pd.DataFrame,
    base_weekday: float = BASE_WEEKDAY,
    weekend_mult: float = WEEKEND_MULT,
) -> pd.Series:
    """Compute deterministic expected daily order demand mu_t from multipliers."""
    dates = pd.to_datetime(weather_df["date"])
    is_weekend = dates.dt.dayofweek.isin([4, 5, 6])
    weekend_m = np.where(is_weekend, weekend_mult, 1.0)
    rain_m = weather_df["precip_mm"].apply(get_rain_multiplier).to_numpy()
    heat_m = weather_df["temp_max_c"].apply(get_temp_multiplier).to_numpy()
    date_strs = dates.dt.strftime("%Y-%m-%d")
    fest_m = date_strs.apply(get_festival_multiplier).to_numpy()

    return pd.Series(
        base_weekday * weekend_m * rain_m * heat_m * fest_m,
        index=weather_df.index,
        name="expected_orders",
    )


def sample_negative_binomial(
    mu: Sequence[float] | np.ndarray | pd.Series,
    dispersion_r: float = NB_DISPERSION,
    seed: int = SEED,
) -> np.ndarray:
    """Sample order counts from Negative Binomial with mean mu and dispersion r."""
    rng = np.random.default_rng(seed)
    mu_arr = np.asarray(mu, dtype=float)
    p = dispersion_r / (dispersion_r + mu_arr)
    return rng.negative_binomial(dispersion_r, p)


def generate_daily_demand(
    weather_df: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Produce daily order counts and regime features given historical weather."""
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    dt_series = pd.to_datetime(df["date"])

    df["dow"] = dt_series.dt.dayofweek
    df["is_weekend"] = df["dow"].isin([4, 5, 6])
    df["is_rainy"] = df["precip_mm"] >= RAIN_TRACE_MM
    df["heavy_rain"] = df["precip_mm"] >= RAIN_HEAVY_MM
    df["is_hot"] = df["temp_max_c"] >= TEMP_HOT_C

    # Regime hierarchy: rainy takes precedence, followed by hot, then normal
    df["regime"] = np.select(
        [df["is_rainy"], df["is_hot"]],
        ["rainy", "hot"],
        default="normal",
    )

    df["festival_name"] = df["date"].map(COIMBATORE_FESTIVALS).fillna("")
    df["is_festival"] = df["festival_name"] != ""

    df["mult_weekend"] = np.where(df["is_weekend"], WEEKEND_MULT, 1.00)
    df["mult_rain"] = df["precip_mm"].apply(get_rain_multiplier)
    df["mult_temp"] = df["temp_max_c"].apply(get_temp_multiplier)
    df["mult_festival"] = np.where(df["is_festival"], FESTIVAL_MULT, 1.00)

    df["expected_orders"] = (
        BASE_WEEKDAY
        * df["mult_weekend"]
        * df["mult_rain"]
        * df["mult_temp"]
        * df["mult_festival"]
    )

    df["order_count"] = sample_negative_binomial(
        df["expected_orders"], dispersion_r=NB_DISPERSION, seed=seed
    )

    return df


# -----------------------------------------------------------------------------
# 5. Basket & Meal-Slot Generator (T09)
# -----------------------------------------------------------------------------
def precompute_item_distributions(
    menu_df: pd.DataFrame,
) -> tuple[dict[tuple[str, str], np.ndarray], list[str], dict[str, int]]:
    """Precompute normalized item sampling probabilities for each (slot, regime)."""
    items = menu_df["item_name"].tolist()
    item_idx = {name: i for i, name in enumerate(items)}
    n_items = len(items)

    base_weights = np.ones(n_items, dtype=float)
    for i, row in menu_df.iterrows():
        name = row["item_name"]
        if name in ["Masala Chai", "South Indian Filter Coffee"]:
            base_weights[i] = 3.0
        elif name in ["Coimbatore Chicken Biryani", "Ghee Podi Dosa", "Onion Pakora", "Fresh Lime Juice"]:
            base_weights[i] = 2.0

    dists = {}
    for slot in SLOT_WINDOWS.keys():
        for regime in ["rainy", "hot", "normal"]:
            w = base_weights.copy()
            for i, row in menu_df.iterrows():
                w[i] *= SLOT_CATEGORY_WEIGHTS[slot].get(row["category"], 1.0)
                w[i] *= CATEGORY_WEATHER_AFFINITY[regime].get(row["weather_affinity"], 1.0)
            dists[(slot, regime)] = w / w.sum()

    return dists, items, item_idx


def generate_order_baskets(
    daily_df: pd.DataFrame,
    menu_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate granular order transactions with meal slots, timestamps, and baskets.

    Returns:
        orders_df: One row per order (order_id, date, timestamp, meal_slot, items, item_count, order_value)
        order_items_df: Long format (order_id, date, timestamp, meal_slot, item_id, item_name, category, price)
    """
    rng = np.random.default_rng(seed)
    dists, items, item_idx = precompute_item_distributions(menu_df)

    item_prices = dict(zip(menu_df["item_name"], menu_df["price"]))
    item_ids = dict(zip(menu_df["item_name"], menu_df["item_id"]))
    item_cats = dict(zip(menu_df["item_name"], menu_df["category"]))

    slot_names = list(SLOT_PROBS.keys())
    slot_p = [SLOT_PROBS[s] for s in slot_names]

    basket_ks = list(BASKET_SIZE_PROBS.keys())
    basket_p = [BASKET_SIZE_PROBS[k] for k in basket_ks]

    orders_records = []
    order_items_records = []
    order_id = 1

    for _, day in daily_df.iterrows():
        date_str = str(day["date"])
        regime = day["regime"]
        count = int(day["order_count"])

        if count <= 0:
            continue

        # Draw meal slots and timestamps for the day's orders
        slots = rng.choice(slot_names, p=slot_p, size=count)
        k_sizes = rng.choice(basket_ks, p=basket_p, size=count)

        # Pre-generate random second offsets within slot window
        offsets = rng.random(size=count)
        times_seconds = []
        for i in range(count):
            start_sec, dur_sec = SLOT_WINDOWS[slots[i]]
            times_seconds.append(start_sec + int(offsets[i] * dur_sec))

        # Sort order indices chronologically for realistic ordering
        sort_order = np.argsort(times_seconds)

        for idx in sort_order:
            slot = slots[idx]
            k = k_sizes[idx]
            sec_val = times_seconds[idx]

            h = sec_val // 3600
            m = (sec_val % 3600) // 60
            s = sec_val % 60
            timestamp_str = f"{date_str} {h:02d}:{m:02d}:{s:02d}"

            # 1. Sample primary item from distribution
            dist = dists[(slot, regime)]
            first_item = rng.choice(items, p=dist)
            basket = [first_item]

            # 2. Add companion items via pair rules or conditioned distribution
            while len(basket) < k:
                added_via_rule = False
                for trig in list(basket):
                    if trig in PAIR_RULES:
                        for addon, prob_map in PAIR_RULES[trig].items():
                            if addon not in basket and rng.random() < prob_map[regime]:
                                basket.append(addon)
                                added_via_rule = True
                                break
                    if added_via_rule or len(basket) >= k:
                        break

                if added_via_rule:
                    continue

                # Sample without replacement from remaining items
                rem_dist = dist.copy()
                for b in basket:
                    rem_dist[item_idx[b]] = 0.0
                sum_rem = rem_dist.sum()
                if sum_rem == 0:
                    break
                rem_dist /= sum_rem
                nxt = rng.choice(items, p=rem_dist)
                basket.append(nxt)

            order_val = sum(item_prices[b] for b in basket)
            items_str = "; ".join(basket)

            orders_records.append(
                (order_id, date_str, timestamp_str, slot, items_str, len(basket), order_val)
            )

            for b in basket:
                order_items_records.append(
                    (order_id, date_str, timestamp_str, slot, item_ids[b], b, item_cats[b], item_prices[b])
                )

            order_id += 1

    orders_df = pd.DataFrame(
        orders_records,
        columns=["order_id", "date", "timestamp", "meal_slot", "items", "item_count", "order_value"],
    )

    order_items_df = pd.DataFrame(
        order_items_records,
        columns=["order_id", "date", "timestamp", "meal_slot", "item_id", "item_name", "category", "price"],
    )

    return orders_df, order_items_df


def evaluate_planted_rules(
    orders_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    trigger: str = "Masala Chai",
    addon: str = "Onion Pakora",
) -> pd.DataFrame:
    """Evaluate empirical support, confidence, and lift for a pair across regimes."""
    merged = orders_df.merge(daily_df[["date", "regime"]], on="date")
    results = []

    for regime in ["rainy", "hot", "normal"]:
        sub = merged[merged["regime"] == regime]
        n = len(sub)
        if n == 0:
            continue
        has_a = sub["items"].str.contains(trigger, regex=False)
        has_b = sub["items"].str.contains(addon, regex=False)
        has_ab = has_a & has_b

        supp_a = has_a.mean()
        supp_b = has_b.mean()
        supp_ab = has_ab.mean()

        conf = supp_ab / supp_a if supp_a > 0 else 0.0
        lift = supp_ab / (supp_a * supp_b) if (supp_a * supp_b) > 0 else 0.0

        results.append({
            "regime": regime,
            "orders": n,
            "supp_trigger": round(supp_a, 4),
            "supp_addon": round(supp_b, 4),
            "support": round(supp_ab, 4),
            "confidence": round(conf, 4),
            "lift": round(lift, 2),
        })

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 6. CLI Execution
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Coimbatore orders.")
    parser.add_argument(
        "--weather",
        default=str(ROOT / "data/raw/weather.csv"),
        help="Path to input weather.csv",
    )
    parser.add_argument(
        "--menu",
        default=str(ROOT / "data/raw/menu.csv"),
        help="Path to input menu.csv",
    )
    parser.add_argument(
        "--out-daily",
        default=str(ROOT / "data/raw/daily_orders.csv"),
        help="Path to output daily_orders.csv",
    )
    parser.add_argument(
        "--out-orders",
        default=str(ROOT / "data/raw/orders.csv"),
        help="Path to output orders.csv",
    )
    parser.add_argument(
        "--out-items",
        default=str(ROOT / "data/raw/order_items.csv"),
        help="Path to output order_items.csv",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    weather_path = Path(args.weather)
    menu_path = Path(args.menu)

    if not weather_path.exists():
        raise FileNotFoundError(f"Weather dataset not found at {weather_path}")
    if not menu_path.exists():
        raise FileNotFoundError(f"Menu catalog not found at {menu_path}")

    print("Loading input data...")
    weather = pd.read_csv(weather_path)
    menu = pd.read_csv(menu_path)

    print("Step 1: Generating daily demand...")
    daily = generate_daily_demand(weather, seed=args.seed)

    print("Step 2: Generating order baskets and meal slot timestamps...")
    orders, order_items = generate_order_baskets(daily, menu, seed=args.seed)

    # Save outputs
    out_daily = Path(args.out_daily)
    out_orders = Path(args.out_orders)
    out_items = Path(args.out_items)

    out_daily.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_daily, index=False)
    orders.to_csv(out_orders, index=False)
    order_items.to_csv(out_items, index=False)

    print(f"\nWrote daily orders ({len(daily)} days) -> {out_daily}")
    print(f"Wrote orders ({len(orders):,} transactions) -> {out_orders}")
    print(f"Wrote order items ({len(order_items):,} line items) -> {out_items}")

    print("\nPlanted Association Rule Validation:")
    print("--- {Masala Chai} -> {Onion Pakora} (Rain Affinity) ---")
    print(evaluate_planted_rules(orders, daily, "Masala Chai", "Onion Pakora").to_string(index=False))

    print("\n--- {Fresh Lime Juice} -> {Tender Coconut Ice Cream} (Heat Affinity) ---")
    print(evaluate_planted_rules(orders, daily, "Fresh Lime Juice", "Tender Coconut Ice Cream").to_string(index=False))

    print("\n--- {South Indian Filter Coffee} -> {Medu Vada} (Rain Affinity) ---")
    print(evaluate_planted_rules(orders, daily, "South Indian Filter Coffee", "Medu Vada").to_string(index=False))


if __name__ == "__main__":
    main()
