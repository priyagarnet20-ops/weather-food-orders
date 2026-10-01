"""Weather-Driven Food Order Generator for Coimbatore (P2 / T08 & T09).

Generates synthetic daily demand volumes and customer order baskets conditioned
on real Coimbatore weather, day of week, and local festivals.

Theoretical Demand Model:
    Y_t ~ NegativeBinomial(r=25, p=r / (r + mu_t))
    mu_t = BASE * M_weekend(t) * M_rain(t) * M_temp(t) * M_festival(t)

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
# 3. Daily Multiplier Calculations (T08)
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
    is_weekend = dates.dt.dayofweek.isin([4, 5, 6])  # Friday, Saturday, Sunday
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
    """Sample order counts from Negative Binomial with mean mu and dispersion r.

    Parameterization:
        P(success) p = r / (r + mu)
        Mean E[Y] = r * (1 - p) / p = mu
        Variance Var(Y) = mu + (mu^2 / r)
    """
    rng = np.random.default_rng(seed)
    mu_arr = np.asarray(mu, dtype=float)
    p = dispersion_r / (dispersion_r + mu_arr)
    return rng.negative_binomial(dispersion_r, p)


def generate_daily_demand(
    weather_df: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Produce daily order counts and regime features given historical weather.

    Returns DataFrame containing original weather columns alongside:
    - dow, is_weekend
    - is_rainy, heavy_rain, is_hot, is_festival, festival_name
    - regime ('rainy', 'hot', or 'normal')
    - mult_weekend, mult_rain, mult_temp, mult_festival
    - expected_orders
    - order_count
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Coimbatore orders.")
    parser.add_argument(
        "--weather",
        default=str(ROOT / "data/raw/weather.csv"),
        help="Path to input weather.csv",
    )
    parser.add_argument(
        "--out-daily",
        default=str(ROOT / "data/raw/daily_orders.csv"),
        help="Path to output daily_orders.csv",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    weather_path = Path(args.weather)
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather dataset not found at {weather_path}")

    weather = pd.read_csv(weather_path)
    daily = generate_daily_demand(weather, seed=args.seed)

    out_daily_path = Path(args.out_daily)
    out_daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_daily_path, index=False)

    print(f"Generated daily demand for {len(daily)} days -> {out_daily_path}")
    print(f"Total orders: {daily['order_count'].sum():,}")
    print(f"Mean orders / day: {daily['order_count'].mean():.1f}")
    print(f"Min: {daily['order_count'].min()}, Max: {daily['order_count'].max()}")
    print("\nRegime breakdown:")
    print(daily.groupby("regime")["order_count"].agg(["count", "mean", "std"]))
    print("\nFestival impact:")
    print(daily.groupby("is_festival")["order_count"].agg(["count", "mean", "std"]))


if __name__ == "__main__":
    main()
