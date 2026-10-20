"""Surabaya F&B Dataset Preprocessing & Cleaning Pipeline (T08, T09, T13).

Transforms raw Indonesian MSME Point-of-Sale (POS) transactions (Transactions.csv)
and catalog (Products.csv) from Surabaya, Indonesia, into standardized orders,
order_items, and daily_orders tables integrated with Open-Meteo weather data.
"""
from __future__ import annotations

import re
from pathlib import Path
import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def build_menu_catalog(products_csv: Path = RAW_DIR / "Products.csv",
                       out_csv: Path = RAW_DIR / "menu.csv") -> pd.DataFrame:
    """Build standardized menu catalog from Products.csv with weather affinity classification."""
    df = pd.read_csv(products_csv)

    def get_full_name(r):
        v = str(r["Variant"]).strip()
        p = str(r["ProductName"]).strip()
        if v and v != "-":
            return f"{p} ({v})"
        return p

    df["item_name"] = df.apply(get_full_name, axis=1)
    df["item_id"] = [f"P{i:02d}" for i in df["ProductId"]]
    df["category"] = df["Category"]
    df["price"] = df["Price"].astype(int)
    df["description"] = df["Description"]

    def assign_affinity(r):
        cat = r["Category"]
        fn = r["item_name"]
        fn_lower = fn.lower()

        if cat == "Add on" or "air mineral" in fn_lower:
            return "neutral"

        if cat == "Snacks" or "cake" in fn_lower or "bread" in fn_lower:
            return "hot_comfort"

        # Regex word boundary avoids matching 'shot'
        if re.search(r"\bhot\b", fn_lower) or fn in ["Chamomile"]:
            return "hot_comfort"

        return "cold_refresh"

    df["weather_affinity"] = df.apply(assign_affinity, axis=1)
    menu_df = df[["item_id", "item_name", "category", "price", "weather_affinity", "description"]]
    menu_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(menu_df)} items)")
    return menu_df


def assign_meal_slot(dt: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Map order timestamp to operational meal slot."""
    h = dt.dt.hour
    slots = pd.Series("Dinner", index=dt.index)
    slots[(h >= 7) & (h < 11)] = "Breakfast"
    slots[(h >= 11) & (h < 15)] = "Lunch"
    slots[(h >= 15) & (h < 18)] = "Snacks"
    return slots


def preprocess_orders(
    tx_csv: Path = RAW_DIR / "Transactions.csv",
    menu_csv: Path = RAW_DIR / "menu.csv",
    orders_out: Path = RAW_DIR / "orders.csv",
    items_out: Path = RAW_DIR / "order_items.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine 3 outlets into a single restaurant order stream, clean values and format items."""
    tx = pd.read_csv(tx_csv)
    menu = pd.read_csv(menu_csv)
    price_map = dict(zip(menu["item_name"], menu["price"]))
    item_id_map = dict(zip(menu["item_name"], menu["item_id"]))
    category_map = dict(zip(menu["item_name"], menu["category"]))

    # Sort chronologically
    tx["datetime"] = pd.to_datetime(tx["Date"] + " " + tx["Time"])
    tx = tx.sort_values("datetime").reset_index(drop=True)

    # Standardize items to semicolon delimited string
    clean_items_list = []
    order_items_records = []

    for order_idx, row in tx.iterrows():
        order_id = order_idx + 1
        d_str = row["Date"]
        ts_str = row["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        items = [x.strip() for x in str(row["Items"]).split(",")]
        clean_items_list.append("; ".join(items))

        # Meal slot determined by hour
        h = row["datetime"].hour
        if 7 <= h < 11:
            slot = "Breakfast"
        elif 11 <= h < 15:
            slot = "Lunch"
        elif 15 <= h < 18:
            slot = "Snacks"
        else:
            slot = "Dinner"

        for it in items:
            order_items_records.append({
                "order_id": order_id,
                "date": d_str,
                "timestamp": ts_str,
                "meal_slot": slot,
                "item_id": item_id_map.get(it, "P00"),
                "item_name": it,
                "category": category_map.get(it, "Other"),
                "price": price_map.get(it, 0)
            })

    tx["items"] = clean_items_list
    tx["meal_slot"] = assign_meal_slot(tx["datetime"])

    # Order value: if TotalAmount > 0 use it; if TotalAmount == 0 (promotional/loyalty), impute catalog price
    def get_val(r):
        amt = r["TotalAmount"]
        if amt > 0:
            return int(amt)
        items = [x.strip() for x in str(r["Items"]).split(",")]
        return int(sum(price_map.get(it, 0) for it in items))

    tx["order_value"] = tx.apply(get_val, axis=1).astype(int)
    tx["order_id"] = np.arange(1, len(tx) + 1)
    tx["date"] = tx["Date"]
    tx["timestamp"] = tx["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    tx["item_count"] = tx["TotalItem"]

    orders_df = tx[["order_id", "date", "timestamp", "meal_slot", "items", "item_count", "order_value"]]
    orders_df.to_csv(orders_out, index=False)
    print(f"Wrote {orders_out} ({len(orders_df)} orders)")

    order_items_df = pd.DataFrame(order_items_records)
    order_items_df.to_csv(items_out, index=False)
    print(f"Wrote {items_out} ({len(order_items_df)} order item rows)")

    return orders_df, order_items_df


def build_daily_orders(
    orders_df: pd.DataFrame,
    weather_csv: Path = RAW_DIR / "weather.csv",
    daily_out: Path = RAW_DIR / "daily_orders.csv",
    merged_out: Path = PROCESSED_DIR / "daily_merged.csv"
) -> pd.DataFrame:
    """Aggregate orders to daily level and merge with Surabaya weather & Indonesian festival calendar."""
    weather = pd.read_csv(weather_csv)

    daily_agg = orders_df.groupby("date").agg(
        order_count=("order_id", "count"),
        total_revenue=("order_value", "sum"),
        mean_order_val=("order_value", "mean")
    ).reset_index()

    # Inner join so we have weather + order observations
    daily = daily_agg.merge(weather, on="date", how="left")

    # Calendar features
    d = pd.to_datetime(daily["date"])
    daily["dow"] = d.dt.dayofweek
    daily["is_weekend"] = daily["dow"].isin([4, 5, 6])
    daily["is_rainy"] = daily["precip_mm"] >= 2.5
    daily["heavy_rain"] = daily["precip_mm"] >= 15.0
    daily["is_hot"] = daily["temp_max_c"] >= 33.0

    def get_regime(r):
        if r["is_rainy"]:
            return "rainy"
        elif r["is_hot"]:
            return "hot"
        return "normal"

    daily["regime"] = daily.apply(get_regime, axis=1)

    # Indonesian national holidays
    years = sorted(d.dt.year.unique().tolist())
    id_hols = holidays.Indonesia(years=years)
    daily["is_festival"] = d.dt.date.apply(lambda dt: dt in id_hols)
    daily["festival_name"] = d.dt.date.apply(lambda dt: id_hols.get(dt, ""))

    # Save to data/raw/daily_orders.csv and data/processed/daily_merged.csv
    daily.to_csv(daily_out, index=False)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(merged_out, index=False)
    print(f"Wrote {daily_out} and {merged_out} ({len(daily)} daily rows)")
    return daily


def main() -> None:
    print("=== Processing Surabaya Dataset ===")
    build_menu_catalog()
    orders_df, _ = preprocess_orders()
    build_daily_orders(orders_df)
    print("=== Successfully completed preprocessing ===")


if __name__ == "__main__":
    main()
