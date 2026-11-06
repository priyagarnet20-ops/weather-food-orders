"""Loader for the real Swiggy order report (Hotel Chinthamani, Coimbatore).

The report is the base the synthetic generator is calibrated against and validated on.
It is read from the original .xlsx (kept out of git) and written as two cleaned CSVs:

    data/real/chinthamani_orders.csv   one row per delivered order
    data/real/chinthamani_items.csv    one row per item line

Swiggy order IDs are replaced by a running number; the restaurant address and
tax/discount columns are dropped.

Usage:
    python src/real_orders.py --report "../report chinthamani.xlsx"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "real"

HEADER_ROW = 5  # rows 0-4 hold the report title, period and disclaimer
STATUS_COL = "Order-status"
TIME_COL = "Order-relay-time(ordered time)"
BILL_COL = "Total-bill-amount <bill>"
COUNT_COL = "Item-count"
FIRST_ITEM_COL = "Item1-name_reward_type_quantity_price+Variants+Addons"
REQUIRED = [STATUS_COL, TIME_COL, BILL_COL, COUNT_COL, FIRST_ITEM_COL]


def parse_item_cell(cell: str) -> tuple[str, int, float]:
    """Split 'Name_reward_qty_price+Addons' into (name, quantity, line price).

    Add-ons after '+' are bundled into the line price by Swiggy, so they are dropped.
    """
    head = str(cell).split("+")[0]
    parts = head.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"Unrecognised item cell: {cell!r}")
    name, _reward, qty, price = parts
    return name.strip(), int(qty), float(price)


def _item_columns(raw: pd.DataFrame) -> list[str]:
    """Item lines sit in the first item column and every column to its right."""
    start = list(raw.columns).index(FIRST_ITEM_COL)
    return list(raw.columns[start:])


def clean_report(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (orders, items) for delivered orders, sorted by order time."""
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Swiggy report is missing columns: {missing}")

    delivered = raw[raw[STATUS_COL] == "delivered"].copy()
    delivered["timestamp"] = pd.to_datetime(delivered[TIME_COL])
    delivered = delivered.sort_values("timestamp").reset_index(drop=True)
    delivered["order_no"] = delivered.index + 1

    item_cols = _item_columns(delivered)
    item_rows = []
    for _, row in delivered.iterrows():
        cells = [row[c] for c in item_cols if isinstance(row[c], str) and row[c].strip()]
        if len(cells) != int(row[COUNT_COL]):
            raise ValueError(
                f"Order {row['order_no']}: item count {row[COUNT_COL]} but {len(cells)} item lines"
            )
        for cell in cells:
            name, qty, price = parse_item_cell(cell)
            item_rows.append((row["order_no"], name, qty, price, price / qty))

    ts = delivered["timestamp"]
    orders = pd.DataFrame({
        "order_no": delivered["order_no"],
        "date": ts.dt.strftime("%Y-%m-%d"),
        "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": ts.dt.hour,
        "dow": ts.dt.day_of_week,
        "bill": delivered[BILL_COL].astype(float),
        "item_count": delivered[COUNT_COL].astype(int),
    })
    items = pd.DataFrame(
        item_rows, columns=["order_no", "item_name", "quantity", "line_price", "unit_price"]
    )
    return orders, items


def load_report(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the original Swiggy .xlsx export and clean it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Swiggy report not found at {path}")
    return clean_report(pd.read_excel(path, header=HEADER_ROW))


def load_clean(out_dir: str | Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the cleaned CSVs written by main()."""
    out_dir = Path(out_dir)
    return (
        pd.read_csv(out_dir / "chinthamani_orders.csv"),
        pd.read_csv(out_dir / "chinthamani_items.csv"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the Swiggy order report.")
    parser.add_argument("--report", required=True, help="Path to the original .xlsx")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    orders, items = load_report(args.report)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orders.to_csv(out_dir / "chinthamani_orders.csv", index=False)
    items.to_csv(out_dir / "chinthamani_items.csv", index=False)
    print(f"Wrote {len(orders)} orders and {len(items)} item lines -> {out_dir}")


if __name__ == "__main__":
    main()
