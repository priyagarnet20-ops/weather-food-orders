"""Loader for the 'Weather & Food Ordering Habits - Coimbatore' Google Form responses.

The survey is the evidence for the generator's weather/festival directions and the
weather-linked item combos. Respondents under 18 are excluded (no guardian consent).

    data/real/survey_responses.csv   one row per adult respondent, short column names

Usage:
    python src/survey.py --responses "../Weather & Food Ordering Habits - Coimbatore (Responses).xlsx"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "real" / "survey_responses.csv"

QUESTION_NAMES: dict[str, str] = {
    "1": "age_group",
    "2": "area",
    "3": "order_frequency",
    "4": "usual_items",
    "5": "rain_volume",
    "6": "rain_amount",
    "7": "heat_volume",
    "8": "weekend_volume",
    "9": "rainy_items",
    "10": "hot_items",
    "11": "normal_items",
    "12": "chai_pakora",
    "13": "chai_samosa",
    "14": "coffee_vada",
    "15": "biryani_buttermilk",
    "16": "icecream_coldcoffee",
    "17": "rainy_combos",
    "18": "basket_size",
    "19": "festival_volume",
    "20": "peak_months",
    "21": "weather_most",
}

CONDITION_NAMES: dict[str, str] = {
    "Light drizzle": "drizzle",
    "Heavy rain": "heavy",
    "On rainy days": "rainy",
    "On dry days": "dry",
    "On hot days": "hot",
    "On normal days": "normal",
}

LIKERT: dict[str, int] = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}

COMBOS = ["chai_pakora", "chai_samosa", "coffee_vada", "biryani_buttermilk", "icecream_coldcoffee"]

VOLUME_WORDS: dict[str, str] = {
    "Less": "less",
    "Order less": "less",
    "Less on weekends": "less",
    "About the same": "same",
    "Same": "same",
    "A bit more": "more",
    "A lot more": "more",
    "More": "more",
    "Order more": "more",
    "Slightly more on weekends": "more",
    "Much more on weekends": "more",
}

MINOR = "Under 18"


def short_name(column: str) -> str:
    """Map a Google Form question header to a short snake_case name."""
    match = re.match(r"\s*(\d+)\.", column)
    if not match or match.group(1) not in QUESTION_NAMES:
        raise KeyError(f"Unknown survey question: {column!r}")
    name = QUESTION_NAMES[match.group(1)]
    bracket = re.search(r"\[(.+?)\]\s*$", column)
    if bracket:
        name = f"{name}_{CONDITION_NAMES[bracket.group(1)]}"
    return name


def clean_survey(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename questions, drop the timestamp and exclude respondents under 18."""
    df = raw.drop(columns=["Timestamp"], errors="ignore")
    df = df.rename(columns={c: short_name(c) for c in df.columns})
    df = df[df["age_group"] != MINOR].reset_index(drop=True)
    if "basket_size" in df.columns:
        df = df.assign(basket_size=df["basket_size"].astype(str))
    return df


def combo_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Likert score (0 = never .. 4 = always) per combo and weather condition."""
    rows = []
    for col in df.columns:
        combo = next((c for c in COMBOS if col.startswith(c + "_")), None)
        if combo is None:
            continue
        scores = df[col].map(LIKERT).dropna()
        rows.append({
            "combo": combo,
            "condition": col[len(combo) + 1:],
            "mean_score": scores.mean(),
            "n": len(scores),
        })
    return pd.DataFrame(rows, columns=["combo", "condition", "mean_score", "n"])


def volume_direction(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Count respondents who order less / the same / more under a condition."""
    mapped = df[column].map(VOLUME_WORDS)
    unknown = df[column][mapped.isna() & df[column].notna()].unique()
    if len(unknown):
        raise ValueError(f"Unmapped answers in {column}: {list(unknown)}")
    counts = mapped.value_counts()
    return {k: int(counts.get(k, 0)) for k in ("less", "same", "more")}


def item_mentions(df: pd.DataFrame, column: str) -> pd.Series:
    """How many respondents ticked each option in a multi-select question."""
    return df[column].dropna().str.split(r",\s*").explode().str.strip().value_counts()


def load_responses(path: str | Path) -> pd.DataFrame:
    """Read the original Google Form .xlsx export and clean it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Survey responses not found at {path}")
    return clean_survey(pd.read_excel(path))


def load_clean(path: str | Path = OUT_CSV) -> pd.DataFrame:
    """Read the cleaned CSV written by main()."""
    return pd.read_csv(path, dtype={"basket_size": str})


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the survey responses.")
    parser.add_argument("--responses", required=True, help="Path to the original .xlsx")
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    df = load_responses(args.responses)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} adult responses -> {out}")


if __name__ == "__main__":
    main()
