"""Association Rule Mining & Evaluation module (Review 2 / Method 1).

Covers:
- T16: ARM Data Preparation & Weather Segmentation (P3)
- T17: FP-Growth & Apriori Implementation (P3)
- T11: Rainy vs. Dry Rule Evaluation & Weather Affinity Analysis (P2)
- T12: Rule Threshold Sensitivity Grid across min_support and min_confidence (P2)

Target City: Coimbatore, Tamil Nadu, India (11.00°N, 76.99°E)
Dataset: 1,097 days · 182,696 orders · 80-item menu catalog
Author: P3 (Data Cleaning, Feature Engineering & ARM Preparation)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[1]

# Shared plotting aesthetics
PALETTE = {
    "ink": "#0B0B0B",
    "ink2": "#52514E",
    "blue": "#2A78D6",
    "orange": "#EB6834",
    "aqua": "#1BAF7A",
    "gray": "#8A8983",
    "bg": "#FCFCFB",
    "white": "#FFFFFF",
    "dark": "#18243A",
}


# =============================================================================
# 1. T16: ARM Data Preparation & Weather Segmentation
# =============================================================================

def load_arm_data(
    orders_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    menu_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load orders, features, and menu catalog, returning merged orders and menu."""
    orders_file = orders_path or (ROOT / "data/raw/orders.csv")
    features_file = features_path or (ROOT / "data/processed/daily_features.csv")
    menu_file = menu_path or (ROOT / "data/raw/menu.csv")

    orders = pd.read_csv(orders_file)
    features = pd.read_csv(features_file)[
        ["date", "is_rainy", "heavy_rain", "is_hot", "precip_mm", "temp_max_c", "monsoon_phase"]
    ].drop_duplicates()
    menu = pd.read_csv(menu_file)

    merged = orders.merge(features, on="date", how="left")
    return merged, menu


def encode_baskets(
    orders_df: pd.DataFrame,
    multi_only: bool = True,
) -> Tuple[pd.DataFrame, TransactionEncoder]:
    """Extract item baskets from orders and encode as a boolean matrix.

    Args:
        orders_df: DataFrame with an 'items' column (semicolon-delimited strings).
        multi_only: If True, keep only orders with item_count >= 2.

    Returns:
        (encoded_df, transaction_encoder)
    """
    df = orders_df[orders_df["item_count"] >= 2] if multi_only else orders_df
    baskets = [s.split("; ") for s in df["items"]]
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_, index=df.index)
    return encoded_df, te


def segment_transactions(
    orders_df: pd.DataFrame,
    multi_only: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Split orders into distinct weather-conditioned transaction segments.

    Segments:
    - overall: all orders (or multi-item orders)
    - rainy: days with precipitation >= 2.5 mm (is_rainy == 1)
    - dry: days with precipitation < 2.5 mm (is_rainy == 0)
    - heavy_rain: days with precipitation >= 15.0 mm (heavy_rain == 1)
    - hot: days with maximum temperature >= 33.0 °C (is_hot == 1)
    """
    base = orders_df[orders_df["item_count"] >= 2] if multi_only else orders_df

    segments = {
        "overall": base,
        "rainy": base[base["is_rainy"] == 1],
        "dry": base[base["is_rainy"] == 0],
        "heavy_rain": base[base["heavy_rain"] == 1],
        "hot": base[base["is_hot"] == 1],
    }
    return segments


# =============================================================================
# 2. T17: FP-Growth & Apriori Implementation
# =============================================================================

def mine_frequent_itemsets(
    encoded_df: pd.DataFrame,
    min_support: float = 0.005,
    algorithm: str = "fpgrowth",
) -> pd.DataFrame:
    """Mine frequent itemsets using FP-Growth or Apriori.

    Args:
        encoded_df: Boolean DataFrame from TransactionEncoder.
        min_support: Minimum support threshold (proportion of transactions).
        algorithm: 'fpgrowth' or 'apriori'.
    """
    algo = algorithm.lower()
    if algo == "fpgrowth":
        fi = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
    elif algo == "apriori":
        fi = apriori(encoded_df, min_support=min_support, use_colnames=True)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'fpgrowth' or 'apriori'.")

    if not fi.empty:
        fi["length"] = fi["itemsets"].apply(len)
        fi = fi.sort_values("support", ascending=False).reset_index(drop=True)
    return fi


def mine_association_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.05,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """Generate association rules from frequent itemsets and format string names."""
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    if rules.empty:
        return pd.DataFrame()

    # Format human-readable antecedent and consequent strings
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["rule_str"] = rules["antecedents_str"] + " -> " + rules["consequents_str"]

    if min_lift > 0:
        rules = rules[rules["lift"] >= min_lift]

    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


# =============================================================================
# 3. T11: Rainy vs. Dry Rule Evaluation & Affinity Recovery
# =============================================================================

def classify_rule_affinity(
    antecedents_str: str,
    consequents_str: str,
    affinity_map: Dict[str, str],
) -> str:
    """Classify pair into hot_comfort, cold_refresh, hybrid, or neutral."""
    ant_items = [x.strip() for x in antecedents_str.split(", ")]
    con_items = [x.strip() for x in consequents_str.split(", ")]

    ant_affs = [affinity_map.get(it, "neutral") for it in ant_items]
    con_affs = [affinity_map.get(it, "neutral") for it in con_items]
    all_affs = set(ant_affs + con_affs)

    if all_affs == {"hot_comfort"}:
        return "hot_comfort_pair"
    elif all_affs == {"cold_refresh"}:
        return "cold_refresh_pair"
    elif "hot_comfort" in all_affs and "cold_refresh" in all_affs:
        return "hot_cold_hybrid"
    else:
        return "neutral_staple"


def evaluate_rainy_vs_dry_rules(
    rules_rain: pd.DataFrame,
    rules_dry: pd.DataFrame,
    menu_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Compare support, confidence, and lift between rainy and dry days (T11).

    Identifies:
    1. Common rules and their lift shifts (Delta lift = lift_rain - lift_dry).
    2. Rainy-only rules (rules discovered in rain segment above thresholds).
    3. Dry-only rules (rules discovered in dry segment above thresholds).
    4. Weather affinity classification for each rule.

    Returns dictionary with:
    - 'common': rules present in both segments with comparison metrics.
    - 'rainy_only': rules present only in the rainy segment.
    - 'dry_only': rules present only in the dry segment.
    """
    affinity_map: Dict[str, str] = {}
    if menu_df is not None and "weather_affinity" in menu_df.columns:
        affinity_map = menu_df.set_index("item_name")["weather_affinity"].to_dict()

    rain_cols = ["rule_str", "antecedents_str", "consequents_str", "support", "confidence", "lift"]
    dry_cols = ["rule_str", "antecedents_str", "consequents_str", "support", "confidence", "lift"]

    r_rain = rules_rain[rain_cols].copy() if not rules_rain.empty else pd.DataFrame(columns=rain_cols)
    r_dry = rules_dry[dry_cols].copy() if not rules_dry.empty else pd.DataFrame(columns=dry_cols)

    # 1. Common rules
    common = r_rain.merge(
        r_dry[["rule_str", "support", "confidence", "lift"]],
        on="rule_str",
        suffixes=("_rain", "_dry"),
    )
    if not common.empty:
        common["lift_diff"] = common["lift_rain"] - common["lift_dry"]
        common["lift_ratio"] = common["lift_rain"] / common["lift_dry"]
        common["conf_diff"] = common["confidence_rain"] - common["confidence_dry"]
        common["support_diff"] = common["support_rain"] - common["support_dry"]
        common["affinity_type"] = common.apply(
            lambda r: classify_rule_affinity(r["antecedents_str"], r["consequents_str"], affinity_map),
            axis=1,
        )
        common = common.sort_values("lift_diff", ascending=False).reset_index(drop=True)

    # 2. Rainy-only rules
    if not r_rain.empty:
        rainy_only = r_rain[~r_rain["rule_str"].isin(r_dry["rule_str"])].copy()
        rainy_only["affinity_type"] = rainy_only.apply(
            lambda r: classify_rule_affinity(r["antecedents_str"], r["consequents_str"], affinity_map),
            axis=1,
        )
        rainy_only = rainy_only.sort_values("lift", ascending=False).reset_index(drop=True)
    else:
        rainy_only = pd.DataFrame()

    # 3. Dry-only rules
    if not r_dry.empty:
        dry_only = r_dry[~r_dry["rule_str"].isin(r_rain["rule_str"])].copy()
        dry_only["affinity_type"] = dry_only.apply(
            lambda r: classify_rule_affinity(r["antecedents_str"], r["consequents_str"], affinity_map),
            axis=1,
        )
        dry_only = dry_only.sort_values("lift", ascending=False).reset_index(drop=True)
    else:
        dry_only = pd.DataFrame()

    return {
        "common": common,
        "rainy_only": rainy_only,
        "dry_only": dry_only,
    }


# =============================================================================
# 4. T12: Rule Threshold Sensitivity Grid
# =============================================================================

def evaluate_threshold_sensitivity(
    encoded_df: pd.DataFrame,
    support_grid: Optional[Sequence[float]] = None,
    confidence_grid: Optional[Sequence[float]] = None,
    algorithm: str = "fpgrowth",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate rule counts, top rules, and lift stability across parameter grid (T12).

    Args:
        encoded_df: Encoded boolean transactions.
        support_grid: List of min_support values to test.
        confidence_grid: List of min_confidence values to test.
        algorithm: 'fpgrowth' or 'apriori'.

    Returns:
        (grid_summary_df, pivot_rule_counts, pivot_max_lift)
    """
    supports = list(support_grid) if support_grid is not None else [0.004, 0.006, 0.008, 0.010, 0.015]
    confidences = list(confidence_grid) if confidence_grid is not None else [0.02, 0.05, 0.10, 0.15, 0.20]

    rows = []
    # Cache frequent itemsets per support level to avoid redundant mining
    itemsets_cache = {}
    for ms in supports:
        fi = mine_frequent_itemsets(encoded_df, min_support=ms, algorithm=algorithm)
        itemsets_cache[ms] = fi

    for ms in supports:
        fi = itemsets_cache[ms]
        n_fi = len(fi)
        n_pairs = len(fi[fi["length"] >= 2]) if n_fi > 0 and "length" in fi.columns else 0

        for mc in confidences:
            if n_pairs > 0:
                rules = mine_association_rules(fi, metric="confidence", min_threshold=mc, min_lift=1.0)
                n_rules = len(rules)
                max_lift = float(rules["lift"].max()) if n_rules > 0 else 0.0
                mean_lift = float(rules["lift"].mean()) if n_rules > 0 else 0.0
                max_conf = float(rules["confidence"].max()) if n_rules > 0 else 0.0
                n_lift_1_5 = int((rules["lift"] >= 1.5).sum()) if n_rules > 0 else 0
                top_rule = rules.iloc[0]["rule_str"] if n_rules > 0 else "None"
            else:
                n_rules = 0
                max_lift = 0.0
                mean_lift = 0.0
                max_conf = 0.0
                n_lift_1_5 = 0
                top_rule = "None"

            rows.append({
                "min_support": ms,
                "min_confidence": mc,
                "total_itemsets": n_fi,
                "pair_itemsets": n_pairs,
                "rule_count": n_rules,
                "rules_lift_ge_1_5": n_lift_1_5,
                "max_lift": round(max_lift, 3),
                "mean_lift": round(mean_lift, 3),
                "max_confidence": round(max_conf, 3),
                "top_rule": top_rule,
            })

    summary_df = pd.DataFrame(rows)
    pivot_counts = summary_df.pivot(index="min_support", columns="min_confidence", values="rule_count")
    pivot_max_lift = summary_df.pivot(index="min_support", columns="min_confidence", values="max_lift")

    return summary_df, pivot_counts, pivot_max_lift


# =============================================================================
# 5. Visualizations & Figures Export
# =============================================================================

def plot_rainy_vs_dry_evaluation(
    common_df: pd.DataFrame,
    rainy_only_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate 2-panel presentation figure for T11 rainy vs dry evaluation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})

    # Panel A: Scatter plot of Confidence vs Lift on common rules
    if not common_df.empty:
        ax1.scatter(
            common_df["confidence_dry"],
            common_df["lift_dry"],
            color=PALETTE["gray"],
            alpha=0.6,
            s=60,
            label="Dry Days",
            edgecolors=PALETTE["ink2"],
        )
        ax1.scatter(
            common_df["confidence_rain"],
            common_df["lift_rain"],
            color=PALETTE["blue"],
            alpha=0.8,
            s=80,
            label="Rainy Days",
            edgecolors=PALETTE["ink"],
        )

        # Draw shift arrows for top 3 rules
        for idx in range(min(4, len(common_df))):
            row = common_df.iloc[idx]
            ax1.annotate(
                "",
                xy=(row["confidence_rain"], row["lift_rain"]),
                xytext=(row["confidence_dry"], row["lift_dry"]),
                arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], lw=1.5),
            )
            ax1.text(
                row["confidence_rain"] + 0.001,
                row["lift_rain"],
                row["rule_str"][:28] + "...",
                fontsize=8,
                color=PALETTE["ink"],
            )

        ax1.axhline(1.0, color=PALETTE["ink2"], linestyle="--", alpha=0.5, label="Baseline Lift = 1.0")
        ax1.set_xlabel("Rule Confidence")
        ax1.set_ylabel("Rule Lift")
        ax1.set_title("Support & Lift Shift: Rainy vs. Dry Day Rules", fontweight="bold", fontsize=11)
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel B: Top Rainy-Only Rules by Lift
    if not rainy_only_df.empty:
        top_rainy = rainy_only_df.head(8).copy()
        y_pos = np.arange(len(top_rainy))
        bar_colors = [PALETTE["blue"] if "hot_comfort" in aff else PALETTE["aqua"] for aff in top_rainy["affinity_type"]]
        ax2.barh(y_pos, top_rainy["lift"], color=bar_colors, alpha=0.85, edgecolor=PALETTE["ink"])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_rainy["rule_str"], fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Lift on Rainy Days")
        ax2.set_title("Top Weather-Specific Combos (Rainy Days Only)", fontweight="bold", fontsize=11)
        ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

        # Annotate lift values
        for i, val in enumerate(top_rainy["lift"]):
            ax2.text(val + 0.05, i, f"{val:.2f}", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["ink"])

    plt.suptitle("Method 1 (ARM): Rainy vs. Dry Day Rule Evaluation (Coimbatore Orders)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logging.info("Saved figure to %s", save_path)
    return fig


def plot_threshold_sensitivity_grid(
    sensitivity_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate 2-panel presentation figure for T12 threshold sensitivity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})

    pivot_counts = sensitivity_df.pivot(index="min_support", columns="min_confidence", values="rule_count")

    # Panel A: Heatmap of rule counts
    sns.heatmap(
        pivot_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax1,
        linewidths=0.5,
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax1.set_title("Rule Count Matrix (Support × Confidence)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Minimum Confidence Threshold")
    ax1.set_ylabel("Minimum Support Threshold")

    # Panel B: Sensitivity decay curves
    for ms in sensitivity_df["min_support"].unique():
        sub = sensitivity_df[sensitivity_df["min_support"] == ms]
        ax2.plot(
            sub["min_confidence"],
            sub["rule_count"],
            marker="o",
            linewidth=2,
            label=f"min_support = {ms:.4f}",
        )

    ax2.set_xlabel("Minimum Confidence")
    ax2.set_ylabel("Discovered Rule Count")
    ax2.set_title("Rule Threshold Sensitivity Decay Curves", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Task T12: Association Rule Threshold Sensitivity (Coimbatore Food Orders)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logging.info("Saved figure to %s", save_path)
    return fig
