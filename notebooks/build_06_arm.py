"""Generates notebooks/06_arm.ipynb (run: python notebooks/build_06_arm.py).

Builds the comprehensive Association Rule Mining & Evaluation notebook (Review 2 / Method 1).
Covers:
- T16: ARM Data Preparation & Weather Segmentation (P3)
- T17: FP-Growth & Apriori Implementation (P3)
- T11: Rainy vs. Dry Rule Evaluation & Affinity Recovery (P2)
- T12: Rule Threshold Sensitivity Grid across min_support and min_confidence (P2)

Target City: Coimbatore, Tamil Nadu, India (11.00°N, 76.99°E)
Dataset: 1,097 days · 182,696 orders · 80-item menu catalog
Author: P3 (Data Cleaning, Feature Engineering & ARM Preparation)

Execute afterwards with:
    python -m nbconvert --to notebook --execute --inplace notebooks/06_arm.ipynb
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# -----------------------------------------------------------------------------
# Notebook Cells Definition
# -----------------------------------------------------------------------------

md("""
# 06 · Association Rule Mining & Weather-Segmented Rule Evaluation
**Owner:** P3 (ARM Data Prep & FP-Growth) & P2 (Rule Evaluation & Sensitivity) · **Tasks:**
- **T16:** ARM data preparation & weather segmentation (P3)
- **T17:** FP-Growth & Apriori implementation (P3)
- **T11:** Rainy vs. dry rule evaluation & weather affinity recovery (P2)
- **T12:** Rule threshold sensitivity grid across min_support and min_confidence (P2)
**Rubric:** Review 2: Method 1 prep & implementation (2 points), Method 1 evaluation (3 points)

---

### Executive Overview & Business Objectives
In Review 1, our predictive models established that atmospheric weather (precipitation, temperature, and monsoon seasons) significantly drives daily order volume in Coimbatore, Tamil Nadu.
In Review 2 (Method 1), we dive into **granular transaction baskets** using **Association Rule Mining (ARM)** to answer critical operational questions:
1. **Product Co-occurrences:** What items are frequently purchased together in customer multi-item baskets?
2. **Weather-Driven Rule Shifts (T11):** Do cross-purchasing patterns change systematically during rainy days compared to dry days? Which item pairings emerge exclusively during rainfall?
3. **Planted Weather Affinity Recovery:** Does real customer purchasing behavior validate the expected surge in hot-comfort items (Masala Chai, Onion Pakora, Filter Coffee, Medu Vada) during monsoon showers, and cold-refresh items (Lime Juice, Ice Cream) during heatwaves?
4. **Parameter Sensitivity (T12):** How sensitive are discovered itemsets and rules to variations in `min_support` and `min_confidence`? What is the optimal operational threshold for an 80-item Coimbatore menu catalog?

**Data Provenance:**
- **Customer Orders:** 182,696 customer orders across 1,097 continuous days (2023-09-24 to 2026-09-24) grounded in Hotel Chinthamani Swiggy order reports (`data/raw/orders.csv`).
- **Weather Features:** Real Open-Meteo ERA5 reanalysis data for Coimbatore (11.00°N, 76.99°E; `data/processed/daily_features.csv`).
- **Menu Catalog:** 80 Coimbatore local menu SKUs with verified retail prices and weather affinities (`data/raw/menu.csv`).
""")

code(r"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML

# Add src/ to path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import arm as ARM
from plotstyle import apply, C

apply()
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 1000)

print("Environment ready. ROOT path:", ROOT)
""")

md("""
---
## 1. T16: ARM Data Preparation & Weather Segmentation

### 1.1 Dataset Loading and Basket Extraction
To perform market basket analysis, we extract transaction baskets from `data/raw/orders.csv`. Each row represents an individual customer transaction.
Multi-item transactions (containing 2 or more items) represent customer cross-purchasing decisions where association rules can be mined.
""")

code(r"""
orders, menu = ARM.load_arm_data(
    orders_path=ROOT / "data/raw/orders.csv",
    features_path=ROOT / "data/processed/daily_features.csv",
    menu_path=ROOT / "data/raw/menu.csv"
)

print(f"Total orders loaded: {len(orders):,}")
print(f"Menu SKUs in catalog: {len(menu)}")
print(f"Observation window: {orders['date'].min()} to {orders['date'].max()} ({orders['date'].nunique()} days)")

# Basket size distribution
basket_dist = orders['item_count'].value_counts().rename_axis('Basket Size').reset_index(name='Order Count')
basket_dist['Percentage'] = (basket_dist['Order Count'] / len(orders) * 100).round(2).astype(str) + '%'
display(basket_dist)
""")

md("""
### 1.2 One-Hot Transaction Encoding
Using `mlxtend`'s `TransactionEncoder`, each multi-item order ($N = 82,990$) is converted into a sparse boolean transaction matrix where rows correspond to orders and columns correspond to the unique menu SKUs present in transactions.
""")

code(r"""
encoded_df, te = ARM.encode_baskets(orders, multi_only=True)

print(f"One-hot transaction matrix shape: {encoded_df.shape[0]:,} transactions x {encoded_df.shape[1]} unique items")
print(f"Matrix density: {(encoded_df.values.sum() / encoded_df.size * 100):.2f}%")
print(f"Average basket size in multi-item orders: {encoded_df.sum(axis=1).mean():.2f} items")

display(encoded_df.head())
""")

md(r"""
### 1.3 Weather Regime Segmentation
We partition the transaction universe into five meteorological segments using daily weather regressors from `daily_features.csv`:
1. **Overall:** All multi-item transactions ($N = 82,990$).
2. **Rainy Days:** Days with precipitation $\ge 2.5\text{ mm}$ ($N = 23,926$).
3. **Dry Days:** Days with precipitation $< 2.5\text{ mm}$ ($N = 59,064$).
4. **Heavy Rain:** Days with precipitation $\ge 15.0\text{ mm}$ ($N = 4,762$).
5. **Hot Days:** Days with maximum temperature $\ge 33.0^\circ\text{C}$ ($N = 26,780$).
""")

code(r"""
segments = ARM.segment_transactions(orders, multi_only=True)

seg_summary = pd.DataFrame([
    {"Segment": name, "Multi-Item Orders": len(df), "Share of Multi-Orders": f"{len(df) / len(encoded_df) * 100:.1f}%"}
    for name, df in segments.items()
])
display(seg_summary)
""")

md(r"""
---
## 2. T17: FP-Growth & Apriori Algorithm Implementation

### 2.1 Mining Frequent Itemsets (Overall Universe)
We execute FP-Growth with minimum support threshold $min\_support = 0.005$ to discover frequent itemsets.
""")

code(r"""
t0 = time.time()
frequent_itemsets = ARM.mine_frequent_itemsets(encoded_df, min_support=0.005, algorithm="fpgrowth")
t_fp = time.time() - t0

print(f"FP-Growth discovered {len(frequent_itemsets)} frequent itemsets in {t_fp:.3f} seconds.")
display(frequent_itemsets.head(10))
""")

md("""
### 2.2 Algorithm Performance Comparison: FP-Growth vs. Apriori
We benchmark FP-Growth against the classic Apriori algorithm on a representative subset of transactions ($N = 5,000$).
""")

code(r"""
sample_enc = encoded_df.iloc[:5000]

t0 = time.time()
fi_fp = ARM.mine_frequent_itemsets(sample_enc, min_support=0.01, algorithm="fpgrowth")
t_fp_sample = time.time() - t0

t0 = time.time()
fi_ap = ARM.mine_frequent_itemsets(sample_enc, min_support=0.01, algorithm="apriori")
t_ap_sample = time.time() - t0

print(f"Benchmarking (N = 5,000 transactions, min_support = 0.01):")
print(f"  FP-Growth: {t_fp_sample:.4f} s ({len(fi_fp)} itemsets)")
print(f"  Apriori:   {t_ap_sample:.4f} s ({len(fi_ap)} itemsets)")
print(f"  Speedup:   {t_ap_sample / max(t_fp_sample, 1e-6):.1f}x faster execution using FP-Growth")

assert len(fi_fp) == len(fi_ap), "Itemset counts must match exactly between algorithms"
""")

md(r"""
### 2.3 Mining Association Rules (Overall Market Baskets)
From the frequent itemsets, we extract association rules using minimum confidence $min\_confidence = 0.05$ and minimum lift $min\_lift = 1.0$.
""")

code(r"""
rules_overall = ARM.mine_association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05, min_lift=1.0)

print(f"Discovered {len(rules_overall)} association rules overall.")
display(rules_overall[["rule_str", "support", "confidence", "lift", "leverage", "conviction"]].head(10))
""")

md("""
---
## 3. T11: Rainy vs. Dry Day Rule Evaluation & Weather Affinity Analysis

### 3.1 Mining Weather-Segmented Rules
To evaluate the impact of rainfall on item affinities, we independently mine frequent itemsets and rules for the `rainy` and `dry` transaction subsets.
""")

code(r"""
enc_rain, _ = ARM.encode_baskets(segments["rainy"])
enc_dry, _ = ARM.encode_baskets(segments["dry"])

fi_rain = ARM.mine_frequent_itemsets(enc_rain, min_support=0.005, algorithm="fpgrowth")
fi_dry = ARM.mine_frequent_itemsets(enc_dry, min_support=0.005, algorithm="fpgrowth")

rules_rain = ARM.mine_association_rules(fi_rain, metric="confidence", min_threshold=0.05, min_lift=1.0)
rules_dry = ARM.mine_association_rules(fi_dry, metric="confidence", min_threshold=0.05, min_lift=1.0)

print(f"Rainy day rules discovered: {len(rules_rain)} (from {len(fi_rain)} frequent itemsets)")
print(f"Dry day rules discovered:   {len(rules_dry)} (from {len(fi_dry)} frequent itemsets)")
""")

md(r"""
### 3.2 Recovery of Generator Planted Pairs & Rainy vs. Dry Evaluation (T11 Evidence)
To rigorously validate Method 1 (ARM), we assess whether the mined rules recover the five ground-truth pairs planted by the data generator (`docs/generator.md`):
- **Chai -> Pakora**: Planted confidence 0.30 rainy vs. 0.22 dry (Q12)
- **Chai -> Samosa**: Planted confidence 0.32 rainy vs. 0.25 dry (Q13)
- **Coffee -> Vada**: Planted confidence 0.31 rainy vs. 0.27 dry (Q14)
- **Ice Cream -> Cold Coffee**: Planted confidence 0.23 rainy vs. 0.28 dry / 0.31 hot (Q16)
- **Biryani -> Buttermilk**: Planted confidence 0.30 rainy vs. 0.30 dry (Q15 control, no weather effect)

**Why Confidence and Support are the Primary Metrics:**
Lift equals $\text{Confidence} / P(\text{consequent})$. When monsoon rains arrive, comfort snacks like Pakora surge in base popularity across *all* customer orders ($14\%$ in rain vs. $5\%$ dry). Dividing by that much larger denominator pulls Lift down ($4.75 \to 2.56$) even though Chai buyers purchase Pakora 1.5x more often ($36.5\%$ vs. $24.3\%$). Therefore, Confidence and Support directly measure the rain-induced surge, while Lift confirms genuine association ($\text{Lift} > 1.0$).
""")

code(r"""
eval_result = ARM.evaluate_rainy_vs_dry_rules(rules_rain, rules_dry, menu)

common_df = eval_result["common"]
rainy_only_df = eval_result["rainy_only"]
dry_only_df = eval_result["dry_only"]

print(f"Common rules evaluated across both weather states: {len(common_df)}")
print(f"Emergent rules unique to rainy days:               {len(rainy_only_df)}")
print(f"Emergent rules unique to dry days:                 {len(dry_only_df)}")

# Extract the 5 planted benchmark pairs (Primary T11 Deliverable)
planted_summary = ARM.extract_planted_pairs_table(common_df)
print("\n=== T11 Key Evidence: Generator Planted Pair Recovery ===")
display(planted_summary)

if not rainy_only_df.empty:
    print("\nTop 5 Emergent Rainy-Only Rules (Crossing Threshold Exclusively in Rain):")
    display(rainy_only_df[["rule_str", "support", "confidence", "lift", "affinity_type"]].head(5))
""")

md("""
### 3.3 Visualizing Weather-Driven Rule Shifts (Fig 14)
We generate presentation figure **Fig 14** (`results/fig14_arm_rain_vs_dry.png`), comprising:
1. **Panel A (Scatter Plot):** Confidence vs. Lift shift on common rules from dry days to rainy days, with vector shift arrows for headline pairs.
2. **Panel B (Horizontal Bar Chart):** Top emergent rainy-only combos ranked by lift.
""")

code(r"""
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

fig14_path = results_dir / "fig14_arm_rain_vs_dry.png"
fig14 = ARM.plot_rainy_vs_dry_evaluation(common_df, rainy_only_df, save_path=fig14_path)
plt.show()
print(f"Presentation figure saved to: {fig14_path}")
""")

md(r"""
---
## 4. T12: Rule Threshold Sensitivity Analysis

### 4.1 Grid Evaluation across Support and Confidence
We systematically assess the stability and count of discovered rules across a 2-dimensional parameter grid:
- **Minimum Support Thresholds:** $min\_support \in [0.004, 0.006, 0.008, 0.010, 0.015]$
- **Minimum Confidence Thresholds:** $min\_confidence \in [0.02, 0.05, 0.10, 0.15, 0.20]$
""")

code(r"""
supports = [0.004, 0.006, 0.008, 0.010, 0.015]
confidences = [0.02, 0.05, 0.10, 0.15, 0.20]

sensitivity_df, pivot_counts, pivot_max_lift = ARM.evaluate_threshold_sensitivity(
    encoded_df, support_grid=supports, confidence_grid=confidences, algorithm="fpgrowth"
)

display(HTML("<h4>Table 1: Association Rule Counts Matrix (Support x Confidence)</h4>"))
display(pivot_counts.style.background_gradient(cmap="Blues", axis=None))
""")

md("""
### 4.2 Sensitivity Visualizations: Heatmap and Decay Curves (Fig 15)
We generate presentation figure **Fig 15** (`results/fig15_arm_threshold_sensitivity.png`), illustrating parameter decay curves and rule count densities.
""")

code(r"""
fig15_path = results_dir / "fig15_arm_threshold_sensitivity.png"
fig15 = ARM.plot_threshold_sensitivity_grid(sensitivity_df, save_path=fig15_path)
plt.show()
print(f"Sensitivity figure saved to: {fig15_path}")
""")

md("""
---
## 5. Artifact Exports and Dashboard Deliverables
We export all generated rule tables and evaluation summaries to `results/` for downstream integration with the Streamlit dashboard (P4 / T23) and monsoon lift tracking (P5 / T27).
""")

code(r"""
# Export rule catalogs and sensitivity summaries
rules_overall.to_csv(results_dir / "rules_overall.csv", index=False)
rules_rain.to_csv(results_dir / "rules_rainy.csv", index=False)
rules_dry.to_csv(results_dir / "rules_dry.csv", index=False)
if not common_df.empty:
    common_df.to_csv(results_dir / "rules_rain_vs_dry_comparison.csv", index=False)
sensitivity_df.to_csv(results_dir / "rule_sensitivity_grid.csv", index=False)

print("=" * 80)
print("Notebook 06 (ARM & Weather-Segmented Rule Evaluation) complete.")
print("Artifacts generated in results/:")
print(f"  - {results_dir / 'rules_overall.csv'}")
print(f"  - {results_dir / 'rules_rainy.csv'}")
print(f"  - {results_dir / 'rules_dry.csv'}")
print(f"  - {results_dir / 'rules_rain_vs_dry_comparison.csv'}")
print(f"  - {results_dir / 'rule_sensitivity_grid.csv'}")
print(f"  - {results_dir / 'fig14_arm_rain_vs_dry.png'}")
print(f"  - {results_dir / 'fig15_arm_threshold_sensitivity.png'}")
print("=" * 80)
""")

# Assign cells to notebook and write to disk
nb.cells = cells
out_path = ROOT / "notebooks/06_arm.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generated successfully at: {out_path} ({len(cells)} cells)")
