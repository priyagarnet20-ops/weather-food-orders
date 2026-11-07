"""Unit tests for Association Rule Mining & Rule Evaluation (T16, T17, T11, T12).

Target: Coimbatore, Tamil Nadu, India
Run: pytest -q tests/test_arm.py
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arm as ARM  # noqa: E402


@pytest.fixture(scope="module")
def arm_data():
    orders, menu = ARM.load_arm_data()
    return orders, menu


@pytest.fixture(scope="module")
def encoded_overall(arm_data):
    orders, _ = arm_data
    encoded_df, te = ARM.encode_baskets(orders, multi_only=True)
    return encoded_df, te


def test_load_arm_data(arm_data):
    """Verify orders, weather features, and menu load properly with required fields."""
    orders, menu = arm_data
    assert len(orders) == 182696
    assert len(menu) == 80
    assert "items" in orders.columns
    assert "is_rainy" in orders.columns
    assert "weather_affinity" in menu.columns


def test_encode_baskets(encoded_overall):
    """Verify transaction encoding produces expected boolean matrix."""
    encoded_df, te = encoded_overall
    assert len(encoded_df) == 82990  # multi-item transactions
    assert encoded_df.shape[1] >= 75  # unique items in multi-item orders
    assert encoded_df.dtypes.iloc[0] == bool
    # Each multi-item transaction should have at least 2 items
    assert (encoded_df.sum(axis=1) >= 2).all()


def test_segment_transactions(arm_data):
    """Verify weather segmentation partitions transactions correctly."""
    orders, _ = arm_data
    segs = ARM.segment_transactions(orders, multi_only=True)
    assert set(segs.keys()) == {"overall", "rainy", "dry", "heavy_rain", "hot"}

    n_total = len(segs["overall"])
    n_rain = len(segs["rainy"])
    n_dry = len(segs["dry"])

    assert n_total == 82990
    assert n_rain == 23926
    assert n_dry == 59064
    assert n_total == n_rain + n_dry
    assert len(segs["heavy_rain"]) == 4762
    assert len(segs["hot"]) == 26780


def test_mine_frequent_itemsets(encoded_overall):
    """Verify frequent itemsets can be extracted with fpgrowth and apriori."""
    encoded_df, _ = encoded_overall
    # Sample subset for speed
    sub = encoded_df.iloc[:1000]

    fi_fp = ARM.mine_frequent_itemsets(sub, min_support=0.01, algorithm="fpgrowth")
    fi_ap = ARM.mine_frequent_itemsets(sub, min_support=0.01, algorithm="apriori")

    assert not fi_fp.empty
    assert "support" in fi_fp.columns
    assert "length" in fi_fp.columns
    # Both algorithms should yield identical itemsets
    assert len(fi_fp) == len(fi_ap)


def test_mine_association_rules(encoded_overall):
    """Verify association rules generation and string representation formatting."""
    encoded_df, _ = encoded_overall
    sub = encoded_df.iloc[:2000]
    fi = ARM.mine_frequent_itemsets(sub, min_support=0.005, algorithm="fpgrowth")
    rules = ARM.mine_association_rules(fi, metric="confidence", min_threshold=0.05, min_lift=1.0)

    assert not rules.empty
    assert "rule_str" in rules.columns
    assert "antecedents_str" in rules.columns
    assert "consequents_str" in rules.columns
    assert "lift" in rules.columns
    assert (rules["lift"] >= 1.0).all()
    assert (rules["confidence"] >= 0.05).all()


def test_evaluate_rainy_vs_dry_rules(arm_data):
    """Verify T11 rainy vs dry comparison metrics, shifts, and rainy-only flags."""
    orders, menu = arm_data
    segs = ARM.segment_transactions(orders, multi_only=True)

    # Use 3000 rows each for fast test execution
    rain_sample = segs["rainy"].iloc[:3000]
    dry_sample = segs["dry"].iloc[:3000]

    enc_rain, _ = ARM.encode_baskets(rain_sample)
    enc_dry, _ = ARM.encode_baskets(dry_sample)

    fi_rain = ARM.mine_frequent_itemsets(enc_rain, min_support=0.005)
    fi_dry = ARM.mine_frequent_itemsets(enc_dry, min_support=0.005)

    rules_rain = ARM.mine_association_rules(fi_rain, min_threshold=0.05)
    rules_dry = ARM.mine_association_rules(fi_dry, min_threshold=0.05)

    eval_result = ARM.evaluate_rainy_vs_dry_rules(rules_rain, rules_dry, menu)

    assert "common" in eval_result
    assert "rainy_only" in eval_result
    assert "dry_only" in eval_result

    common = eval_result["common"]
    if not common.empty:
        assert "lift_diff" in common.columns
        assert "lift_ratio" in common.columns
        assert "affinity_type" in common.columns
        # Verify planted pairs table extraction
        planted_tbl = ARM.extract_planted_pairs_table(common)
        assert not planted_tbl.empty
        assert "planted_rain" in planted_tbl.columns
        assert "conf_rain" in planted_tbl.columns
        assert len(planted_tbl) >= 3

    rainy_only = eval_result["rainy_only"]
    if not rainy_only.empty:
        assert "rule_str" in rainy_only.columns
        assert "affinity_type" in rainy_only.columns
        # Ensure rainy_only rules are genuinely not in dry rules
        assert not rainy_only["rule_str"].isin(rules_dry["rule_str"]).any()


def test_evaluate_threshold_sensitivity(encoded_overall):
    """Verify T12 threshold sensitivity grid, monotonicity, and pivot tables."""
    encoded_df, _ = encoded_overall
    sub = encoded_df.iloc[:2000]
    supports = [0.008, 0.012]
    confidences = [0.05, 0.10]

    summary_df, p_cnt, p_max = ARM.evaluate_threshold_sensitivity(
        sub, support_grid=supports, confidence_grid=confidences
    )

    assert len(summary_df) == len(supports) * len(confidences)
    assert "rule_count" in summary_df.columns
    assert "max_lift" in summary_df.columns
    assert "top_rule" in summary_df.columns

    assert p_cnt.shape == (2, 2)
    # Monotonicity: higher confidence at fixed support must have <= rules
    for ms in supports:
        cnt_05 = p_cnt.loc[ms, 0.05]
        cnt_10 = p_cnt.loc[ms, 0.10]
        assert cnt_05 >= cnt_10


def test_visualization_functions(arm_data, tmp_path):
    """Verify presentation plotting functions run and save image files."""
    orders, menu = arm_data
    segs = ARM.segment_transactions(orders, multi_only=True)

    enc_rain, _ = ARM.encode_baskets(segs["rainy"].iloc[:2000])
    enc_dry, _ = ARM.encode_baskets(segs["dry"].iloc[:2000])

    fi_rain = ARM.mine_frequent_itemsets(enc_rain, min_support=0.005)
    fi_dry = ARM.mine_frequent_itemsets(enc_dry, min_support=0.005)

    r_rain = ARM.mine_association_rules(fi_rain, min_threshold=0.05)
    r_dry = ARM.mine_association_rules(fi_dry, min_threshold=0.05)

    eval_result = ARM.evaluate_rainy_vs_dry_rules(r_rain, r_dry, menu)

    fig1_path = tmp_path / "fig_arm_rain_vs_dry.png"
    planted = ARM.extract_planted_pairs_table(eval_result["common"])
    fig1 = ARM.plot_rainy_vs_dry_evaluation(planted, save_path=fig1_path)
    assert fig1_path.exists()
    assert fig1_path.stat().st_size > 0

    enc_all, _ = ARM.encode_baskets(orders.iloc[:2000], multi_only=True)
    summary_df, _, _ = ARM.evaluate_threshold_sensitivity(
        enc_all, support_grid=[0.008, 0.012], confidence_grid=[0.05, 0.10]
    )
    fig2_path = tmp_path / "fig_arm_sensitivity.png"
    fig2 = ARM.plot_threshold_sensitivity_grid(summary_df, save_path=fig2_path)
    assert fig2_path.exists()
    assert fig2_path.stat().st_size > 0
