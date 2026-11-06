"""Unit tests for deriving generator parameters from the real evidence.

Run: pytest -q tests/test_calibration.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import calibration as C  # noqa: E402


def _days(effects: dict[tuple[int, int], int], reps: int = 40) -> pd.DataFrame:
    """Daily table where (is_weekend, is_rainy) cells have fixed order counts."""
    rows = []
    for (wk, rain), n in effects.items():
        rows += [{"n": n, "is_weekend": wk, "is_rainy": rain, "is_hot": 0}] * reps
    return pd.DataFrame(rows)


def test_fit_volume_effects_recovers_known_cell_means():
    days = _days({(0, 0): 5, (1, 0): 10, (0, 1): 8, (1, 1): 6})
    eff = C.fit_volume_effects(days)
    assert eff["base_weekday_dry"]["value"] == pytest.approx(5.0, rel=1e-6)
    assert eff["weekend"]["value"] == pytest.approx(2.0, rel=1e-6)
    assert eff["rain_weekday"]["value"] == pytest.approx(1.6, rel=1e-6)
    assert eff["rain_weekend"]["value"] == pytest.approx(0.6, rel=1e-6)


def test_fit_volume_effects_reports_interval_around_value():
    days = _days({(0, 0): 5, (1, 0): 10, (0, 1): 8, (1, 1): 6})
    days.loc[::3, "n"] += 2  # add noise so the interval has width
    eff = C.fit_volume_effects(days)
    for key in ("weekend", "rain_weekday", "rain_weekend", "heat"):
        assert eff[key]["low"] <= eff[key]["value"] <= eff[key]["high"]


def test_outlet_dispersion_is_large_for_poisson_like_counts():
    rng = np.random.default_rng(0)
    days = pd.DataFrame({"n": rng.poisson(8, 400), "is_weekend": 0, "is_rainy": 0, "is_hot": 0})
    assert C.outlet_dispersion(days) > 20


def test_outlet_dispersion_detects_overdispersion():
    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(4, 4 / (4 + 8), 2000)
    days = pd.DataFrame({"n": counts, "is_weekend": 0, "is_rainy": 0, "is_hot": 0})
    assert C.outlet_dispersion(days) == pytest.approx(4, rel=0.25)


def test_festival_effect_is_one_plus_net_share():
    survey = pd.DataFrame({"festival_volume": ["Order more", "Order less", "Order less", "Same"]})
    assert C.festival_effect(survey) == pytest.approx(1 - 1 / 4)


def _survey() -> pd.DataFrame:
    return pd.DataFrame({
        "rainy_items": ["Chai, Biryani", "Samosa"],
        "hot_items": ["Ice Cream, Biryani", "Biryani"],
        "normal_items": ["Biryani", "Pizza"],
        "chai_pakora_rainy": ["Always", "Often"],
        "chai_pakora_dry": ["Never", "Rarely"],
        "icecream_coldcoffee_hot": ["Always", "Always"],
        "icecream_coldcoffee_normal": ["Sometimes", "Sometimes"],
        "icecream_coldcoffee_rainy": ["Never", "Never"],
        "basket_size": ["1", "4 or more"],
    })


def test_snack_share_counts_snack_mentions_per_regime():
    share = C.snack_share(_survey())
    assert share["rainy"] == pytest.approx(2 / 3)
    assert share["hot"] == pytest.approx(1 / 3)
    assert share["normal"] == pytest.approx(0.0)


def test_snack_weights_are_smoothed_and_normalised():
    weights = C.snack_weights(_survey())
    for regime in ("rainy", "hot", "normal"):
        assert sum(weights[regime].values()) == pytest.approx(1.0)
        assert min(weights[regime].values()) > 0
    assert weights["rainy"]["Chai"] > weights["rainy"]["Ice Cream"]
    assert weights["hot"]["Ice Cream"] > weights["hot"]["Chai"]


def test_pair_probabilities_map_dry_answers_to_normal_and_hot():
    probs = C.pair_probabilities(_survey())
    assert probs[("Chai", "Pakora")]["rainy"] == pytest.approx(3.5 / 4)
    assert probs[("Chai", "Pakora")]["normal"] == pytest.approx(0.5 / 4)
    assert probs[("Chai", "Pakora")]["hot"] == pytest.approx(0.5 / 4)
    assert probs[("Ice Cream", "Cold Coffee")] == pytest.approx({"hot": 1.0, "normal": 0.5, "rainy": 0.0})


def test_snack_basket_probs_folds_four_or_more_into_four():
    assert C.snack_basket_probs(_survey()) == {1: 0.5, 4: 0.5}


def test_hour_profile_sums_to_one():
    orders = pd.DataFrame({"hour": [12, 12, 13, 20]})
    assert C.hour_profile(orders) == {12: 0.5, 13: 0.25, 20: 0.25}


def test_food_baskets_group_lines_by_order():
    items = pd.DataFrame({
        "order_no": [1, 1, 2],
        "item_name": ["Non Veg Meals", "Veg Clear Soup", "Plain Biryani"],
        "quantity": [1, 1, 2],
        "line_price": [286.0, 108.0, 572.0],
    })
    baskets = C.food_baskets(items)
    assert baskets[0] == (("Non Veg Meals", 286.0), ("Veg Clear Soup", 108.0))
    assert baskets[1] == (("Plain Biryani", 572.0),)


def test_is_soup_matches_soups_and_rasam():
    assert C.is_soup("Chicken Kozhi Rasam")
    assert C.is_soup("Veg Clear Soup")
    assert not C.is_soup("Non Veg Meals")


def test_median_prices_take_median_per_item():
    prices = pd.DataFrame({"item_name": ["Chai", "Chai", "Vada"], "price": [36, 40, 15]})
    assert C.median_prices(prices) == {"Chai": 38.0, "Vada": 15.0}
