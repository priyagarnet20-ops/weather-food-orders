"""Unit tests for the Coimbatore ordering-habits survey loader.

Run: pytest -q tests/test_survey.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import survey as S  # noqa: E402


def _raw(rows: list[dict]) -> pd.DataFrame:
    base = {
        "Timestamp": "2026-11-04 22:29:41",
        "1. Age group": "18-24",
        "5. On rainy days, you order...": "Less",
        "6. Does the amount of rain matter? [Heavy rain]": "Order less",
        "12. When you order chai, how often do you also order pakora? [On rainy days]": "Often",
        "12. When you order chai, how often do you also order pakora? [On dry days]": "Never",
        "18. How many items are in your typical order?": 2,
        "9. RAINY day: which items are you most likely to order?": "Chai, Samosa",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_short_name_maps_numbered_question():
    assert S.short_name("5. On rainy days, you order...") == "rain_volume"


def test_short_name_maps_bracketed_condition():
    col = "13. When you order chai, how often do you also order samosa? [On dry days]"
    assert S.short_name(col) == "chai_samosa_dry"


def test_short_name_rejects_unknown_question():
    with pytest.raises(KeyError):
        S.short_name("99. Something new")


def test_clean_survey_excludes_under_18_and_timestamp():
    df = S.clean_survey(_raw([{}, {"1. Age group": "Under 18"}]))
    assert len(df) == 1
    assert "Timestamp" not in df.columns
    assert "rain_volume" in df.columns


def test_clean_survey_normalises_basket_size_to_text():
    df = S.clean_survey(_raw([{"18. How many items are in your typical order?": 2}]))
    assert df.loc[0, "basket_size"] == "2"


def test_likert_scores_map_frequency_words():
    assert [S.LIKERT[w] for w in ["Never", "Rarely", "Sometimes", "Often", "Always"]] == [0, 1, 2, 3, 4]


def test_combo_scores_average_per_condition():
    df = S.clean_survey(_raw([
        {"12. When you order chai, how often do you also order pakora? [On rainy days]": "Always"},
        {"12. When you order chai, how often do you also order pakora? [On rainy days]": "Sometimes"},
    ]))
    scores = S.combo_scores(df)
    rainy = scores[(scores["combo"] == "chai_pakora") & (scores["condition"] == "rainy")]
    dry = scores[(scores["combo"] == "chai_pakora") & (scores["condition"] == "dry")]
    assert rainy["mean_score"].item() == pytest.approx(3.0)
    assert dry["mean_score"].item() == pytest.approx(0.0)


def test_volume_direction_counts_less_same_more():
    df = S.clean_survey(_raw([
        {"5. On rainy days, you order...": "Less"},
        {"5. On rainy days, you order...": "A bit more"},
        {"5. On rainy days, you order...": "About the same"},
        {"5. On rainy days, you order...": "A lot more"},
    ]))
    assert S.volume_direction(df, "rain_volume") == {"less": 1, "same": 1, "more": 2}


def test_item_mentions_splits_multi_select_answers():
    df = S.clean_survey(_raw([{}, {"9. RAINY day: which items are you most likely to order?": "Samosa"}]))
    counts = S.item_mentions(df, "rainy_items")
    assert counts["Samosa"] == 2
    assert counts["Chai"] == 1
