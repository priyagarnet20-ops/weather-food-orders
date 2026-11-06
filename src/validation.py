"""Validation of the synthetic Coimbatore dataset against its evidence.

Three groups of checks, one summary row each (results/validation/validation_summary.csv):
    1. grounding  - synthetic food orders vs the real Swiggy orders on the overlapping days
    2. survey     - every volume effect and planted pair points the way the survey says
    3. recovery   - refitting on the synthetic data gives back the planted parameters

Usage:
    python src/validation.py      # writes the CSVs in results/validation/
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import calibration as C
import generator as G
import real_orders
import survey as survey_mod

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results" / "validation"

ALPHA = 0.05
MIN_EXPECTED = 5  # chi-square rule of thumb: merge categories expected below 5
TIE_TOL = 0.03  # pairs the survey rates equally must stay within 3 points of confidence
TOP_ITEMS = 12
SUMMARY_COLUMNS = ["group", "check", "expected", "observed", "passed", "method"]
SURVEY_VOLUME_QUESTIONS = {
    "weekend": "weekend_volume",
    "rain_weekday": "rain_volume",
    "rain_weekend": "rain_volume",
    "heat": "heat_volume",
    "festival": "festival_volume",
}
SYNTH_FORMULA = "n ~ is_weekend * is_rainy + is_hot + is_festival"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def check_row(group: str, check: str, expected: str, observed: str, passed: bool, method: str) -> dict:
    return dict(zip(SUMMARY_COLUMNS, [group, check, expected, observed, bool(passed), method]))


def chi_square_fit(observed: pd.Series, expected_share: pd.Series) -> tuple[float, float]:
    """Goodness of fit of observed counts to expected shares; sparse categories are merged."""
    shares = expected_share.reindex(observed.index).fillna(0)
    shares = shares / shares.sum()
    expected = shares * observed.sum()
    order = expected.sort_values(ascending=False).index
    obs, exp = observed[order].astype(float).copy(), expected[order].copy()
    small = exp < MIN_EXPECTED
    if small.any():
        obs = pd.concat([obs[~small], pd.Series({"other": obs[small].sum()})])
        exp = pd.concat([exp[~small], pd.Series({"other": exp[small].sum()})])
        if exp["other"] < MIN_EXPECTED:
            obs, exp = obs.drop("other"), exp.drop("other")
            exp = exp / exp.sum() * obs.sum()
    stat, p = stats.chisquare(obs, exp)
    return float(stat), float(p)


def size_bins(sizes: pd.Series) -> pd.Series:
    return sizes.map(lambda k: "4+" if k >= 4 else str(k))


def pair_stats(baskets: pd.Series, trigger: str, addon: str) -> dict[str, float]:
    has_t = baskets.map(lambda b: any(G.triggers(item, trigger) for item in b))
    has_a = baskets.map(lambda b: addon in b)
    p_t, p_a, p_ta = has_t.mean(), has_a.mean(), (has_t & has_a).mean()
    return {
        "trigger_orders": int(has_t.sum()),
        "confidence": p_ta / p_t if p_t else np.nan,
        "lift": p_ta / (p_t * p_a) if p_t and p_a else np.nan,
    }


def same_direction(multiplier: float, direction: str) -> bool:
    if direction == "more":
        return multiplier > 1
    if direction == "less":
        return multiplier < 1
    return True


def survey_majority(counts: dict[str, int]) -> str:
    """'more' or 'less' when one side outnumbers the other, else 'same'."""
    if counts["more"] > counts["less"]:
        return "more"
    if counts["less"] > counts["more"]:
        return "less"
    return "same"


def ordering_matches(survey_vals: dict[str, float], synth_vals: dict[str, float], tol: float = TIE_TOL) -> bool:
    """Every pairwise ordering in the survey holds in the synthetic data (ties within tol)."""
    for a, b in combinations(survey_vals, 2):
        s_diff = survey_vals[a] - survey_vals[b]
        y_diff = synth_vals[a] - synth_vals[b]
        if abs(s_diff) < 1e-9:
            if abs(y_diff) > tol:
                return False
        elif np.sign(s_diff) != np.sign(y_diff):
            return False
    return True


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_synthetic(raw_dir: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(raw_dir / "daily_orders.csv")
    orders = pd.read_csv(raw_dir / "orders.csv")
    orders = orders.assign(
        basket=orders["items"].str.split("; "),
        is_snack=orders["items"].map(G.is_snack_order),
        hour=pd.to_datetime(orders["timestamp"]).dt.hour,
    )
    return daily, orders


def overlap_dates(real: pd.DataFrame, daily: pd.DataFrame) -> list[str]:
    return sorted(set(real["date"]) & set(daily["date"]))


# -----------------------------------------------------------------------------
# 1. Grounding against the Swiggy orders
# -----------------------------------------------------------------------------
def grounding(real: pd.DataFrame, real_items: pd.DataFrame,
              daily: pd.DataFrame, orders: pd.DataFrame) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    dates = overlap_dates(real, daily)
    r = real[real["date"].isin(dates)]
    ri = real_items[real_items["order_no"].isin(r["order_no"])]
    food = orders[orders["date"].isin(dates) & ~orders["is_snack"]]
    rows, tables = [], {}

    hours = pd.DataFrame({"real": r["hour"].value_counts(normalize=True),
                          "synthetic": food["hour"].value_counts(normalize=True)}).fillna(0).sort_index()
    stat, p = chi_square_fit(r["hour"].value_counts(), hours["synthetic"])
    rows.append(check_row("grounding", "Order hour profile", "same as Swiggy",
                          f"chi2={stat:.1f}, p={p:.2f}", p > ALPHA, "chi-square goodness of fit"))
    tables["hours"] = hours

    sizes = pd.DataFrame({"real": size_bins(r["item_count"]).value_counts(normalize=True),
                          "synthetic": size_bins(food["item_count"]).value_counts(normalize=True)}).fillna(0)
    stat, p = chi_square_fit(size_bins(r["item_count"]).value_counts(), sizes["synthetic"])
    rows.append(check_row("grounding", "Food basket size", "same as Swiggy",
                          f"chi2={stat:.1f}, p={p:.2f}", p > ALPHA, "chi-square goodness of fit"))
    tables["basket_size"] = sizes

    real_value = ri.groupby("order_no")["line_price"].sum()
    ks = stats.ks_2samp(real_value, food["order_value"])
    rows.append(check_row("grounding", "Food order value", "same distribution as Swiggy",
                          f"KS D={ks.statistic:.3f}, p={ks.pvalue:.2f}", ks.pvalue > ALPHA, "two-sample KS"))
    tables["order_value"] = pd.DataFrame({"real": real_value.describe(), "synthetic": food["order_value"].describe()})

    real_mix = ri["item_name"].value_counts()
    synth_mix = food["basket"].explode().value_counts(normalize=True)
    stat, p = chi_square_fit(real_mix, synth_mix)
    rows.append(check_row("grounding", "Food item mix", "same as Swiggy",
                          f"chi2={stat:.1f}, p={p:.2f}", p > ALPHA, "chi-square goodness of fit"))
    tables["item_mix"] = pd.DataFrame({"real": real_mix / real_mix.sum(),
                                       "synthetic": synth_mix}).fillna(0).head(TOP_ITEMS)

    real_daily = r.groupby("date").size().reindex(dates, fill_value=0)
    synth_daily = daily.set_index("date")["order_count"].reindex(dates)
    corr, p = stats.pearsonr(real_daily, synth_daily)
    rows.append(check_row("grounding", "Day-to-day volume pattern", "positive correlation with Swiggy",
                          f"r={corr:.2f}, p={p:.3f}", corr > 0 and p < ALPHA, "Pearson correlation"))
    tables["daily"] = pd.DataFrame({"real": real_daily, "synthetic_per_outlet": synth_daily / G.OUTLETS})
    return rows, tables


# -----------------------------------------------------------------------------
# 2 + 3. Volume effects: survey direction and recovery
# -----------------------------------------------------------------------------
def fit_synthetic_volume(daily: pd.DataFrame) -> dict[str, dict[str, float]]:
    d = daily.assign(n=daily["order_count"], **{c: daily[c].astype(int) for c in
                                                ("is_weekend", "is_rainy", "is_hot", "is_festival")})
    fit = smf.glm(SYNTH_FORMULA, data=d, family=sm.families.Poisson()).fit(scale="X2")
    return {
        "weekend": C.glm_effect(fit, ["is_weekend"]),
        "rain_weekday": C.glm_effect(fit, ["is_rainy"]),
        "rain_weekend": C.glm_effect(fit, ["is_rainy", "is_weekend:is_rainy"]),
        "heat": C.glm_effect(fit, ["is_hot"]),
        "festival": C.glm_effect(fit, ["is_festival"]),
    }


def volume_checks(daily: pd.DataFrame, cal: C.Calibration,
                  responses: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    fitted = fit_synthetic_volume(daily)
    planted = {k: cal.volume[k]["value"] for k in ("weekend", "rain_weekday", "rain_weekend", "heat")}
    planted["festival"] = cal.festival
    rows, table = [], []
    for name, est in fitted.items():
        recovered = est["low"] <= planted[name] <= est["high"]
        rows.append(check_row("recovery", f"Volume effect: {name}", f"planted x{planted[name]:.2f}",
                              f"x{est['value']:.2f} [{est['low']:.2f}, {est['high']:.2f}]",
                              recovered, "quasi-Poisson GLM on 1,097 synthetic days, 95% CI"))
        counts = survey_mod.volume_direction(responses, SURVEY_VOLUME_QUESTIONS[name])
        direction = survey_majority(counts)
        rows.append(check_row("survey", f"Direction: {name}", f"survey says {direction} {counts}",
                              f"x{est['value']:.2f}", same_direction(est["value"], direction),
                              "sign of synthetic effect vs survey majority"))
        swiggy = cal.volume.get(name)
        table.append({
            "effect": name, "planted": planted[name],
            "synthetic": est["value"], "synthetic_low": est["low"], "synthetic_high": est["high"],
            "swiggy_low": swiggy["low"] if swiggy else np.nan,
            "swiggy_high": swiggy["high"] if swiggy else np.nan,
            "survey_less": counts["less"], "survey_same": counts["same"], "survey_more": counts["more"],
        })
    return rows, pd.DataFrame(table)


# -----------------------------------------------------------------------------
# 2 + 3. Planted pairs and snack mix
# -----------------------------------------------------------------------------
def pair_table(daily: pd.DataFrame, orders: pd.DataFrame, cal: C.Calibration) -> pd.DataFrame:
    regime = daily.set_index("date")["regime"]
    o = orders.assign(regime=orders["date"].map(regime))
    rows = []
    for (trigger, addon), probs in cal.pair_probs.items():
        for reg in C.REGIMES:
            s = pair_stats(o.loc[o["regime"] == reg, "basket"], trigger, addon)
            rows.append({"trigger": trigger, "addon": addon, "regime": reg, "planted": probs[reg], **s})
    return pd.DataFrame(rows)


def pair_checks(pairs: pd.DataFrame) -> list[dict]:
    # Bonferroni: one two-sided test per pair x regime cell, family-wise 5%
    z_crit = stats.norm.ppf(1 - ALPHA / (2 * len(pairs)))
    rows = []
    for (trigger, addon), g in pairs.groupby(["trigger", "addon"], sort=False):
        planted = dict(zip(g["regime"], g["planted"]))
        observed = dict(zip(g["regime"], g["confidence"]))
        # rainy-vs-dry questions plant the same value on normal and hot days: compare rainy vs normal only
        asked_rainy_vs_dry = len(set(planted.values())) == 2
        conds = ["rainy", "normal"] if asked_rainy_vs_dry else list(planted)
        rows.append(check_row(
            "survey", f"Pair {trigger} -> {addon}: weather ordering",
            ", ".join(f"{c} {planted[c]:.2f}" for c in conds),
            ", ".join(f"{c} {observed[c]:.2f}" for c in conds),
            ordering_matches({c: planted[c] for c in conds}, {c: observed[c] for c in conds}),
            "confidence by weather vs survey frequency score",
        ))
        se = np.sqrt(g["planted"] * (1 - g["planted"]) / g["trigger_orders"])
        recovered = bool((((g["confidence"] - g["planted"]).abs() <= z_crit * se) & (g["lift"] > 1)).all())
        rows.append(check_row(
            "recovery", f"Pair {trigger} -> {addon}: rule found",
            f"confidence = planted (within {z_crit:.2f} SE, Bonferroni) and lift > 1 in every weather",
            "lift " + ", ".join(f"{r} {l:.1f}" for r, l in zip(g["regime"], g["lift"])),
            recovered, "association rule confidence and lift per regime",
        ))
    return rows


def snack_mix_checks(daily: pd.DataFrame, orders: pd.DataFrame, cal: C.Calibration) -> list[dict]:
    regime = daily.set_index("date")["regime"]
    share = orders.assign(regime=orders["date"].map(regime)).groupby("regime")["is_snack"].mean()
    survey_order = sorted(cal.snack_share, key=cal.snack_share.get, reverse=True)
    synth_order = list(share.sort_values(ascending=False).index)
    return [check_row(
        "survey", "Share of snack/drink orders by weather",
        " > ".join(f"{r} {cal.snack_share[r]:.2f}" for r in survey_order),
        " > ".join(f"{r} {share[r]:.2f}" for r in synth_order),
        survey_order == synth_order, "snack/drink share of orders per regime vs survey Q9-Q11",
    )]


# -----------------------------------------------------------------------------
# Run everything
# -----------------------------------------------------------------------------
def run_all() -> dict[str, pd.DataFrame]:
    cal = C.calibrate()
    real, real_items = real_orders.load_clean()
    responses = survey_mod.load_clean()
    daily, orders = load_synthetic()

    g_rows, g_tables = grounding(real, real_items, daily, orders)
    v_rows, volume = volume_checks(daily, cal, responses)
    pairs = pair_table(daily, orders, cal)
    rows = g_rows + v_rows + pair_checks(pairs) + snack_mix_checks(daily, orders, cal)
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return {"summary": summary, "volume": volume, "pairs": pairs, **g_tables}


def main() -> None:
    results = run_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results["summary"].to_csv(OUT_DIR / "validation_summary.csv", index=False)
    results["volume"].to_csv(OUT_DIR / "volume_effects.csv", index=False)
    results["pairs"].to_csv(OUT_DIR / "pair_rules_by_regime.csv", index=False)
    s = results["summary"]
    print(s.to_string(index=False))
    print(f"\n{s['passed'].sum()} of {len(s)} checks passed -> {OUT_DIR}")


if __name__ == "__main__":
    main()
