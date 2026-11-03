"""Time-series preparation, forecasting models and evaluation (P1 / T04-T06).

Public functions are used by notebooks/07_timeseries.ipynb.

Design choices (all documented in the notebook):
  * Target is the daily order count. Real weather (data/raw/weather.csv) is joined on date.
  * The order series is 731 days: 272 real POS days (2025-01-02 to 2025-09-30) and 459 synthetic days
    (`is_synthetic = 1`). Only real days are ever scored; synthetic days may be used as training history.
  * Weather regressors use the team-wide regime thresholds (rain >= 2.5 mm, heavy >= 15 mm,
    hot >= 33 C). Day-of-week enters as dummies, so the model never has to be told which
    days the generator treats as "weekend".
  * Models are compared with a rolling-origin backtest: fit on everything up to an origin,
    forecast the next 7 days, move the origin forward 7 days, repeat. Hyper-parameters are
    chosen only on the first `initial` days, i.e. before any backtest day is seen.
  * Weather regressors are the *observed* weather of the forecast days (an "oracle").
    A live system would use a weather forecast, so accuracy here is an upper bound.
"""
from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]

RAIN_MM = 2.5     # meaningful rain (team regime definition)
HEAVY_MM = 15.0   # heavy rain
HOT_C = 33.0      # hot day (temp_max)
WEATHER_COLS = ["is_rainy", "heavy_rain", "is_hot"]
CALENDAR_COLS = ["is_festival"]   # known in advance; used by every model that takes regressors

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)


# =============================================================================
# T04  data preparation
# =============================================================================
def load_daily(weather_csv: str | Path, orders_csv: str | Path) -> pd.DataFrame:
    """Daily order series joined with weather; indexed by date, one row per order day.

    `orders_csv` may be a daily table (date, order_count) or an order-level table
    (date or timestamp column plus order_id), which is aggregated to daily counts.
    The index follows the order dates (the weather file is longer than the order series).
    Optional columns `is_festival` and `is_synthetic` are carried through (default 0).
    """
    w = pd.read_csv(weather_csv, parse_dates=["date"]).set_index("date")
    o = pd.read_csv(orders_csv)
    if "order_count" in o.columns:
        o = o.assign(date=pd.to_datetime(o["date"]).dt.normalize())
        y = o.drop_duplicates("date").set_index("date")[["order_count"]].astype(float)
        by_day = o.set_index("date")
    else:
        date_col = next(c for c in ("date", "order_time", "timestamp", "datetime") if c in o.columns)
        o = o.assign(date=pd.to_datetime(o[date_col]).dt.normalize())
        y = o.groupby("date")["order_id"].nunique().rename("order_count").astype(float).to_frame()
        by_day = o.set_index("date")
    df = y.sort_index().join(w, how="left").asfreq("D")
    for flag in ("is_festival", "is_synthetic"):
        if flag in by_day.columns:
            f = by_day[flag].astype(int).groupby(level=0).max()
            df[flag] = f.reindex(df.index).fillna(0).astype(int)
        else:
            df[flag] = 0
    df["is_rainy"] = (df["precip_mm"] >= RAIN_MM).astype(int)
    df["heavy_rain"] = (df["precip_mm"] >= HEAVY_MM).astype(int)
    df["is_hot"] = (df["temp_max_c"] >= HOT_C).astype(int)
    df["dow"] = df.index.dayofweek
    df["regime"] = np.select([df["is_rainy"] == 1, df["is_hot"] == 1], ["rainy", "hot"], "normal")
    return df


def data_quality(df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "days": len(df),
        "start": df.index.min().date(),
        "end": df.index.max().date(),
        "real_days": int((df["is_synthetic"] == 0).sum()),
        "synthetic_days": int((df["is_synthetic"] == 1).sum()),
        "missing_order_days": int(df["order_count"].isna().sum()),
        "missing_weather_cells": int(df[["precip_mm", "temp_max_c"]].isna().sum().sum()),
        "min_orders": df["order_count"].min(),
        "max_orders": df["order_count"].max(),
    })


def stationarity_tests(y: pd.Series) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary). Agreement of both is the strong result."""
    adf_stat, adf_p, adf_lags, *_ = adfuller(y.dropna(), autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp_stat, kp_p, kp_lags, _ = kpss(y.dropna(), regression="c", nlags="auto")
    rows = [
        ["ADF", adf_stat, adf_p, "unit root rejected -> stationary" if adf_p < 0.05 else "cannot reject unit root"],
        ["KPSS (level)", kp_stat, kp_p, "stationary not rejected" if kp_p >= 0.05 else "stationarity rejected"],
    ]
    return pd.DataFrame(rows, columns=["test", "statistic", "p_value", "reading"])


def decompose(y: pd.Series, period: int = 7):
    """Additive weekly decomposition (trend + weekly seasonality + remainder)."""
    return seasonal_decompose(y, model="additive", period=period, extrapolate_trend="freq")


def stl_decompose(y: pd.Series, period: int = 7):
    return STL(y, period=period, robust=True).fit()


def weekly_strength(dec) -> float:
    """Seasonal strength F_s = max(0, 1 - Var(remainder) / Var(seasonal + remainder))."""
    r, s = dec.resid.dropna(), dec.seasonal
    return float(max(0.0, 1 - r.var() / (r + s.loc[r.index]).var()))


def time_split(df: pd.DataFrame, test_days: int = 56) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split; the test set is the final `test_days` days. No shuffling."""
    return df.iloc[:-test_days].copy(), df.iloc[-test_days:].copy()


def exog_matrix(df: pd.DataFrame, weather: bool = True, dow: bool = True) -> pd.DataFrame:
    """Regressors for SARIMAX: festival flag (+ day-of-week dummies) (+ weather flags)."""
    X = df[CALENDAR_COLS].astype(float)
    if dow:
        D = pd.get_dummies(df["dow"].astype(int), prefix="dow", drop_first=True).astype(float)
        D = D.reindex(columns=[f"dow_{i}" for i in range(1, 7)], fill_value=0.0)
        X = pd.concat([D, X], axis=1)
    if weather:
        X = pd.concat([X, df[WEATHER_COLS].astype(float)], axis=1)
    return X


# =============================================================================
# T05  models. Every model: fn(train_df, test_df) -> DataFrame[yhat, lo80, hi80]
# =============================================================================
def _frame(index, yhat, lo=None, hi=None) -> pd.DataFrame:
    return pd.DataFrame({"yhat": np.asarray(yhat, float),
                         "lo80": np.nan if lo is None else np.asarray(lo, float),
                         "hi80": np.nan if hi is None else np.asarray(hi, float)}, index=index)


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Value of the same weekday one week earlier (last observed week repeated)."""
    last_week = train["order_count"].iloc[-7:].to_numpy()
    return _frame(test.index, [last_week[i % 7] for i in range(len(test))])


def weekday_mean(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Average of the same weekday over the whole training window (stronger baseline)."""
    m = train.groupby("dow")["order_count"].mean()
    return _frame(test.index, test["dow"].map(m).to_numpy())


def rolling_mean7(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Mean of the last 7 training days, held flat over the horizon (level-only baseline)."""
    return _frame(test.index, np.full(len(test), train["order_count"].iloc[-7:].mean()))


@dataclass
class SarimaxSpec:
    order: tuple = (0, 0, 0)
    seasonal_order: tuple = (0, 0, 0, 0)
    log: bool = False
    weather: bool = True
    dow: bool = True      # day-of-week dummies as regressors
    label: str = ""


def _fit_sarimax(y, X, spec: SarimaxSpec):
    mod = SARIMAX(y, exog=X, order=spec.order, seasonal_order=spec.seasonal_order, trend="c",
                  enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.fit(disp=False, maxiter=200)


def make_sarimax(spec: SarimaxSpec) -> Callable:
    def fn(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        y = np.log(train["order_count"]) if spec.log else train["order_count"]
        res = _fit_sarimax(y, exog_matrix(train, spec.weather, spec.dow), spec)
        fc = res.get_forecast(steps=len(test), exog=exog_matrix(test, spec.weather, spec.dow))
        mu, ci = fc.predicted_mean.to_numpy(), fc.conf_int(alpha=0.20).to_numpy()
        if spec.log:  # back-transform with the usual variance correction for the mean
            s2 = float(res.params["sigma2"]) if "sigma2" in res.params.index else float(np.var(res.resid))
            return _frame(test.index, np.exp(mu + s2 / 2), np.exp(ci[:, 0]), np.exp(ci[:, 1]))
        return _frame(test.index, mu, ci[:, 0], ci[:, 1])
    fn.__name__ = spec.label or "sarimax"
    return fn


def select_sarimax_order(train: pd.DataFrame, weather: bool = True, log: bool = False) -> tuple[SarimaxSpec, pd.DataFrame]:
    """Small AIC grid on the initial window only: ARMA order, seasonal AR/MA at lag 7, and whether
    day-of-week dummies are kept. Returns the best spec and the full table."""
    y = np.log(train["order_count"]) if log else train["order_count"]
    rows = []
    for use_dow, ((p, q), (P, Q)) in itertools.product(
            [True, False], itertools.product([(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)], [(0, 0), (1, 0)])):
        spec = SarimaxSpec(order=(p, 0, q), seasonal_order=(P, 0, Q, 7 if (P or Q) else 0), log=log,
                           weather=weather, dow=use_dow)
        try:
            res = _fit_sarimax(y, exog_matrix(train, weather, use_dow), spec)
            rows.append({"order": (p, 0, q), "seasonal": spec.seasonal_order, "dow_dummies": use_dow,
                         "aic": res.aic, "bic": res.bic})
        except Exception:  # pragma: no cover - a failed candidate is simply skipped
            rows.append({"order": (p, 0, q), "seasonal": spec.seasonal_order, "dow_dummies": use_dow,
                         "aic": np.inf, "bic": np.inf})
    tab = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    best = tab.iloc[0]
    return SarimaxSpec(order=best["order"], seasonal_order=best["seasonal"], log=log, weather=weather,
                       dow=bool(best["dow_dummies"])), tab


def weather_effect(df: pd.DataFrame, log: bool = True, order_window: int | None = None) -> pd.DataFrame:
    """Estimated effect of the weather flags, from a SARIMAX fit on all of `df` (inference, not forecasting).

    With `log=True` the coefficients are proportional effects; `pct_effect` = exp(coef) - 1.
    The ARMA order is chosen by AIC on the first `order_window` rows (all rows if None).
    """
    spec, _ = select_sarimax_order(df.iloc[:order_window] if order_window else df, weather=True, log=log)
    y = np.log(df["order_count"]) if log else df["order_count"]
    res = _fit_sarimax(y, exog_matrix(df, True, spec.dow), spec)
    ci = res.conf_int()
    rows = []
    for term in WEATHER_COLS + CALENDAR_COLS:
        c = float(res.params[term])
        row = {"term": term, "coef": c, "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
               "p_value": float(res.pvalues[term]), "n_days_flag": int(df[term].sum())}
        if log:
            row.update(pct_effect=(np.exp(c) - 1) * 100, pct_low=(np.exp(row["ci_low"]) - 1) * 100,
                       pct_high=(np.exp(row["ci_high"]) - 1) * 100)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("term")
    out.attrs["order"] = (spec.order, spec.seasonal_order, spec.dow)
    return out


def make_prophet(weather: bool = True, changepoint_prior_scale: float = 0.01) -> Callable:
    """Prophet with weekly seasonality, festival flag, optional weather regressors.
    Yearly seasonality is off: the real series covers under one year, so a yearly term cannot be estimated."""
    def fn(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        from prophet import Prophet
        for name in ("cmdstanpy", "prophet"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.CRITICAL)
            lg.handlers.clear()
            lg.propagate = False
        m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=changepoint_prior_scale, interval_width=0.80)
        cols = CALENDAR_COLS + (WEATHER_COLS if weather else [])
        for c in cols:
            m.add_regressor(c)
        tr = train.reset_index().rename(columns={"date": "ds", "order_count": "y"})[["ds", "y"] + cols]
        te = test.reset_index().rename(columns={"date": "ds"})[["ds"] + cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(tr)
            p = m.predict(te)
        return _frame(test.index, p["yhat"], p["yhat_lower"], p["yhat_upper"])
    fn.__name__ = "prophet"
    return fn


def select_prophet_cps(init: pd.DataFrame, weather: bool = True, grid=(0.01, 0.05, 0.2, 0.5),
                       val_days: int = 28) -> tuple[float, dict]:
    """Pick Prophet's trend flexibility on the initial window only: fit on all but the last
    `val_days` of it, score the last `val_days`. Returns (best value, {value: MAE})."""
    tr, va = init.iloc[:-val_days], init.iloc[-val_days:]
    scores = {}
    for c in grid:
        f = make_prophet(weather, c)(tr, va)
        scores[c] = float((va["order_count"] - f["yhat"]).abs().mean())
    return min(scores, key=scores.get), scores


# =============================================================================
# rolling-origin backtest
# =============================================================================
def rolling_origin(df: pd.DataFrame, models: dict[str, Callable], initial: int = 180,
                   horizon: int = 7, step: int = 7, verbose: bool = True) -> pd.DataFrame:
    """Expanding-window backtest. Returns long table: date, fold, h, model, y, yhat, lo80, hi80.

    `initial` is the number of rows before the first forecast day. The last fold may be shorter
    than `horizon`, so every row after `initial` is forecast exactly once when step == horizon.
    """
    out = []
    origins = list(range(initial, len(df), step))
    for k, o in enumerate(origins, 1):
        train, test = df.iloc[:o], df.iloc[o:o + horizon]
        for name, fn in models.items():
            f = fn(train, test)
            f = f.assign(model=name, fold=k, h=np.arange(1, len(test) + 1), y=test["order_count"].to_numpy())
            out.append(f.rename_axis("date").reset_index())
        if verbose and (k % 5 == 0 or k == len(origins)):
            print(f"  fold {k}/{len(origins)} (origin {df.index[o].date()})")
    return pd.concat(out, ignore_index=True)


def static_holdout(df: pd.DataFrame, models: dict[str, Callable], test_days: int = 56) -> pd.DataFrame:
    """Fit once on the training part, forecast the whole test block (multi-week horizon)."""
    train, test = time_split(df, test_days)
    out = []
    for name, fn in models.items():
        f = fn(train, test)
        out.append(f.assign(model=name, h=np.arange(1, len(test) + 1), y=test["order_count"].to_numpy())
                   .rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)


# =============================================================================
# T06  evaluation
# =============================================================================
def metrics(fc: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Accuracy table, one row per model (and per segment value when `by` is given)."""
    keys = ["model"] + (by or [])
    rows = []
    for k, g in fc.groupby(keys, sort=False):
        e = g["y"] - g["yhat"]
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        row.update({"n": len(g), "MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())),
                    "MAPE_%": float((e.abs() / g["y"]).mean() * 100), "bias": e.mean()})
        if g["lo80"].notna().all():
            row["cover80_%"] = float(((g["y"] >= g["lo80"]) & (g["y"] <= g["hi80"])).mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows).set_index(keys)


def add_context(fc: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    ctx = df[["regime", "is_rainy", "dow"]].copy()
    ctx["day_type"] = np.where(ctx["dow"].isin([4, 5, 6]), "Fri-Sun", "Mon-Thu")
    ctx["rain"] = np.where(ctx["is_rainy"] == 1, "rainy", "dry")
    return fc.merge(ctx.reset_index(), on="date", how="left")


def paired_bootstrap_mae_diff(fc: pd.DataFrame, a: str, b: str, n_boot: int = 5000, seed: int = 0,
                              block: int = 7) -> dict:
    """MAE(a) - MAE(b) with a 95% moving-block bootstrap interval. Negative => a is better.

    Forecast errors of consecutive days are correlated, so days are resampled in blocks of `block`
    consecutive days (block=1 gives the ordinary bootstrap, which is too narrow for this data).
    """
    w = fc.pivot(index="date", columns="model", values=["y", "yhat"]).sort_index()
    ea = (w["y"][a] - w["yhat"][a]).abs().to_numpy()
    eb = (w["y"][b] - w["yhat"][b]).abs().to_numpy()
    n = len(ea)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    d = ea[idx].mean(axis=1) - eb[idx].mean(axis=1)
    return {"a": a, "b": b, "mae_diff": float(ea.mean() - eb.mean()),
            "ci95": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
            "share_boot_a_better": float((d < 0).mean())}


def residual_diagnostics(fc: pd.DataFrame, model: str, lags=(7, 14)) -> dict:
    r = fc.loc[fc["model"] == model].sort_values("date")
    e = (r["y"] - r["yhat"]).to_numpy()
    lb = acorr_ljungbox(e, lags=list(lags), return_df=True)
    return {"ljung_box_p": {int(k): float(v) for k, v in lb["lb_pvalue"].items()},
            "resid_mean": float(e.mean()), "resid_std": float(e.std(ddof=1)),
            "resid": pd.Series(e, index=pd.to_datetime(r["date"]))}


def planning_buffer(fc: pd.DataFrame, model: str, q: float = 0.90) -> dict:
    """How far above the forecast to plan so that `q` of days are covered."""
    r = fc.loc[fc["model"] == model]
    over = (r["y"] - r["yhat"])
    return {"q": q, "buffer_orders": float(np.quantile(over, q)),
            "buffer_pct_of_forecast": float(np.quantile(over / r["yhat"], q) * 100)}
