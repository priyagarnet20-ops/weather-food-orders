"""Time-series preparation, forecasting models and evaluation (P1 / T04-T06).

Public functions are used by notebooks/07_timeseries.ipynb.

Design choices (all documented in the notebook):
  * Target is the daily order count. Real weather (data/raw/weather.csv) is joined on date.
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

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)


# =============================================================================
# T04  data preparation
# =============================================================================
def load_daily(weather_csv: str | Path, orders_csv: str | Path) -> pd.DataFrame:
    """Daily order series joined with weather; indexed by date, one row per day.

    `orders_csv` may be a daily table (date, order_count) or an order-level table
    (date or timestamp column plus order_id), which is aggregated to daily counts.
    """
    w = pd.read_csv(weather_csv, parse_dates=["date"])
    o = pd.read_csv(orders_csv)
    if "order_count" in o.columns:
        date_col = "date"
        y = o.assign(date=pd.to_datetime(o[date_col]).dt.normalize())[["date", "order_count"]]
    else:
        date_col = next(c for c in ("date", "order_time", "timestamp", "datetime") if c in o.columns)
        d = pd.to_datetime(o[date_col]).dt.normalize()
        y = o.assign(date=d).groupby("date")["order_id"].nunique().rename("order_count").reset_index()
    df = w.merge(y, on="date", how="left").sort_values("date").set_index("date")
    df["order_count"] = df["order_count"].astype(float)
    df = df.asfreq("D")
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


def exog_matrix(df: pd.DataFrame, weather: bool = True) -> pd.DataFrame:
    """Regressors for SARIMAX: day-of-week dummies (+ weather flags)."""
    X = pd.get_dummies(df["dow"].astype(int), prefix="dow", drop_first=True).astype(float)
    X = X.reindex(columns=[f"dow_{i}" for i in range(1, 7)], fill_value=0.0)
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


@dataclass
class SarimaxSpec:
    order: tuple = (0, 0, 0)
    seasonal_order: tuple = (0, 0, 0, 0)
    log: bool = False
    weather: bool = True
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
        res = _fit_sarimax(y, exog_matrix(train, spec.weather), spec)
        fc = res.get_forecast(steps=len(test), exog=exog_matrix(test, spec.weather))
        mu, ci = fc.predicted_mean.to_numpy(), fc.conf_int(alpha=0.20).to_numpy()
        if spec.log:  # back-transform with the usual variance correction for the mean
            s2 = float(res.params["sigma2"]) if "sigma2" in res.params.index else float(np.var(res.resid))
            return _frame(test.index, np.exp(mu + s2 / 2), np.exp(ci[:, 0]), np.exp(ci[:, 1]))
        return _frame(test.index, mu, ci[:, 0], ci[:, 1])
    fn.__name__ = spec.label or "sarimax"
    return fn


def select_sarimax_order(train: pd.DataFrame, weather: bool = True, log: bool = False) -> tuple[SarimaxSpec, pd.DataFrame]:
    """Small AIC grid on the initial window only. Returns the best spec and the full table."""
    y = np.log(train["order_count"]) if log else train["order_count"]
    X = exog_matrix(train, weather)
    rows = []
    for (p, q), (P, Q) in itertools.product([(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)], [(0, 0), (1, 0)]):
        spec = SarimaxSpec(order=(p, 0, q), seasonal_order=(P, 0, Q, 7 if (P or Q) else 0), log=log, weather=weather)
        try:
            res = _fit_sarimax(y, X, spec)
            rows.append({"order": (p, 0, q), "seasonal": spec.seasonal_order, "aic": res.aic, "bic": res.bic})
        except Exception as e:  # pragma: no cover - a failed candidate is simply skipped
            rows.append({"order": (p, 0, q), "seasonal": spec.seasonal_order, "aic": np.inf, "bic": np.inf})
    tab = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    best = tab.iloc[0]
    return SarimaxSpec(order=best["order"], seasonal_order=best["seasonal"], log=log, weather=weather), tab


def make_prophet(weather: bool = True, changepoint_prior_scale: float = 0.01) -> Callable:
    """Prophet with weekly seasonality, no yearly term (only one year of history), optional weather regressors."""
    def fn(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        from prophet import Prophet
        for name in ("cmdstanpy", "prophet"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.CRITICAL)
            lg.handlers.clear()
            lg.propagate = False
        m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=changepoint_prior_scale, interval_width=0.80)
        cols = WEATHER_COLS if weather else []
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


# =============================================================================
# rolling-origin backtest
# =============================================================================
def rolling_origin(df: pd.DataFrame, models: dict[str, Callable], initial: int = 180,
                   horizon: int = 7, step: int = 7, verbose: bool = True) -> pd.DataFrame:
    """Expanding-window backtest. Returns long table: date, fold, h, model, y, yhat, lo80, hi80."""
    out = []
    origins = list(range(initial, len(df) - horizon + 1, step))
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


def paired_bootstrap_mae_diff(fc: pd.DataFrame, a: str, b: str, n_boot: int = 5000, seed: int = 0) -> dict:
    """MAE(a) - MAE(b) with a 95% bootstrap interval, resampling forecast days. Negative => a is better."""
    w = fc.pivot(index="date", columns="model", values=["y", "yhat"])
    ea = (w["y"][a] - w["yhat"][a]).abs().to_numpy()
    eb = (w["y"][b] - w["yhat"][b]).abs().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(n_boot, len(ea)))
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
