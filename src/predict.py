"""Daily order-volume prediction models (P5 / T25-T26).

Used by notebooks/05_model.ipynb.

Design choices (documented in the notebook):
  * Target is the daily order count, predicted one day ahead from information available the
    evening before: lagged orders plus calendar features. Weather regressors use the *observed*
    weather of the predicted day (an "oracle"), so accuracy is an upper bound on a live system
    that would use a weather forecast. This matches P1's convention in src/forecast.py.
  * Split is chronological (final `TEST_DAYS` days held out). No shuffling anywhere.
  * Hyper-parameters are chosen only on the training window with TimeSeriesSplit.
  * Baseline "Seasonal naive" is the order count of the same weekday one week earlier.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from forecast import load_daily

SEED = 42
TEST_DAYS = 56
CV_SPLITS = 5
MODEL_NAMES = ("Seasonal naive", "Ridge", "Random Forest", "XGBoost")
SW_MONSOON = range(6, 10)   # Jun-Sep
NE_MONSOON = range(10, 13)  # Oct-Dec

WEATHER_FEATURES = ["precip_mm", "temp_max_c", "humidity_mean_pct", "is_rainy", "heavy_rain", "is_hot"]


def load_model_frame(weather_csv: str | Path, orders_csv: str | Path) -> pd.DataFrame:
    """Daily frame (weather + order_count) plus the generator's festival flag when available."""
    df = load_daily(weather_csv, orders_csv)
    orders = pd.read_csv(orders_csv, parse_dates=["date"]).set_index("date")
    df["is_festival"] = orders["is_festival"].astype(int).reindex(df.index).fillna(0) if "is_festival" in orders else 0
    return df.dropna(subset=["order_count"])


def monsoon_phase(month: int) -> str:
    if month in SW_MONSOON:
        return "sw_monsoon"
    return "ne_monsoon" if month in NE_MONSOON else "dry_season"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix, one row per predictable day. Uses only past orders (shifted) and same-day weather/calendar."""
    y = df["order_count"]
    X = pd.DataFrame(index=df.index)
    X["lag1"] = y.shift(1)
    X["lag7"] = y.shift(7)
    X["roll7"] = y.shift(1).rolling(7).mean()
    for col in WEATHER_FEATURES:
        X[col] = df[col].astype(float)
    X["is_fri_sun"] = df.index.dayofweek.isin([4, 5, 6]).astype(int)
    X["is_festival"] = df["is_festival"].astype(float)
    dow = pd.get_dummies(df.index.dayofweek, prefix="dow").astype(float)
    dow.index = df.index
    X = pd.concat([X, dow.reindex(columns=[f"dow_{i}" for i in range(7)], fill_value=0.0)], axis=1)
    phase = pd.Series([monsoon_phase(m) for m in df.index.month], index=df.index)
    for p in ("sw_monsoon", "ne_monsoon"):
        X[f"phase_{p}"] = (phase == p).astype(float)
    return X.dropna()


def split_train_test(X: pd.DataFrame, y: pd.Series, test_days: int = TEST_DAYS):
    """Chronological split: the last `test_days` rows are the test set."""
    return X.iloc[:-test_days], X.iloc[-test_days:], y.iloc[:-test_days], y.iloc[-test_days:]


def seasonal_naive_pred(X: pd.DataFrame) -> pd.Series:
    return X["lag7"].copy()


def _tuned(estimator, grid: dict, Xtr: pd.DataFrame, ytr: pd.Series):
    search = GridSearchCV(estimator, grid, cv=TimeSeriesSplit(CV_SPLITS),
                          scoring="neg_mean_absolute_error", n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(Xtr, ytr)
    return search.best_estimator_, search.best_params_


def fit_and_predict(Xtr, ytr, Xte, quick: bool = False):
    """Fit Ridge, Random Forest and XGBoost; return ({name: fitted}, {name: predictions on Xte})."""
    ridge_grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}
    rf_grid = {"max_depth": [4, 8], "min_samples_leaf": [3, 10]} if not quick else {"max_depth": [4]}
    xgb_grid = ({"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "n_estimators": [200, 400]}
                if not quick else {"max_depth": [2], "n_estimators": [100]})
    fitted, params = {}, {}
    fitted["Ridge"], params["Ridge"] = _tuned(make_pipeline(StandardScaler(), Ridge()), ridge_grid, Xtr, ytr)
    fitted["Random Forest"], params["Random Forest"] = _tuned(
        RandomForestRegressor(n_estimators=300 if not quick else 50, random_state=SEED, n_jobs=1), rf_grid, Xtr, ytr)
    fitted["XGBoost"], params["XGBoost"] = _tuned(
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=1, verbosity=0), xgb_grid, Xtr, ytr)
    preds = {"Seasonal naive": seasonal_naive_pred(Xte)}
    preds.update({n: pd.Series(m.predict(Xte), index=Xte.index) for n, m in fitted.items()})
    fit_and_predict.best_params = params
    return fitted, preds


def score(y: pd.Series, yhat: pd.Series) -> dict:
    e = y - yhat
    mse = float((e ** 2).mean())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"R2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot else float("nan"),
            "MAE": float(e.abs().mean()), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAPE_%": float((e.abs() / y).mean() * 100), "bias": float(e.mean())}


def score_table(y: pd.Series, preds: dict, by: pd.Series | None = None) -> pd.DataFrame:
    """One row per model (and per segment value when `by` is given)."""
    rows = []
    for name in MODEL_NAMES:
        p = preds[name]
        groups = [("all", slice(None))] if by is None else [(k, by == k) for k in by.unique()]
        for label, mask in groups:
            m = score(y[mask], p[mask]) if by is not None else score(y, p)
            rows.append({"model": name, **({} if by is None else {"segment": label}), "n": int(
                (mask.sum() if by is not None else len(y))), **m})
    keys = ["model"] + ([] if by is None else ["segment"])
    return pd.DataFrame(rows).set_index(keys)
