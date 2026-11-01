"""Model selection and validation for daily order volume (P5).

Used by notebooks/06_model_selection.ipynb. Answers "why these models, and how do we know the winner is best?"

Design choices (documented in the notebook):
  * Only real POS days are ever scored. Synthetic days are independent draws with no day-to-day memory, so a score
    on them says nothing about real demand.
  * Rolling-origin backtest: every forecast uses only earlier days. Models are refitted and re-tuned every 7 days.
  * Version A trains on real days only and decides the winner. Version B also trains on a synthetic prefix whose
    generator is calibrated only on days before the first forecast, so no test-period information can reach it.
  * The decision rule is fixed in advance: see `decide`.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

import forecast
import predict

SEED = predict.SEED
INITIAL_TRAIN_DAYS = 120
REFIT_EVERY = 7
WEEK = 7
BOOTSTRAP_DRAWS = 5000
ALPHA = 0.05
INCUMBENT = "Ridge"
MIN_DAYS_FOR_MULTIPLIER = 5  # a condition seen on fewer days cannot support an estimate
MIN_NB_ALPHA = 1e-4
OVERDISPERSED_ABOVE = 1.5
WEEKLY_ACF_ABOVE = 0.2
SMALL_DATA_DAYS_PER_FEATURE = 10
COLLINEAR_COLUMNS = ("is_fri_sun", "dow_0")  # dummies + intercept would otherwise be rank deficient
SARIMAX_EXOG = ("is_rainy", "heavy_rain", "is_hot", "is_festival")
SARIMAX_ORDERS = ((1, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
SARIMAX_SEASONAL = ((0, 0, 0, WEEK), (1, 0, 0, WEEK), (0, 1, 1, WEEK))

Predictor = Callable[[pd.DataFrame, "pd.Series | None"], pd.Series]
ModelFactory = Callable[[pd.DataFrame, pd.Series], Predictor]


# =============================================================================
# Data
# =============================================================================
def load_real_frame(weather_csv: str | Path, orders_csv: str | Path) -> pd.DataFrame:
    """Daily model frame restricted to real POS days (is_synthetic == 0)."""
    flags = pd.read_csv(orders_csv, parse_dates=["date"]).set_index("date")
    if "is_synthetic" not in flags:
        raise ValueError(f"{orders_csv} has no is_synthetic column, so real days cannot be identified")
    df = predict.load_model_frame(weather_csv, orders_csv)
    return df[flags["is_synthetic"].reindex(df.index) == 0]


def first_test_day(X: pd.DataFrame, initial: int = INITIAL_TRAIN_DAYS) -> pd.Timestamp:
    """The first forecast day, leaving `initial` rows of X for the first training window."""
    if not 0 < initial < len(X):
        raise ValueError(f"initial={initial} must leave both training and test rows out of {len(X)}")
    return X.index[initial]


def no_weather_columns(X: pd.DataFrame) -> list[str]:
    """Feature columns left after removing same-day weather and monsoon phase."""
    return [c for c in X.columns if c not in predict.WEATHER_FEATURES and not c.startswith("phase_")]


# =============================================================================
# Data profile: which traits point to which model family
# =============================================================================
def _design_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Full-rank numeric features: no collinear dummies and no constant columns."""
    kept = X.drop(columns=[c for c in COLLINEAR_COLUMNS if c in X])
    return kept.loc[:, kept.std() > 0]


def profile_data(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """One row per data trait: measured value, what it means, and the model family it points to."""
    tests = forecast.stationarity_tests(y).set_index("test")["p_value"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sm.OLS(y, sm.add_constant(_design_matrix(X))).fit()
        reset_p = float(linear_reset(ols, power=2, use_f=True).pvalue)
    per_feature = len(y) / X.shape[1]
    dispersion = float(y.var() / y.mean())
    acf1, acf7 = float(y.autocorr(1)), float(y.autocorr(WEEK))
    rows = [
        ("Training days", len(y), "Sample available before the first forecast", "Sets how complex a model can be"),
        ("Days per feature", per_feature,
         "Small sample, overfitting risk" if per_feature < SMALL_DATA_DAYS_PER_FEATURE else "Enough days per feature",
         "Regularised linear: Ridge, ElasticNet" if per_feature < SMALL_DATA_DAYS_PER_FEATURE else "Any family"),
        ("Variance / mean", dispersion,
         "Counts spread wider than Poisson" if dispersion > OVERDISPERSED_ABOVE else "Close to Poisson",
         "Negative Binomial" if dispersion > OVERDISPERSED_ABOVE else "Poisson"),
        ("Autocorrelation at 1 day", acf1, "Yesterday carries information about today", "Lag features, SARIMAX"),
        ("Autocorrelation at 7 days", acf7,
         "Weekly memory" if acf7 > WEEKLY_ACF_ABOVE else "Little weekly memory",
         "SARIMAX, seasonal baselines" if acf7 > WEEKLY_ACF_ABOVE else "Weekly terms are optional"),
        ("ADF p-value", float(tests["ADF"]), "Below 0.05 rejects a unit root", "Decides differencing in SARIMAX"),
        ("KPSS p-value", float(tests["KPSS (level)"]), "Below 0.05 rejects stationarity",
         "Decides differencing in SARIMAX"),
        ("RESET p-value", reset_p,
         "Linear form is missing something" if reset_p < ALPHA else "No evidence against a linear form",
         "Tree models: RF, XGBoost, Extra Trees" if reset_p < ALPHA else "Linear models; trees as a check"),
    ]
    return pd.DataFrame(rows, columns=["trait", "value", "reading", "points_to"])


# =============================================================================
# Models: each factory fits on past data and returns a one-step predictor
# =============================================================================
def _as_predictor(model) -> Predictor:
    return lambda Xb, _: pd.Series(model.predict(Xb), index=Xb.index)


def _tuned_factory(estimator, grid: dict) -> ModelFactory:
    """Grid search with TimeSeriesSplit on the training window only (same procedure as predict.py)."""
    return lambda Xtr, ytr: _as_predictor(predict._tuned(estimator, grid, Xtr, ytr)[0])


def _column_factory(column: str) -> ModelFactory:
    return lambda Xtr, ytr: lambda Xb, _: Xb[column].copy()


def _weekday_mean(Xtr: pd.DataFrame, ytr: pd.Series) -> Predictor:
    cols = [f"dow_{i}" for i in range(WEEK)]
    means = ytr.groupby(Xtr[cols].to_numpy().argmax(axis=1)).mean().reindex(range(WEEK)).fillna(ytr.mean())

    def predictor(Xb: pd.DataFrame, _) -> pd.Series:
        return pd.Series(means.to_numpy()[Xb[cols].to_numpy().argmax(axis=1)], index=Xb.index)

    return predictor


def _negative_binomial_alpha(y: pd.Series, mu: np.ndarray) -> float:
    """Method-of-moments dispersion from a Poisson fit: Var = mu + alpha * mu^2."""
    return max(float((((y - mu) ** 2 - mu) / mu ** 2).mean()), MIN_NB_ALPHA)


def _fit_negative_binomial(design: pd.DataFrame, y: pd.Series):
    exog = sm.add_constant(design, has_constant="add")
    poisson = sm.GLM(y, exog, family=sm.families.Poisson()).fit()
    alpha = _negative_binomial_alpha(y, poisson.mu)
    return sm.GLM(y, exog, family=sm.families.NegativeBinomial(alpha=alpha)).fit(), alpha


def _negative_binomial(Xtr: pd.DataFrame, ytr: pd.Series) -> Predictor:
    design = _design_matrix(Xtr)
    mean, std = design.mean(), design.std()
    fit, _ = _fit_negative_binomial((design - mean) / std, ytr)

    def predictor(Xb: pd.DataFrame, _) -> pd.Series:
        scaled = (Xb[design.columns] - mean) / std
        return pd.Series(fit.predict(sm.add_constant(scaled, has_constant="add")), index=Xb.index)

    return predictor


def _fit_sarimax(y: np.ndarray, exog: np.ndarray, order: tuple, seasonal: tuple):
    trend = "c" if order[1] == 0 and seasonal[1] == 0 else "n"
    model = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False, maxiter=200)


def _sarimax_factory() -> ModelFactory:
    """SARIMAX with weather regressors. The order is chosen by AIC at the first fit, then kept."""
    chosen: dict[str, tuple] = {}

    def factory(Xtr: pd.DataFrame, ytr: pd.Series) -> Predictor:
        cols = [c for c in SARIMAX_EXOG if c in Xtr]
        y, exog = ytr.to_numpy(float), Xtr[cols].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not chosen:
                candidates = list(itertools.product(SARIMAX_ORDERS, SARIMAX_SEASONAL))
                chosen["spec"] = min(candidates, key=lambda spec: _fit_sarimax(y, exog, *spec).aic)
            fit = _fit_sarimax(y, exog, *chosen["spec"])

        def predictor(Xb: pd.DataFrame, yb: pd.Series) -> pd.Series:
            # extend() keeps the fitted parameters; each prediction uses actuals only up to the day before.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ext = fit.extend(yb.to_numpy(float), exog=Xb[cols].to_numpy(float))
            return pd.Series(ext.predict(start=0, end=len(Xb) - 1), index=Xb.index)

        return predictor

    return factory


def make_models(quick: bool = False) -> dict[str, ModelFactory]:
    """All candidates. Existing models keep the grids from predict.py so the comparison is like for like."""
    trees = 50 if quick else 300
    forest_grid = {"max_depth": [4]} if quick else {"max_depth": [4, 8], "min_samples_leaf": [3, 10]}
    xgb_grid = ({"max_depth": [2], "n_estimators": [100]} if quick
                else {"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "n_estimators": [200, 400]})
    linear_alphas = [1.0] if quick else [0.01, 0.1, 1.0, 10.0]
    return {
        "Seasonal naive": _column_factory("lag7"),
        "Rolling mean": _column_factory("roll7"),
        "Weekday mean": _weekday_mean,
        "Ridge": _tuned_factory(make_pipeline(StandardScaler(), Ridge()),
                                {"ridge__alpha": [1.0] if quick else [0.1, 1.0, 10.0, 100.0]}),
        "ElasticNet": _tuned_factory(make_pipeline(StandardScaler(), ElasticNet(max_iter=10_000)),
                                     {"elasticnet__alpha": linear_alphas, "elasticnet__l1_ratio": [0.2, 0.5, 0.8]}),
        "Poisson": _tuned_factory(make_pipeline(StandardScaler(), PoissonRegressor(max_iter=1000)),
                                  {"poissonregressor__alpha": linear_alphas}),
        "Negative Binomial": _negative_binomial,
        "SARIMAX": _sarimax_factory(),
        "Random Forest": _tuned_factory(RandomForestRegressor(n_estimators=trees, random_state=SEED, n_jobs=1),
                                        forest_grid),
        "Extra Trees": _tuned_factory(ExtraTreesRegressor(n_estimators=trees, random_state=SEED, n_jobs=1),
                                      forest_grid),
        "XGBoost": _tuned_factory(XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=1,
                                               verbosity=0), xgb_grid),
    }


# =============================================================================
# Rolling-origin backtest
# =============================================================================
def rolling_backtest(X: pd.DataFrame, y: pd.Series, models: dict[str, ModelFactory], first_test: pd.Timestamp,
                     refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    """One-step forecasts for every day from `first_test` on. Rows: test days. Columns: models.

    At each refit a model sees only rows before the block. Inside a block, lag features and the state update
    of SARIMAX use actuals only up to the evening before the forecast day.
    """
    start = int(X.index.searchsorted(first_test))
    if not 0 < start < len(X):
        raise ValueError(f"first_test={first_test} leaves no training or no test rows")
    columns: dict[str, list[pd.Series]] = {name: [] for name in models}
    for origin in range(start, len(X), refit_every):
        Xtr, ytr = X.iloc[:origin], y.iloc[:origin]
        Xb, yb = X.iloc[origin:origin + refit_every], y.iloc[origin:origin + refit_every]
        for name, factory in models.items():
            columns[name].append(factory(Xtr, ytr)(Xb, yb))
    return pd.DataFrame({name: pd.concat(parts) for name, parts in columns.items()})


# =============================================================================
# Synthetic prefix for version B
# =============================================================================
def _generator_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Same conditions as src/generator.py."""
    return pd.DataFrame({
        "weekend": df.index.day_of_week.isin([4, 5, 6]).astype(float),
        "rain_mid": ((df["is_rainy"] == 1) & (df["heavy_rain"] == 0)).astype(float),
        "rain_heavy": (df["heavy_rain"] == 1).astype(float),
        "hot": (df["is_hot"] == 1).astype(float),
        "festival": (df["is_festival"] == 1).astype(float),
    }, index=df.index)


def calibrate_generator(real: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Fit the generator's Negative Binomial form on days before `cutoff` only."""
    past = real[real.index < cutoff]
    flags = _generator_flags(past)
    supported = flags.loc[:, flags.sum() >= MIN_DAYS_FOR_MULTIPLIER]
    fit, alpha = _fit_negative_binomial(supported, past["order_count"])
    multipliers = {name: 1.0 for name in flags.columns}
    multipliers.update({name: float(np.exp(fit.params[name])) for name in supported.columns})
    return {"base": float(np.exp(fit.params["const"])), "r": 1.0 / alpha, "multipliers": multipliers}


def simulate_orders(df: pd.DataFrame, params: dict, seed: int = SEED) -> pd.Series:
    """Independent Negative Binomial draws, the same process as generator.generate_daily_demand."""
    flags = _generator_flags(df)
    log_mu = sum(flags[name] * np.log(mult) for name, mult in params["multipliers"].items())
    mu = params["base"] * np.exp(log_mu)
    r = params["r"]
    draws = np.random.default_rng(seed).negative_binomial(r, r / (r + mu.to_numpy()))
    return pd.Series(draws.astype(float), index=df.index, name="order_count")


def build_hybrid_frame(full: pd.DataFrame, real: pd.DataFrame, cutoff: pd.Timestamp, seed: int = SEED) -> pd.DataFrame:
    """Synthetic days before the real period, regenerated from pre-cutoff parameters, followed by the real days.

    Synthetic days after the real period are dropped: they are future data for every test day.
    """
    earlier = full[full.index < real.index.min()]
    orders = simulate_orders(earlier, calibrate_generator(real, cutoff), seed)
    return pd.concat([earlier.assign(order_count=orders), real])


# =============================================================================
# Statistics
# =============================================================================
def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold corrected DM test on absolute errors at horizon 1.

    A positive statistic means model A has the larger errors. Returns (statistic, two-sided p-value).
    """
    d = np.abs(np.asarray(errors_a, float)) - np.abs(np.asarray(errors_b, float))
    n = len(d)
    mean, variance = d.mean(), d.var() / n
    if variance == 0:
        return (0.0, 1.0) if mean == 0 else (float(np.sign(mean) * np.inf), 0.0)
    stat = mean / np.sqrt(variance) * np.sqrt((n - 1) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, for running several tests at once."""
    n = len(p_values)
    adjusted, running = [0.0] * n, 0.0
    for rank, i in enumerate(np.argsort(p_values)):
        running = max(running, min(1.0, (n - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def block_bootstrap_ci(d: np.ndarray, block: int = WEEK, draws: int = BOOTSTRAP_DRAWS,
                       seed: int = SEED) -> tuple[float, float]:
    """95% moving-block bootstrap interval for the mean. Blocks keep the weekly dependence intact."""
    d = np.asarray(d, float)
    n = len(d)
    blocks = int(np.ceil(n / block))
    starts = np.random.default_rng(seed).integers(0, n - block + 1, size=(draws, blocks))
    positions = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]
    means = d[positions].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mase(y: pd.Series, yhat: pd.Series, train_y: pd.Series) -> float:
    """MAE relative to the weekly naive error on the training window. Below 1 beats that naive."""
    return float((y - yhat).abs().mean() / train_y.diff(WEEK).abs().mean())


# =============================================================================
# Comparison and decision
# =============================================================================
def compare_to_incumbent(y: pd.Series, preds: pd.DataFrame, train_y: pd.Series,
                         incumbent: str = INCUMBENT) -> pd.DataFrame:
    """Accuracy per model and the evidence for its difference from the incumbent, best MAE first."""
    errors = preds.rsub(y, axis=0)
    challengers = [m for m in preds.columns if m != incumbent]
    dm = {m: diebold_mariano(errors[m], errors[incumbent]) for m in challengers}
    adjusted = dict(zip(challengers, holm([dm[m][1] for m in challengers])))
    rows = {}
    for model in preds.columns:
        is_incumbent = model == incumbent
        low, high = (np.nan, np.nan) if is_incumbent else block_bootstrap_ci(
            errors[model].abs() - errors[incumbent].abs())
        rows[model] = {
            **predict.score(y, preds[model]),
            "MASE": mase(y, preds[model], train_y),
            "delta_MAE": float((errors[model].abs() - errors[incumbent].abs()).mean()),
            "dm_stat": np.nan if is_incumbent else dm[model][0],
            "dm_p": np.nan if is_incumbent else dm[model][1],
            "dm_p_holm": np.nan if is_incumbent else adjusted[model],
            "ci_low": low,
            "ci_high": high,
        }
    return pd.DataFrame(rows).T.sort_values("MAE")


def decide(table: pd.DataFrame, incumbent: str = INCUMBENT) -> dict:
    """The rule fixed in advance: replace the incumbent only with lower MAE, Holm p < 0.05 and a CI below zero."""
    challengers = table.drop(index=incumbent)
    qualified = challengers[(challengers["MAE"] < table.loc[incumbent, "MAE"])
                            & (challengers["dm_p_holm"] < ALPHA)
                            & (challengers["ci_high"] < 0)]
    if qualified.empty:
        return {"winner": incumbent, "replaced": False,
                "reason": f"No candidate cleared all three conditions, so {incumbent} stays."}
    winner = qualified["MAE"].astype(float).idxmin()
    return {"winner": winner, "replaced": True,
            "reason": f"{winner} has lower MAE, Holm p < {ALPHA} and a bootstrap interval below zero."}


def segment_table(y: pd.Series, preds: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """MAE and day count per model and segment."""
    frame = preds.rsub(y, axis=0).abs().groupby(by).mean().T
    frame.columns = [f"MAE {c}" for c in frame.columns]
    counts = by.value_counts()
    frame.attrs["days"] = {str(k): int(v) for k, v in counts.items()}
    return frame
