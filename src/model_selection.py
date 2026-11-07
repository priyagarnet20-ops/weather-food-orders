"""Model selection and validation for daily order volume (P5).

Used by notebooks/05b_model_selection.ipynb. Answers "why these models, and how do we know the winner is best?"

Design choices (documented in the notebook and docs/p5_model_selection.md):
  * Rolling-origin backtest over the last year (`TEST_DAYS`): every forecast uses only earlier days. Models are
    refitted and re-tuned every 7 days.
  * The generator's own expected value is scored as a reference ("best possible"). It is not a candidate.
  * The decision rule is fixed in advance: see `decide`.
"""
from __future__ import annotations

import itertools
import warnings
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
TEST_DAYS = 365  # last year: both monsoons and both hot seasons
REFIT_EVERY = 7
WEEK = 7
BOOTSTRAP_DRAWS = 5000
ALPHA = 0.05
INCUMBENT = "Ridge"
REFERENCE = "Generator expected value"  # best possible score, never a candidate
MIN_NB_ALPHA = 1e-4
OVERDISPERSED_ABOVE = 1.5
WEEKLY_ACF_ABOVE = 0.2
SMALL_DATA_DAYS_PER_FEATURE = 10
COLLINEAR_COLUMNS = ("is_fri_sun", "dow_0")  # dummies + intercept would otherwise be rank deficient
SARIMAX_EXOG = ("is_rainy", "heavy_rain", "is_hot", "rain_x_weekend", "is_festival")
SARIMAX_ORDERS = ((1, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
SARIMAX_SEASONAL = ((0, 0, 0, WEEK), (1, 0, 0, WEEK), (0, 1, 1, WEEK))

Predictor = Callable[[pd.DataFrame, "pd.Series | None"], pd.Series]
ModelFactory = Callable[[pd.DataFrame, pd.Series], Predictor]


# =============================================================================
# Data
# =============================================================================
def first_test_day(X: pd.DataFrame, initial: int = INITIAL_TRAIN_DAYS) -> pd.Timestamp:
    """The first forecast day, leaving `initial` rows of X for the first training window."""
    if not 0 < initial < len(X):
        raise ValueError(f"initial={initial} must leave both training and test rows out of {len(X)}")
    return X.index[initial]


def no_weather_columns(X: pd.DataFrame) -> list[str]:
    """Feature columns left after removing same-day weather (including rain x weekend) and monsoon phase."""
    return [c for c in X.columns if c not in predict.WEATHER_INPUTS and not c.startswith("phase_")]


def generator_expected(orders_csv, index: pd.DatetimeIndex) -> pd.Series:
    """The generator's expected orders for each day: the best any model could do on average."""
    daily = pd.read_csv(orders_csv, parse_dates=["date"]).set_index("date")
    return daily["expected_orders"].reindex(index).rename(REFERENCE)


# =============================================================================
# Data profile: which traits point to which model family
# =============================================================================
def _design_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Full-rank numeric features: no collinear dummies and no constant columns."""
    kept = X.drop(columns=[c for c in COLLINEAR_COLUMNS if c in X])
    return kept.loc[:, kept.std() > 0]


def _reset_p(X: pd.DataFrame, y: pd.Series) -> float:
    """Ramsey RESET p-value of a linear fit: small means the linear form misses something."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sm.OLS(y, sm.add_constant(_design_matrix(X))).fit()
        return float(linear_reset(ols, power=2, use_f=True).pvalue)


def _interaction_f_p(X: pd.DataFrame, y: pd.Series, column: str) -> float:
    """F-test p-value for adding `column` to a linear model that has every other feature."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = sm.OLS(y, sm.add_constant(_design_matrix(X))).fit()
        reduced = sm.OLS(y, sm.add_constant(_design_matrix(X.drop(columns=[column])))).fit()
        return float(full.compare_f_test(reduced)[1])


def profile_data(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """One row per data trait: measured value, what it means, and the model family it points to."""
    tests = forecast.stationarity_tests(y).set_index("test")["p_value"]
    reset_p = _reset_p(X, y)
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
    if "rain_x_weekend" in X:
        f_p = _interaction_f_p(X, y, "rain_x_weekend")
        rows.append(("Rain x weekend F-test p-value", f_p,
                     "Rain works differently on weekdays and weekends" if f_p < ALPHA else "No evidence of an interaction",
                     "Linear models need the interaction term; trees can find it alone" if f_p < ALPHA
                     else "Interaction optional"))
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
    challengers = [m for m in preds.columns if m not in (incumbent, REFERENCE)]
    dm = {m: diebold_mariano(errors[m], errors[incumbent]) for m in challengers}
    adjusted = dict(zip(challengers, holm([dm[m][1] for m in challengers])))
    rows = {}
    for model in preds.columns:
        is_incumbent = model == incumbent
        if model == REFERENCE:
            rows[model] = {**predict.score(y, preds[model]), "MASE": mase(y, preds[model], train_y),
                           "delta_MAE": float((errors[model].abs() - errors[incumbent].abs()).mean()),
                           "dm_stat": np.nan, "dm_p": np.nan, "dm_p_holm": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            continue
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
    challengers = table.drop(index=[incumbent, REFERENCE], errors="ignore")
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
