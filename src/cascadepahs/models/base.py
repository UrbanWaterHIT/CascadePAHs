"""Algorithm factory, hyper-parameter grids and date-grouped tuning.

The six regression algorithms of the study (Table 1, Table S28):
RF, SVR, L2-regularised linear regression (LR), XGBoost, MLP and DNN.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ALGORITHMS = ["RF", "SVR", "XGBoost", "LR", "DNN", "MLP"]

try:  # XGBoost is optional at import time; requirements.txt installs it.
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - depends on the environment
    HAS_XGBOOST = False

_WARNED_XGB = False


def _xgb(random_state: int, n_jobs: int):
    global _WARNED_XGB
    if HAS_XGBOOST:
        return XGBRegressor(random_state=random_state, n_jobs=n_jobs, tree_method="hist", verbosity=0)
    if not _WARNED_XGB:
        warnings.warn(
            "xgboost is not installed; falling back to sklearn GradientBoostingRegressor "
            "(same hyper-parameters: n_estimators, learning_rate, max_depth, subsample). "
            "Install xgboost to reproduce the published analysis.",
            RuntimeWarning, stacklevel=2,
        )
        _WARNED_XGB = True
    return GradientBoostingRegressor(random_state=random_state)


def make_estimator(name: str, *, scale_X: bool = True, random_state: int = 0, n_jobs: int = 1,
                   quantile: bool = False):
    """Return an unfitted estimator. The core estimator is always the Pipeline step ``est``.

    * ``scale_X`` – add a StandardScaler for X (sensor-only models). In SOLACE the
      sensor block is standardised internally and the one-hot leaf block is binary,
      so the final stage uses ``scale_X=False``.
    * SVR, MLP and DNN are wrapped in a TransformedTargetRegressor so that the
      target is standardised during fitting (the SVR epsilon grid of Table S28
      is then expressed in standard-deviation units).
    * ``quantile=True`` (RF only) returns a quantile regression forest.
    """
    from .qrf import QuantileRandomForestRegressor

    if name == "RF":
        cls = QuantileRandomForestRegressor if quantile else RandomForestRegressor
        est = cls(random_state=random_state, n_jobs=n_jobs)
    elif name == "SVR":
        est = SVR()
    elif name == "XGBoost":
        est = _xgb(random_state, n_jobs)
    elif name == "LR":
        est = Ridge(random_state=random_state)
    elif name == "DNN":
        est = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=random_state)
    elif name == "MLP":
        est = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"unknown algorithm {name!r}; choose from {ALGORITHMS}")

    steps = [("scale", StandardScaler(with_mean=True))] if (scale_X and name in {"SVR", "LR", "DNN", "MLP"}) else []
    pipe = Pipeline(steps + [("est", est)])
    if name in {"SVR", "DNN", "MLP"}:
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    return pipe


def _prefix(estimator) -> str:
    return "regressor__est__" if isinstance(estimator, TransformedTargetRegressor) else "est__"


def prefixed_grid(estimator, grid: dict[str, list]) -> dict[str, list]:
    """Translate an algorithm grid (Table S28 names) to Pipeline parameter names."""
    p = _prefix(estimator)
    return {p + k: list(v) for k, v in grid.items()}


def _core(estimator):
    if isinstance(estimator, TransformedTargetRegressor):
        return estimator.regressor.named_steps["est"]
    return estimator.named_steps["est"]


def set_core_params(estimator, params: dict[str, Any]):
    """Set parameters on the core estimator using unprefixed names."""
    p = _prefix(estimator)
    estimator.set_params(**{p + k: v for k, v in params.items()})
    return estimator


def core_params(search_best_params: dict[str, Any]) -> dict[str, Any]:
    """Strip Pipeline prefixes from GridSearchCV.best_params_."""
    return {k.split("__")[-1]: v for k, v in search_best_params.items()}


def inner_cv(groups: np.ndarray, n_inner: int) -> GroupKFold:
    n_groups = len(np.unique(groups))
    return GroupKFold(n_splits=max(2, min(n_inner, n_groups)))


def tune(estimator, grid: dict[str, list], X, y, groups, *, n_inner: int = 5, scoring="neg_mean_absolute_error",
         n_jobs: int = 1):
    """Grid search with folds grouped by survey date. Returns (fitted best estimator, best params).

    If the grid is empty or has a single combination the estimator is simply fitted.
    """
    pgrid = prefixed_grid(estimator, grid) if grid else {}
    n_comb = int(np.prod([len(v) for v in pgrid.values()])) if pgrid else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if n_comb <= 1:
            est = clone(estimator)
            if pgrid:
                est.set_params(**{k: v[0] for k, v in pgrid.items()})
            est.fit(X, y)
            return est, core_params({k: v[0] for k, v in pgrid.items()})
        search = GridSearchCV(clone(estimator), pgrid, cv=inner_cv(groups, n_inner), scoring=scoring,
                              n_jobs=n_jobs, refit=True, error_score=np.nan)
        search.fit(X, y, groups=groups)
    return search.best_estimator_, core_params(search.best_params_)


def fit_quiet(estimator, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return estimator.fit(X, y)
