"""Sensor-only models. SORFE = Sensor-Only Random-Forest Estimator (Section 2.4)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .base import fit_quiet, make_estimator, set_core_params, tune


class SensorOnlyModel(BaseEstimator, RegressorMixin):
    """Maps the seven sonde variables directly to ΣPAHs with one of the six algorithms.

    ``fit(X, y, groups=None, O=None)`` – ``O`` (optical descriptors) is accepted for a
    uniform interface and ignored.
    """

    requires_optics_at_predict = False

    def __init__(self, algorithm: str = "RF", params: dict | None = None, grid: dict | None = None,
                 tune: bool = False, n_inner: int = 5, quantile: bool = False,
                 random_state: int = 0, n_jobs: int = 1):
        self.algorithm = algorithm
        self.params = params
        self.grid = grid
        self.tune = tune
        self.n_inner = n_inner
        self.quantile = quantile
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, groups=None, O=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        est = make_estimator(self.algorithm, scale_X=True, random_state=self.random_state,
                             n_jobs=self.n_jobs, quantile=self.quantile and self.algorithm == "RF")
        if self.tune and self.grid:
            if groups is None:
                raise ValueError("date groups are required for tuning")
            self.estimator_, self.best_params_ = tune(est, self.grid, X, y, groups, n_inner=self.n_inner,
                                                      n_jobs=self.n_jobs)
        else:
            if self.params:
                set_core_params(est, self.params)
            self.estimator_ = fit_quiet(est, X, y)
            self.best_params_ = dict(self.params or {})
        self.X_train_min_ = X.min(axis=0)
        self.X_train_max_ = X.max(axis=0)
        return self

    def predict(self, X, O=None):
        return self.estimator_.predict(np.asarray(X, float))

    def predict_quantiles(self, X, quantiles=(0.05, 0.95)):
        if not (self.quantile and self.algorithm == "RF"):
            raise RuntimeError("quantiles are available only for algorithm='RF' with quantile=True")
        return self.estimator_.named_steps["est"].predict_quantiles(np.asarray(X, float), quantiles)

    def frozen_clone(self) -> "SensorOnlyModel":
        """Unfitted copy with the tuned hyper-parameters fixed (no further tuning)."""
        from sklearn.base import clone

        return clone(self).set_params(tune=False, params=dict(self.best_params_))

    # tree model (for SHAP TreeExplainer)
    @property
    def core_(self):
        return self.estimator_.named_steps["est"] if hasattr(self.estimator_, "named_steps") else self.estimator_


def SORFE(**kwargs) -> SensorOnlyModel:
    """Sensor-Only Random-Forest Estimator."""
    kwargs.setdefault("algorithm", "RF")
    return SensorOnlyModel(**kwargs)
