"""Conformalized quantile regression (Romano et al., 2019) with date-grouped scores.

The final stage of the model is a quantile regression forest giving the 5th and
95th conditional percentiles. Conformity scores
``E_i = max(q_lo(x_i) − y_i, y_i − q_hi(x_i))`` are computed from out-of-fold
predictions of the *complete pipeline* with folds grouped by survey date, so a
sample's own day is never used both to fit and to calibrate its interval
(Section 2.5, Note S3.6). The interval for a new record is
``[q_lo(x) − Q, q_hi(x) + Q]`` with ``Q`` the ⌈(1−α)(n+1)⌉/n empirical quantile of E.
"""
from __future__ import annotations

import numpy as np

from .base import inner_cv


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k-th smallest score with k = ⌈(1 − α)(n + 1)⌉ (the largest score if k > n)."""
    s = np.sort(np.asarray(scores, float))
    k = int(np.ceil((1 - alpha) * (len(s) + 1)))
    return float(s[min(k, len(s)) - 1])


class ConformalIntervalModel:
    """Wraps a model exposing ``fit(X, y, groups, O)``, ``predict`` and ``predict_quantiles``."""

    def __init__(self, model, alpha: float = 0.10, n_folds: int = 5):
        self.model = model
        self.alpha = alpha
        self.n_folds = n_folds

    def fit(self, X, y, groups, O=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        groups = np.asarray(groups)
        q = (self.alpha / 2, 1 - self.alpha / 2)
        self.quantiles_ = q
        self.model_ = self.model.fit(X, y, groups, O)
        scores = np.empty(len(y))
        for tr, te in inner_cv(groups, self.n_folds).split(X, y, groups):
            m = self.model_.frozen_clone()
            m.fit(X[tr], y[tr], groups[tr], None if O is None else np.asarray(O)[tr])
            qq = m.predict_quantiles(X[te], q)
            scores[te] = np.maximum(qq[:, 0] - y[te], y[te] - qq[:, 1])
        self.scores_ = scores
        self.qhat_ = conformal_quantile(scores, self.alpha)
        return self

    def predict(self, X, O=None):
        return self.model_.predict(X)

    def predict_interval(self, X) -> np.ndarray:
        """Return (n, 2) array of calibrated lower and upper bounds."""
        qq = self.model_.predict_quantiles(X, self.quantiles_)
        return np.column_stack([qq[:, 0] - self.qhat_, qq[:, 1] + self.qhat_])

    def __getattr__(self, item):
        # delegate (e.g. predict_leaves, X_train_min_) to the fitted model
        if item.startswith("__") or "model_" not in self.__dict__:
            raise AttributeError(item)
        return getattr(self.model_, item)
