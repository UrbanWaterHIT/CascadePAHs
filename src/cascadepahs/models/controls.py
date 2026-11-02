"""Complexity-matched controls, alternative uses of the optical data and reference models.

All models share the interface ``fit(X, y, groups, O)`` / ``predict(X, O=None)``
(Section 2.4, Note S4.3, Table S17):

=========================  ==========================================================  ==========
Model                      Description                                                 Deployable
=========================  ==========================================================  ==========
capacity_control()         SOLACE architecture, leaves built from the sensor data        yes
shuffled_control()         SOLACE architecture, leaf targets permuted                    yes
TwoStageRegression         X → Ô (optical descriptors); [X, Ô] → ΣPAHs                   yes
GeneralizedDistillation    teacher RF(O) → soft labels; student RF(X)                    yes
SVRPlus (svr_plus.py)      optical descriptors model the slack variables                 yes
MultiTaskMLP               shared hidden layers; ΣPAHs + optical descriptors outputs     yes
OpticsOnlyRF               RF on measured optical descriptors                            no
SensorsPlusOpticsRF        RF on sonde variables + measured optical descriptors          no
=========================  ==========================================================  ==========
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import inner_cv, make_estimator, set_core_params, tune
from .solace import SOLACE


def capacity_control(**kw) -> SOLACE:
    return SOLACE(leaf_source="sensor", **kw)


def shuffled_control(**kw) -> SOLACE:
    return SOLACE(shuffle_leaf_targets=True, **kw)


def _rf(params, random_state, n_jobs):
    rf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
    return rf.set_params(**(params or {}))


def _fit_rf(X, y, groups, params, grid, do_tune, n_inner, random_state, n_jobs):
    est = make_estimator("RF", random_state=random_state, n_jobs=n_jobs)
    if do_tune and grid:
        return tune(est, grid, X, y, groups, n_inner=n_inner)
    if params:
        set_core_params(est, params)
    return est.fit(X, y), dict(params or {})


class _Frozen:
    def frozen_clone(self):
        m = clone(self)
        p = getattr(self, "best_params_", None)
        if p is not None and "tune" in m.get_params():
            m.set_params(tune=False)
            for k, v in p.items():
                if k in m.get_params():
                    m.set_params(**{k: v})
        return m


class TwoStageRegression(_Frozen, BaseEstimator, RegressorMixin):
    requires_optics_at_predict = False

    def __init__(self, params: dict | None = None, grid: dict | None = None, tune: bool = False,
                 n_inner: int = 5, random_state: int = 0, n_jobs: int = 1):
        self.params = params
        self.grid = grid
        self.tune = tune
        self.n_inner = n_inner
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, groups, O):
        X, y, O = np.asarray(X, float), np.asarray(y, float), np.asarray(O, float)
        p = dict(self.params or {})
        self.stage1_ = _rf(p, self.random_state, self.n_jobs).fit(X, O)
        O_oof = np.empty_like(O)
        for tr, te in inner_cv(groups, self.n_inner).split(X, y, groups):
            O_oof[te] = _rf(p, self.random_state, self.n_jobs).fit(X[tr], O[tr]).predict(X[te])
        self.stage2_, self.best_params_ = _fit_rf(np.hstack([X, O_oof]), y, groups, p, self.grid, self.tune,
                                                  self.n_inner, self.random_state, self.n_jobs)
        return self

    def predict(self, X, O=None):
        X = np.asarray(X, float)
        return self.stage2_.predict(np.hstack([X, self.stage1_.predict(X)]))


class GeneralizedDistillation(_Frozen, BaseEstimator, RegressorMixin):
    """Lopez-Paz et al. (2016) for regression: student target = (1−λ)·y + λ·teacher(O)."""

    requires_optics_at_predict = False

    def __init__(self, params: dict | None = None, lambdas=(0.25, 0.5, 0.75), imitation: float | None = None,
                 tune: bool = True, n_inner: int = 5, random_state: int = 0, n_jobs: int = 1):
        self.params = params
        self.lambdas = lambdas
        self.imitation = imitation
        self.tune = tune
        self.n_inner = n_inner
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _soft_labels(self, O, y, groups):
        s = np.empty_like(y)
        for tr, te in inner_cv(groups, self.n_inner).split(O, y, groups):
            s[te] = _rf(self.params, self.random_state, self.n_jobs).fit(O[tr], y[tr]).predict(O[te])
        return s

    def fit(self, X, y, groups, O):
        X, y, O, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(O, float), np.asarray(groups)
        self.teacher_ = _rf(self.params, self.random_state, self.n_jobs).fit(O, y)
        s = self._soft_labels(O, y, groups)            # out-of-fold teacher predictions
        lam = self.imitation
        if lam is None:
            if self.tune and len(self.lambdas) > 1:
                best = np.inf
                for cand in self.lambdas:
                    err = []
                    for tr, te in inner_cv(groups, self.n_inner).split(X, y, groups):
                        st = _rf(self.params, self.random_state, self.n_jobs)
                        st.fit(X[tr], (1 - cand) * y[tr] + cand * s[tr])
                        err.append(np.abs(st.predict(X[te]) - y[te]))
                    mae = np.mean(np.concatenate(err))
                    if mae < best:
                        best, lam = mae, cand
            else:
                lam = self.lambdas[0]
        self.best_params_ = {"imitation": lam}
        self.student_ = _rf(self.params, self.random_state, self.n_jobs).fit(X, (1 - lam) * y + lam * s)
        return self

    def predict(self, X, O=None):
        return self.student_.predict(np.asarray(X, float))


class MultiTaskMLP(_Frozen, BaseEstimator, RegressorMixin):
    """Multi-task learning (Caruana, 1997): one network, shared hidden layers.

    Outputs are the standardised ΣPAHs (main task) and the standardised optical
    descriptors (auxiliary tasks). ``aux_weight`` scales the squared loss of each
    auxiliary output (default 1/n_aux, so all auxiliary tasks together weigh as much
    as the main task). Only the sonde variables are inputs.
    """

    requires_optics_at_predict = False

    def __init__(self, hidden_grid=((50,), (100,), (100, 100)), hidden_layer_sizes=None,
                 aux_weight: float | None = None, tune: bool = True, n_inner: int = 5,
                 max_iter: int = 2000, random_state: int = 0):
        self.hidden_grid = hidden_grid
        self.hidden_layer_sizes = hidden_layer_sizes
        self.aux_weight = aux_weight
        self.tune = tune
        self.n_inner = n_inner
        self.max_iter = max_iter
        self.random_state = random_state

    def _fit_one(self, X, y, O, hidden):
        sx = StandardScaler().fit(X)
        sy = StandardScaler().fit(y.reshape(-1, 1))
        so = StandardScaler().fit(O)
        w_aux = self.aux_weight if self.aux_weight is not None else 1.0 / O.shape[1]
        Y = np.hstack([sy.transform(y.reshape(-1, 1)), so.transform(O) * np.sqrt(w_aux)])
        net = MLPRegressor(hidden_layer_sizes=tuple(hidden), max_iter=self.max_iter,
                           random_state=self.random_state, early_stopping=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            net.fit(sx.transform(X), Y)
        return sx, sy, net

    @staticmethod
    def _predict_one(parts, X):
        sx, sy, net = parts
        return sy.inverse_transform(net.predict(sx.transform(X))[:, :1]).ravel()

    def fit(self, X, y, groups, O):
        X, y, O, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(O, float), np.asarray(groups)
        hidden = self.hidden_layer_sizes
        if hidden is None:
            if self.tune and len(self.hidden_grid) > 1:
                best = np.inf
                for h in self.hidden_grid:
                    err = []
                    for tr, te in inner_cv(groups, self.n_inner).split(X, y, groups):
                        parts = self._fit_one(X[tr], y[tr], O[tr], h)
                        err.append(np.abs(self._predict_one(parts, X[te]) - y[te]))
                    mae = np.mean(np.concatenate(err))
                    if mae < best:
                        best, hidden = mae, h
            else:
                hidden = self.hidden_grid[0]
        self.best_params_ = {"hidden_layer_sizes": tuple(hidden)}
        self.parts_ = self._fit_one(X, y, O, hidden)
        return self

    def predict(self, X, O=None):
        return self._predict_one(self.parts_, np.asarray(X, float))


class OpticsOnlyRF(_Frozen, BaseEstimator, RegressorMixin):
    """Reference (NOT deployable): random forest on measured optical descriptors."""

    requires_optics_at_predict = True

    def __init__(self, params=None, grid=None, tune=False, n_inner=5, random_state=0, n_jobs=1):
        self.params, self.grid, self.tune = params, grid, tune
        self.n_inner, self.random_state, self.n_jobs = n_inner, random_state, n_jobs

    def fit(self, X, y, groups, O):
        self.model_, self.best_params_ = _fit_rf(np.asarray(O, float), y, groups, self.params, self.grid,
                                                 self.tune, self.n_inner, self.random_state, self.n_jobs)
        return self

    def predict(self, X, O=None):
        if O is None:
            raise ValueError("OpticsOnlyRF requires measured optical descriptors at prediction time")
        return self.model_.predict(np.asarray(O, float))


class SensorsPlusOpticsRF(OpticsOnlyRF):
    """Reference (NOT deployable): RF on sonde variables concatenated with measured optics."""

    def fit(self, X, y, groups, O):
        Z = np.hstack([np.asarray(X, float), np.asarray(O, float)])
        self.model_, self.best_params_ = _fit_rf(Z, y, groups, self.params, self.grid, self.tune,
                                                 self.n_inner, self.random_state, self.n_jobs)
        return self

    def predict(self, X, O=None):
        if O is None:
            raise ValueError("SensorsPlusOpticsRF requires measured optical descriptors at prediction time")
        return self.model_.predict(np.hstack([np.asarray(X, float), np.asarray(O, float)]))
