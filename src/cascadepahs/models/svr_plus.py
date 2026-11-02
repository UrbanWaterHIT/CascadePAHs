"""SVR+ — regression form of SVM+ (Vapnik & Vashist, 2009), a classical LUPI method.

The slack variables of the ε-insensitive regression in the sensor space are
modelled by correcting functions in the privileged (optical) space::

    min  ½‖w‖² + γ/2 (‖w*‖² + ‖w**‖²) + C Σ_i (ξ_i + ξ*_i)
    ξ_i  = ⟨w*,  φ*(z_i)⟩ + b*   ≥ 0
    ξ*_i = ⟨w**, φ*(z_i)⟩ + b**  ≥ 0
    s.t. y_i − ⟨w, φ(x_i)⟩ − b ≤ ε + ξ_i,   ⟨w, φ(x_i)⟩ + b − y_i ≤ ε + ξ*_i

Dual (solved here with SLSQP, variables scaled by C)::

    min  ε1ᵀ(α+α*) − yᵀ(α−α*) + ½(α−α*)ᵀK(α−α*)
         + 1/(2γ) [(α+β−C)ᵀK*(α+β−C) + (α*+β*−C)ᵀK*(α*+β*−C)]
    s.t. 1ᵀ(α−α*) = 0,  1ᵀ(α+β−C) = 0,  1ᵀ(α*+β*−C) = 0,  α, α*, β, β* ≥ 0

Prediction uses the sensor space only: f(x) = Σ_i (α_i − α*_i) K(x_i, x) + b.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.preprocessing import StandardScaler

from .base import inner_cv


def _kernel(A, B, kind: str, gamma: float | None):
    if kind == "rbf":
        return rbf_kernel(A, B, gamma=gamma)
    if kind == "linear":
        return linear_kernel(A, B)
    raise ValueError(kind)


class SVRPlusCore(BaseEstimator, RegressorMixin):
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, gamma_priv: float = 1.0,
                 kernel: str = "rbf", kernel_priv: str = "rbf", max_iter: int = 500):
        self.C = C
        self.epsilon = epsilon
        self.gamma_priv = gamma_priv
        self.kernel = kernel
        self.kernel_priv = kernel_priv
        self.max_iter = max_iter

    def fit(self, X, y, O):
        self.sx_ = StandardScaler().fit(X)
        self.so_ = StandardScaler().fit(O)
        self.sy_ = StandardScaler().fit(np.asarray(y, float).reshape(-1, 1))
        Xs, Os = self.sx_.transform(X), self.so_.transform(O)
        ys = self.sy_.transform(np.asarray(y, float).reshape(-1, 1)).ravel()
        n = len(ys)
        self.gx_ = 1.0 / X.shape[1]
        go = 1.0 / O.shape[1]
        K = _kernel(Xs, Xs, self.kernel, self.gx_)
        Ks = _kernel(Os, Os, self.kernel_priv, go)
        C, eps, g = float(self.C), float(self.epsilon), float(self.gamma_priv)

        # variables u = v / C with v = [α, α*, β, β*]
        def unpack(u):
            return u[:n], u[n:2 * n], u[2 * n:3 * n], u[3 * n:]

        def fun(u):
            a, as_, b, bs = unpack(u)
            d = a - as_
            p = a + b - 1.0
            q = as_ + bs - 1.0
            Kd, Kp, Kq = K @ d, Ks @ p, Ks @ q
            f = eps * (a.sum() + as_.sum()) - ys @ d + 0.5 * C * d @ Kd + 0.5 * C / g * (p @ Kp + q @ Kq)
            ga = eps - ys + C * Kd + C / g * Kp
            gas = eps + ys - C * Kd + C / g * Kq
            gb = C / g * Kp
            gbs = C / g * Kq
            return f, np.concatenate([ga, gas, gb, gbs])

        ones, zeros = np.ones(n), np.zeros(n)
        A_eq = np.vstack([
            np.concatenate([ones, -ones, zeros, zeros]),
            np.concatenate([ones, zeros, ones, zeros]),
            np.concatenate([zeros, ones, zeros, ones]),
        ])
        b_eq = np.array([0.0, n, n])
        cons = {"type": "eq", "fun": lambda u: A_eq @ u - b_eq, "jac": lambda u: A_eq}
        u0 = np.concatenate([zeros, zeros, ones, ones])          # feasible start: α = α* = 0, β = β* = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(fun, u0, jac=True, method="SLSQP", bounds=[(0, None)] * (4 * n),
                           constraints=[cons], options={"maxiter": self.max_iter, "ftol": 1e-9})
        u = np.clip(res.x, 0, None)
        a, as_, b, bs = unpack(u)
        self.dual_coef_ = C * (a - as_)
        self.X_fit_ = Xs
        f0 = K @ self.dual_coef_
        tol = 1e-5 * max(1.0, u.max())
        cand = list(ys[(a > tol) & (b > tol)] - f0[(a > tol) & (b > tol)] - eps)
        cand += list(ys[(as_ > tol) & (bs > tol)] - f0[(as_ > tol) & (bs > tol)] + eps)
        self.intercept_ = float(np.mean(cand)) if cand else float(np.median(ys - f0))
        self.converged_ = bool(res.success)
        return self

    def predict(self, X):
        Xs = self.sx_.transform(np.asarray(X, float))
        f = _kernel(Xs, self.X_fit_, self.kernel, self.gx_) @ self.dual_coef_ + self.intercept_
        return self.sy_.inverse_transform(f.reshape(-1, 1)).ravel()


class SVRPlus(BaseEstimator, RegressorMixin):
    """SVR+ with date-grouped tuning of C and ε. Uniform interface ``fit(X, y, groups, O)``."""

    requires_optics_at_predict = False

    def __init__(self, C_grid=(0.1, 1, 10, 100), epsilon_grid=(0.01, 0.1, 0.5), gamma_priv: float = 1.0,
                 tune: bool = True, C: float = 1.0, epsilon: float = 0.1, n_inner: int = 5):
        self.C_grid = C_grid
        self.epsilon_grid = epsilon_grid
        self.gamma_priv = gamma_priv
        self.tune = tune
        self.C = C
        self.epsilon = epsilon
        self.n_inner = n_inner

    def fit(self, X, y, groups, O):
        X, y, O, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(O, float), np.asarray(groups)
        best = (self.C, self.epsilon)
        if self.tune:
            splits = list(inner_cv(groups, self.n_inner).split(X, y, groups))
            best_mae = np.inf
            for C in self.C_grid:
                for eps in self.epsilon_grid:
                    err = []
                    for tr, te in splits:
                        m = SVRPlusCore(C=C, epsilon=eps, gamma_priv=self.gamma_priv).fit(X[tr], y[tr], O[tr])
                        err.append(np.abs(m.predict(X[te]) - y[te]))
                    mae = np.mean(np.concatenate(err))
                    if mae < best_mae:
                        best_mae, best = mae, (C, eps)
        self.best_params_ = {"C": best[0], "epsilon": best[1]}
        self.model_ = SVRPlusCore(C=best[0], epsilon=best[1], gamma_priv=self.gamma_priv).fit(X, y, O)
        return self

    def predict(self, X, O=None):
        return self.model_.predict(X)
