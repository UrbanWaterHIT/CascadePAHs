"""SHAP attributions and sensor-to-optical predictability (Section 2.5, Note S3.4, Table S22, Fig. 5).

* panel a – first-stage optical forest (17 descriptors): TreeExplainer
* panel b – SORFE (7 sonde variables): TreeExplainer
* panel c – complete SOLACE deployment pipeline as a function of the 7 sonde
  variables: permutation explainer, background = training samples of each fold
* panel d – R² of predicting each optical descriptor from the 7 sonde variables

The ``shap`` package is used when installed. Otherwise interventional Shapley
values are computed here: exactly (all 2^M coalitions) for M ≤ 12 features,
and by antithetic permutation sampling for larger M. With the same background
data both estimate the interventional SHAP values computed by ``shap``.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import SENSOR_VARS
from .metrics import r2

try:
    import shap  # type: ignore

    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False


# ----------------------------------------------------------------------------- fallback Shapley
def exact_interventional_shap(f, X: np.ndarray, background: np.ndarray, batch_rows: int = 200_000) -> np.ndarray:
    """Exact interventional Shapley values: v(S) = mean_b f(x_S, b_~S)."""
    X, B = np.asarray(X, float), np.asarray(background, float)
    n, M = X.shape
    masks = np.array(list(itertools.product([0, 1], repeat=M)), dtype=bool)        # (2^M, M)
    K, nb = len(masks), len(B)
    v = np.empty((n, K))
    per = max(1, batch_rows // (K * nb))
    for s in range(0, n, per):
        xs = X[s:s + per]
        Z = np.where(masks[None, :, None, :], xs[:, None, None, :], B[None, None, :, :])  # (m, K, nb, M)
        v[s:s + per] = f(Z.reshape(-1, M)).reshape(len(xs), K, nb).mean(axis=2)
    index = {tuple(m): k for k, m in enumerate(masks)}
    phi = np.zeros((n, M))
    for j in range(M):
        for k, m in enumerate(masks):
            if m[j]:
                continue
            size = int(m.sum())
            w = math.factorial(size) * math.factorial(M - size - 1) / math.factorial(M)
            m2 = m.copy()
            m2[j] = True
            phi[:, j] += w * (v[:, index[tuple(m2)]] - v[:, k])
    return phi


def sampling_interventional_shap(f, X: np.ndarray, background: np.ndarray, n_permutations: int = 32,
                                 seed: int = 0) -> np.ndarray:
    """Antithetic permutation-sampling estimate of interventional Shapley values."""
    rng = np.random.default_rng(seed)
    X, B = np.asarray(X, float), np.asarray(background, float)
    n, M = X.shape
    phi = np.zeros((n, M))
    perms = []
    for _ in range(max(1, n_permutations // 2)):
        p = rng.permutation(M)
        perms += [p, p[::-1]]
    for i in range(n):
        for p in perms:
            Z = np.repeat(B[None], M + 1, axis=0)                 # (M+1, nb, M)
            for k in range(1, M + 1):
                Z[k:, :, p[k - 1]] = X[i, p[k - 1]]
            vals = f(Z.reshape(-1, M)).reshape(M + 1, len(B)).mean(axis=1)
            phi[i, p] += np.diff(vals)
    return phi / len(perms)


# ----------------------------------------------------------------------------- explainers
def tree_shap(model, X: np.ndarray, background: np.ndarray, n_permutations: int = 32) -> np.ndarray:
    if HAS_SHAP:
        ex = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        return np.asarray(ex.shap_values(X, check_additivity=False))
    f = model.predict
    if X.shape[1] <= 12:
        return exact_interventional_shap(f, X, background)
    return sampling_interventional_shap(f, X, background, n_permutations)


def pipeline_shap(predict_fn, X: np.ndarray, background: np.ndarray, max_evals: int | None = None) -> np.ndarray:
    """Permutation explainer for a black-box function of the seven sonde variables."""
    if HAS_SHAP:
        masker = shap.maskers.Independent(background, max_samples=len(background))
        ex = shap.PermutationExplainer(predict_fn, masker)
        M = X.shape[1]
        return np.asarray(ex(X, max_evals=max_evals or 10 * (2 * M + 1), silent=True).values)
    if X.shape[1] <= 12:
        return exact_interventional_shap(predict_fn, X, background)
    return sampling_interventional_shap(predict_fn, X, background)


def shap_over_folds(df: pd.DataFrame, folds, *, sorfe_factory, solace_factory, optical_vars, max_background: int = 50,
                    summer_only: bool = True, seed: int = 0) -> dict[str, pd.DataFrame]:
    """SHAP values of the three models on the held-out samples of the blocked validation.

    Returns DataFrames (one row per held-out sample) of SHAP values and feature values
    for keys ``optical``, ``sorfe`` and ``solace``.
    """
    rng = np.random.default_rng(seed)
    from .features import TARGET

    X = df[SENSOR_VARS].to_numpy(float)
    O = df[optical_vars].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    g = df["date"].astype(str).to_numpy()
    res = {"optical": [], "sorfe": [], "solace": []}
    for f in folds:
        if summer_only and f.kind != "summer":
            continue
        tr, te = f.train, f.test
        bg = rng.choice(tr, size=min(max_background, len(tr)), replace=False)
        sol = solace_factory().fit(X[tr], y[tr], g[tr], O[tr])
        sor = sorfe_factory().fit(X[tr], y[tr], g[tr], O[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_opt = tree_shap(sol.optical_forest_, O[te], O[bg])
            s_sor = tree_shap(sor.core_, X[te], X[bg])
            s_sol = pipeline_shap(sol.predict, X[te], X[bg])
        meta = df.iloc[te][["sample_id"]].assign(fold=f.name).reset_index(drop=True)
        res["optical"].append(pd.concat([meta, pd.DataFrame(s_opt, columns=[f"shap_{c}" for c in optical_vars]),
                                         pd.DataFrame(O[te], columns=optical_vars)], axis=1))
        for key, s in [("sorfe", s_sor), ("solace", s_sol)]:
            res[key].append(pd.concat([meta, pd.DataFrame(s, columns=[f"shap_{c}" for c in SENSOR_VARS]),
                                       pd.DataFrame(X[te], columns=SENSOR_VARS)], axis=1))
    return {k: pd.concat(v, ignore_index=True) for k, v in res.items()}


def descriptor_predictability(df: pd.DataFrame, folds, optical_vars, rf_params: dict | None = None,
                              summer_only: bool = True, n_jobs: int = 1) -> pd.DataFrame:
    """R² of predicting each optical descriptor from the seven sonde variables (Fig. 5d)."""
    X = df[SENSOR_VARS].to_numpy(float)
    rows = []
    for c in optical_vars:
        o = df[c].to_numpy(float)
        yy, pp = [], []
        for f in folds:
            if summer_only and f.kind != "summer":
                continue
            rf = RandomForestRegressor(random_state=0, n_jobs=n_jobs, **(rf_params or {"n_estimators": 200}))
            rf.fit(X[f.train], o[f.train])
            yy.append(o[f.test])
            pp.append(rf.predict(X[f.test]))
        rows.append({"descriptor": c, "R2": r2(np.concatenate(yy), np.concatenate(pp))})
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


def mean_abs_shap(table: pd.DataFrame) -> pd.Series:
    cols = [c for c in table.columns if c.startswith("shap_")]
    return table[cols].abs().mean().rename(lambda c: c[5:]).sort_values(ascending=False)
