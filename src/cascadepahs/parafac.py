"""Non-negative PARAFAC of EEM cubes (Note S2.1).

X[i, j, k] ≈ Σ_r A[i, r] · B[j, r] · C[k, r]   (i: sample, j: excitation, k: emission)

Fitted by hierarchical alternating least squares (HALS) with non-negativity;
missing values (NaN, e.g. un-interpolated scatter) are handled by EM imputation.
After fitting, each excitation and emission loading is scaled to a maximum of 1,
so A holds Fmax (R.U.), the maximum fluorescence intensity of each component.

``split_half`` checks the number of components by fitting two random halves of
the samples and comparing the loadings with Tucker's congruence coefficient.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _reconstruct(A, B, C):
    return np.einsum("ir,jr,kr->ijk", A, B, C)


def _hals_update(F, XV, G, eps=1e-12):
    for r in range(F.shape[1]):
        F[:, r] = np.maximum(eps, F[:, r] + (XV[:, r] - F @ G[:, r]) / max(G[r, r], eps))
    return F


def nn_parafac(X: np.ndarray, rank: int, *, n_iter: int = 2000, tol: float = 1e-8, n_init: int = 5,
               seed: int = 0) -> dict:
    """Non-negative CP decomposition. Returns dict(A=Fmax, B=ex loadings, C=em loadings, explained, iters)."""
    X = np.asarray(X, float)
    miss = np.isnan(X)
    has_miss = miss.any()
    rng = np.random.default_rng(seed)
    best = None
    for init in range(n_init):
        Xf = np.where(miss, np.nanmean(X), X)
        n, J, K = X.shape
        A = rng.random((n, rank)) + 0.1
        B = rng.random((J, rank)) + 0.1
        C = rng.random((K, rank)) + 0.1
        prev = np.inf
        for it in range(n_iter):
            A = _hals_update(A, np.einsum("ijk,jr,kr->ir", Xf, B, C), (B.T @ B) * (C.T @ C))
            B = _hals_update(B, np.einsum("ijk,ir,kr->jr", Xf, A, C), (A.T @ A) * (C.T @ C))
            C = _hals_update(C, np.einsum("ijk,ir,jr->kr", Xf, A, B), (A.T @ A) * (B.T @ B))
            R = _reconstruct(A, B, C)
            if has_miss:
                Xf = np.where(miss, R, X)
            sse = float(np.sum((Xf - R)[~miss] ** 2))
            if np.isfinite(prev) and abs(prev - sse) <= tol * max(prev, 1e-12):
                break
            prev = sse
        if best is None or sse < best[0]:
            best = (sse, A.copy(), B.copy(), C.copy(), it + 1)
    sse, A, B, C, iters = best
    # scale loadings to max 1 -> A = Fmax
    sb, sc = B.max(axis=0), C.max(axis=0)
    B, C, A = B / sb, C / sc, A * sb * sc
    order = np.argsort(np.argmax(C, axis=0))[::-1]                # longest emission maximum first (humic-like C1)
    A, B, C = A[:, order], B[:, order], C[:, order]
    sst = float(np.sum((X[~miss] - 0) ** 2))
    return {"A": A, "B": B, "C": C, "explained": 1 - sse / sst, "iters": iters, "sse": sse}


def tucker_congruence(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / np.sqrt((u @ u) * (v @ v)))


def match_components(L1: np.ndarray, L2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal matching of component columns by Tucker congruence; returns (order for L2, TCC)."""
    R = L1.shape[1]
    T = np.array([[tucker_congruence(L1[:, a], L2[:, b]) for b in range(R)] for a in range(R)])
    rows, cols = linear_sum_assignment(-T)
    return cols, T[rows, cols]


def split_half(X: np.ndarray, rank: int, seed: int = 0, **kw) -> dict:
    """Split-half validation: TCC of excitation and emission loadings between two random halves."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
    m1 = nn_parafac(X[h1], rank, seed=seed, **kw)
    m2 = nn_parafac(X[h2], rank, seed=seed + 1, **kw)
    order, tcc_em = match_components(m1["C"], m2["C"])
    tcc_ex = np.array([tucker_congruence(m1["B"][:, r], m2["B"][:, order[r]]) for r in range(rank)])
    return {"tcc_ex": tcc_ex, "tcc_em": tcc_em, "valid": bool(np.all(tcc_ex > 0.95) and np.all(tcc_em > 0.95))}


def component_peaks(model: dict, ex, em) -> list[dict]:
    """Excitation/emission maxima of each component (for comparison with OpenFluor)."""
    out = []
    for r in range(model["B"].shape[1]):
        out.append({"component": f"C{r + 1}", "ex_max_nm": float(np.asarray(ex)[np.argmax(model["B"][:, r])]),
                    "em_max_nm": float(np.asarray(em)[np.argmax(model["C"][:, r])])})
    return out
