"""Descriptive and dependence statistics used in the paper.

* one-way ANOVA of ΣPAHs by survey date and the intraclass correlation (Section 2.5)
* lag-1 correlation of daily means
* Fisher's exact test / χ² test of sample losses (Note S1.6)
* Kruskal–Wallis test with ε² = H/(n − 1) (composition, Fig. S6)
* lagged correlation of daily rainfall and daily mean ΣPAHs (Table S27)
* autocorrelation of daily mean residuals and its ±1.96/√n bound (block-length criterion)
* one-way MANOVA statistics (Wilks, Pillai, Hotelling–Lawley, Roy) for site heterogeneity (Table S9)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def anova_icc(values, groups) -> dict:
    """One-way ANOVA and ICC(1) = (MSB − MSW) / (MSB + (k0 − 1) MSW) for unbalanced groups."""
    df = pd.DataFrame({"v": values, "g": groups})
    gs = [g["v"].to_numpy() for _, g in df.groupby("g")]
    F, p = stats.f_oneway(*gs)
    n_i = np.array([len(g) for g in gs])
    N, a = n_i.sum(), len(gs)
    grand = df["v"].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in gs)
    msb, msw = ssb / (a - 1), ssw / (N - a)
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    return {"F": float(F), "df1": a - 1, "df2": int(N - a), "p": float(p), "ICC": float(icc)}


def lag_correlation(daily: pd.Series, lag: int = 1) -> float:
    s = daily.sort_index()
    return float(np.corrcoef(s.to_numpy()[:-lag], s.to_numpy()[lag:])[0, 1])


def fisher_loss_test(analysed_a: int, scheduled_a: int, analysed_b: int, scheduled_b: int) -> dict:
    table = [[analysed_a, scheduled_a - analysed_a], [analysed_b, scheduled_b - analysed_b]]
    odds, p = stats.fisher_exact(table)
    return {"odds_ratio": float(odds), "p": float(p)}


def chi2_loss_by_station(analysed: dict, scheduled_per_station: int) -> dict:
    obs = np.array([[a, scheduled_per_station - a] for a in analysed.values()])
    chi2, p, dof, _ = stats.chi2_contingency(obs)
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p)}


def kruskal_epsilon2(values, groups) -> dict:
    df = pd.DataFrame({"v": values, "g": groups})
    H, p = stats.kruskal(*[g["v"].to_numpy() for _, g in df.groupby("g")])
    n = len(df)
    return {"H": float(H), "df": df["g"].nunique() - 1, "p": float(p), "epsilon2": float(H / (n - 1))}


def rainfall_lag_correlations(rain: pd.Series, daily_mean: pd.Series, lags=(0, 1, 2, 3)) -> pd.DataFrame:
    """Correlation between rainfall on day t − lag and daily mean ΣPAHs on day t."""
    rain = rain.sort_index()
    daily_mean = daily_mean.sort_index()
    rows = []
    for L in lags:
        r_shift = rain.shift(L)
        j = pd.concat([r_shift, daily_mean], axis=1, join="inner").dropna()
        rows.append({"lag_days": L, "r": float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]), "n_days": len(j)})
    return pd.DataFrame(rows)


def residual_acf(daily_residuals: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Autocorrelation of daily mean residuals; bound ±1.96/√n for zero autocorrelation."""
    x = daily_residuals.sort_index().to_numpy(float)
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x ** 2)
    rows = [{"lag": k, "acf": float(np.sum(x[:-k] * x[k:]) / denom), "bound": 1.96 / np.sqrt(n)}
            for k in range(1, max_lag + 1)]
    return pd.DataFrame(rows)


def manova(X: np.ndarray, groups) -> pd.DataFrame:
    """One-way MANOVA test statistics with F approximations (standardise X beforehand)."""
    X = np.asarray(X, float)
    groups = np.asarray(groups)
    labels = np.unique(groups)
    n, p = X.shape
    g = len(labels)
    mean = X.mean(0)
    E = np.zeros((p, p))
    H = np.zeros((p, p))
    for lab in labels:
        Xi = X[groups == lab]
        mi = Xi.mean(0)
        E += (Xi - mi).T @ (Xi - mi)
        H += len(Xi) * np.outer(mi - mean, mi - mean)
    eig = np.clip(np.real(np.linalg.eigvals(np.linalg.solve(E, H))), 0, None)
    eig = np.sort(eig)[::-1][: min(p, g - 1)]
    q, ve = g - 1, n - g
    s = min(p, q)
    m = (abs(p - q) - 1) / 2
    nn = (ve - p - 1) / 2
    rows = []
    wilks = float(np.prod(1 / (1 + eig)))
    # Rao's F approximation
    t = np.sqrt((p ** 2 * q ** 2 - 4) / (p ** 2 + q ** 2 - 5)) if p ** 2 + q ** 2 - 5 > 0 else 1.0
    df1 = p * q
    df2 = (ve - (p - q + 1) / 2) * t - (p * q - 2) / 2
    Fw = (1 - wilks ** (1 / t)) / (wilks ** (1 / t)) * df2 / df1
    rows.append({"test": "Wilks' lambda", "value": wilks, "F": Fw, "df1": df1, "df2": df2,
                 "p": float(stats.f.sf(Fw, df1, df2))})
    pillai = float(np.sum(eig / (1 + eig)))
    Fp = (2 * nn + s + 1) / (2 * m + s + 1) * pillai / (s - pillai)
    d1, d2 = s * (2 * m + s + 1), s * (2 * nn + s + 1)
    rows.append({"test": "Pillai's trace", "value": pillai, "F": Fp, "df1": d1, "df2": d2,
                 "p": float(stats.f.sf(Fp, d1, d2))})
    hl = float(np.sum(eig))
    Fh = 2 * (s * nn + 1) * hl / (s ** 2 * (2 * m + s + 1))
    d1, d2 = s * (2 * m + s + 1), 2 * (s * nn + 1)
    rows.append({"test": "Hotelling–Lawley trace", "value": hl, "F": Fh, "df1": d1, "df2": d2,
                 "p": float(stats.f.sf(Fh, d1, d2))})
    roy = float(eig[0]) if len(eig) else 0.0
    r = max(p, q)
    Fr = roy * (ve - r + q) / r
    rows.append({"test": "Roy's largest root", "value": roy, "F": Fr, "df1": r, "df2": ve - r + q,
                 "p": float(stats.f.sf(Fr, r, ve - r + q))})
    return pd.DataFrame(rows)
