"""Performance metrics, paired date-level bootstrap and screening statistics (Section 2.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def bias(y, p) -> float:
    """mean(predicted − observed)."""
    return float(np.mean(np.asarray(p) - np.asarray(y)))


def summary(y, p) -> dict:
    return {"n": len(y), "R2": r2(y, p), "MAE": mae(y, p), "RMSE": rmse(y, p), "bias": bias(y, p),
            "median_pred": float(np.median(p))}


def paired_date_bootstrap(df: pd.DataFrame, models: list[str], *, date_col: str = "date", y_col: str = "y",
                          n_boot: int = 2000, seed: int = 0, metric=r2, ci: float = 0.95) -> pd.DataFrame:
    """95 % CIs of a metric for each model and of pairwise differences, resampling survey dates.

    ``df`` holds one row per held-out sample with the observed value ``y`` and one
    column of predictions per model name in ``models``. Differences are computed
    as model[0] − model[k] for k ≥ 1 (e.g. SOLACE − SORFE).
    """
    rng = np.random.default_rng(seed)
    dates = df[date_col].astype(str).to_numpy()
    uniq = np.unique(dates)
    idx_by_date = {d: np.where(dates == d)[0] for d in uniq}
    y = df[y_col].to_numpy(float)
    P = {m: df[m].to_numpy(float) for m in models}
    boot = {m: [] for m in models}
    diffs = {m: [] for m in models[1:]}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_date[d] for d in pick])
        vals = {m: metric(y[ii], P[m][ii]) for m in models}
        for m in models:
            boot[m].append(vals[m])
        for m in models[1:]:
            diffs[m].append(vals[models[0]] - vals[m])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    rows = []
    for m in models:
        rows.append({"model": m, "estimate": metric(y, P[m]), "ci_low": np.nanpercentile(boot[m], lo),
                     "ci_high": np.nanpercentile(boot[m], hi)})
    for m in models[1:]:
        rows.append({"model": f"{models[0]} - {m}", "estimate": metric(y, P[models[0]]) - metric(y, P[m]),
                     "ci_low": np.nanpercentile(diffs[m], lo), "ci_high": np.nanpercentile(diffs[m], hi)})
    return pd.DataFrame(rows)


def blockwise(df: pd.DataFrame, models: list[str], block_col: str = "fold", y_col: str = "y") -> pd.DataFrame:
    rows = []
    for b, g in df.groupby(block_col, sort=False):
        row = {"block": b, "n": len(g)}
        for m in models:
            row[f"MAE_{m}"] = mae(g[y_col], g[m])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_block_tests(block_table: pd.DataFrame, a: str, b: str) -> dict:
    """Wilcoxon signed-rank and two-sided sign test on block MAEs (model a vs model b)."""
    d = block_table[f"MAE_{b}"].to_numpy() - block_table[f"MAE_{a}"].to_numpy()   # positive: a better
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    k = int(np.sum(d > 0))
    n = int(np.sum(d != 0))
    return {"n_blocks": len(d), f"{a}_better": k, "wilcoxon_p": float(w.pvalue) if w else np.nan,
            "sign_test_p": float(stats.binomtest(k, n, 0.5).pvalue) if n else np.nan}


# ----------------------------------------------------------------------------- screening / intervals
def screening(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pos, pred = y > threshold, p > threshold
    sens = float(np.mean(pred[pos])) if pos.any() else np.nan
    spec = float(np.mean(~pred[~pos])) if (~pos).any() else np.nan
    return {"threshold": threshold, "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": np.nanmean([sens, spec])}


def interval_triage(y, lo, hi, threshold: float) -> dict:
    """Confident 'above'/'below' when the whole interval lies on one side; otherwise flag for the lab."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    above, below = lo > threshold, hi < threshold
    conf = above | below
    correct = (above & (y > threshold)) | (below & (y <= threshold))
    return {"threshold": threshold, "share_confident": float(conf.mean()),
            "accuracy_confident": float(correct[conf].mean()) if conf.any() else np.nan,
            "share_flagged": float((~conf).mean())}


def coverage(y, lo, hi) -> dict:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo))}


def concentration_range_share(y_cal, y_val) -> float:
    """Share of the validation concentration range covered by the calibration samples."""
    rng_v = np.ptp(y_val)
    if rng_v == 0:
        return np.nan
    lo, hi = max(np.min(y_cal), np.min(y_val)), min(np.max(y_cal), np.max(y_val))
    return float(max(0.0, hi - lo) / rng_v)


def describe_draws(values) -> dict:
    v = np.asarray(values, float)
    return {"median": np.nanmedian(v), "q1": np.nanpercentile(v, 25), "q3": np.nanpercentile(v, 75),
            "p2.5": np.nanpercentile(v, 2.5), "p97.5": np.nanpercentile(v, 97.5)}
