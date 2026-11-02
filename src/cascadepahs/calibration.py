"""Few-shot cross-site calibration (Section 2.5, Note S6, Tables S21, S25, S26, Figs. 6 and S14–S15).

Primary scheme: the 10 calibration samples are drawn from a *single* sampling day
of the target site; the samples of the other days are validation samples. Each
day is used once (Yangyong River: 3, Shenzhen Bay: 4), with 200 random draws per day.
Reference scheme: the 10 samples are drawn at random from all days (optimistic).

In every draw, SOLACE and SORFE are refitted on the 142 source (Dasha River)
samples pooled with the same calibration samples, with architecture and
hyper-parameters fixed. Only the 14 optical descriptors defined identically at
all sites are used (PARAFAC components excluded). Validation samples are predicted
from the seven sonde variables alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .features import CROSS_SITE_OPTICAL_VARS, SENSOR_VARS, TARGET
from .metrics import concentration_range_share, coverage, describe_draws, mae, r2, rmse


def _xyog(df, optical_vars):
    return (df[SENSOR_VARS].to_numpy(float), df[TARGET].to_numpy(float), df[optical_vars].to_numpy(float),
            df["date"].astype(str).to_numpy())


def fit_pooled(factory, source: pd.DataFrame, cal: pd.DataFrame, optical_vars=CROSS_SITE_OPTICAL_VARS):
    pool = pd.concat([source, cal], ignore_index=True)
    X, y, O, g = _xyog(pool, optical_vars)
    return factory().fit(X, y, g, O)


def _one_draw(factories, source, cal, val, optical_vars, setting, draw, keep_predictions,
              importance_for=None, importance_repeats=10):
    Xv, yv, _, _ = _xyog(val, optical_vars)
    rows, preds = [], []
    for name, fac in factories.items():
        m = fit_pooled(fac, source, cal, optical_vars)
        p = m.predict(Xv)
        row = {"setting": setting, "draw": draw, "model": name, "n_cal": len(cal), "n_val": len(val),
               "R2": r2(yv, p), "RMSE": rmse(yv, p), "MAE": mae(yv, p),
               "range_share": concentration_range_share(cal[TARGET], val[TARGET]),
               "cal_ids": ";".join(cal["sample_id"].astype(str))}
        if hasattr(m, "predict_interval"):
            iv = m.predict_interval(Xv)
            row.update(coverage(yv, iv[:, 0], iv[:, 1]))
        if importance_for == name:
            imp = permutation_importance_pct(m, Xv, yv, n_repeats=importance_repeats, seed=draw)
            row.update({f"imp_{v}": x for v, x in zip(SENSOR_VARS, imp)})
        rows.append(row)
        if keep_predictions:
            preds.append(pd.DataFrame({"setting": setting, "draw": draw, "model": name,
                                       "sample_id": val["sample_id"].to_numpy(), "y": yv, "pred": p}))
    return rows, preds


def fewshot_calibration(factories: dict, source: pd.DataFrame, target: pd.DataFrame, *, n_cal: int = 10,
                        n_draws: int = 200, mode: str = "day", optical_vars=CROSS_SITE_OPTICAL_VARS,
                        seed: int = 0, n_jobs: int = 1, keep_predictions: bool = True,
                        importance_for: str | None = None, importance_repeats: int = 10):
    """Run the day-separated (``mode='day'``) or random-reference (``mode='random'``) calibration.

    Returns ``(draws, predictions)`` as long DataFrames.
    """
    rng = np.random.default_rng(seed)
    target = target.reset_index(drop=True)
    dates = target["date"].astype(str).to_numpy()
    tasks = []
    if mode == "day":
        for k, day in enumerate(sorted(np.unique(dates))):
            pool_idx = np.where(dates == day)[0]
            val = target[dates != day]
            for r in range(n_draws):
                cal_idx = rng.choice(pool_idx, size=min(n_cal, len(pool_idx)), replace=False)
                tasks.append((f"Day {k + 1}", r, target.iloc[cal_idx], val))
    elif mode == "random":
        for r in range(n_draws):
            cal_idx = rng.choice(len(target), size=n_cal, replace=False)
            mask = np.ones(len(target), bool)
            mask[cal_idx] = False
            tasks.append(("Random", r, target.iloc[cal_idx], target[mask]))
    else:
        raise ValueError("mode must be 'day' or 'random'")
    out = Parallel(n_jobs=n_jobs)(delayed(_one_draw)(factories, source, c, v, optical_vars, s, r, keep_predictions,
                                                     importance_for, importance_repeats)
                                  for s, r, c, v in tasks)
    draws = pd.DataFrame([row for rows, _ in out for row in rows])
    preds = pd.concat([p for _, ps in out for p in ps], ignore_index=True) if keep_predictions else None
    return draws, preds


def summarize_calibration(draws: pd.DataFrame, a: str = "SOLACE", b: str = "SORFE") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table S26: (A) per-model summaries across draws, (B) paired ΔR² = R²(a) − R²(b)."""
    rows = []
    for (setting, model), g in draws.groupby(["setting", "model"], sort=False):
        row = {"setting": setting, "model": model, "n_val": int(g["n_val"].iloc[0])}
        has_cov = "coverage" in g and g["coverage"].notna().any()
        for metric in ["R2", "RMSE", "MAE"] + (["coverage", "mean_width"] if has_cov else []):
            for k, v in describe_draws(g[metric]).items():
                row[f"{metric}_{k}"] = v
        rows.append(row)
    A = pd.DataFrame(rows)
    rows = []
    for setting, g in draws.groupby("setting", sort=False):
        wide = g.pivot(index="draw", columns="model", values="R2")
        if a in wide and b in wide:
            d = (wide[a] - wide[b]).to_numpy()
            rows.append({"setting": setting, **{f"dR2_{k}": v for k, v in describe_draws(d).items()},
                         f"{a}_better_pct": 100 * float(np.mean(d > 0))})
    return A, pd.DataFrame(rows)


def range_coverage_spearman(draws: pd.DataFrame, model: str = "SOLACE") -> dict:
    g = draws[draws["model"] == model]
    rho, p = stats.spearmanr(g["range_share"], g["R2"], nan_policy="omit")
    return {"spearman_rho": float(rho), "p": float(p), "median_range_share": float(np.nanmedian(g["range_share"]))}


def representative_draw(draws: pd.DataFrame, model: str = "SOLACE") -> tuple[str, int]:
    """Draw whose R² is closest to the median across all single-day draws (Fig. 6a–d)."""
    g = draws[(draws["model"] == model) & (draws["setting"] != "Random")]
    i = (g["R2"] - g["R2"].median()).abs().idxmin()
    return g.loc[i, "setting"], int(g.loc[i, "draw"])


def direct_transfer(fitted_models: dict, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero-calibration application of source-fitted models; returns (metrics, absolute errors)."""
    X = target[SENSOR_VARS].to_numpy(float)
    y = target[TARGET].to_numpy(float)
    rows, errs = [], {"sample_id": target["sample_id"].to_numpy(), "y": y}
    for name, m in fitted_models.items():
        p = m.predict(X)
        rows.append({"model": name, "R2": r2(y, p), "RMSE": rmse(y, p), "MAE": mae(y, p)})
        errs[f"pred_{name}"] = p
        errs[f"abs_err_{name}"] = np.abs(y - p)
    return pd.DataFrame(rows), pd.DataFrame(errs)


def calibration_size_curve(solace_factory, source_model, source: pd.DataFrame, target: pd.DataFrame, *,
                           sizes=(3, 5, 10, 15, 20), n_draws: int = 200, optical_vars=CROSS_SITE_OPTICAL_VARS,
                           local_params: dict | None = None, seed: int = 0, n_jobs: int = 1) -> pd.DataFrame:
    """Fig. S15: validation R² vs number of calibration samples (random selection).

    Methods: SOLACE refitted on source + n calibration samples; a local sensor-only RF
    trained on the n calibration samples alone; a linear correction y ≈ a + b·ŷ_source
    of the source-model output fitted on the calibration samples.
    """
    rng = np.random.default_rng(seed)
    target = target.reset_index(drop=True)
    Xt, yt, _, _ = _xyog(target, optical_vars)
    src_pred = source_model.predict(Xt)
    tasks = []
    for n in sizes:
        for r in range(n_draws):
            cal_idx = rng.choice(len(target), size=n, replace=False)
            tasks.append((n, r, cal_idx))

    def run(n, r, cal_idx):
        mask = np.ones(len(target), bool)
        mask[cal_idx] = False
        yv = yt[mask]
        m = fit_pooled(solace_factory, source, target.iloc[cal_idx], optical_vars)
        out = [{"n_cal": n, "draw": r, "method": "SOLACE (pooled refit)", "R2": r2(yv, m.predict(Xt[mask]))}]
        loc = RandomForestRegressor(random_state=r, **(local_params or {"n_estimators": 200}))
        loc.fit(Xt[cal_idx], yt[cal_idx])
        out.append({"n_cal": n, "draw": r, "method": "Local model", "R2": r2(yv, loc.predict(Xt[mask]))})
        lc = LinearRegression().fit(src_pred[cal_idx, None], yt[cal_idx])
        out.append({"n_cal": n, "draw": r, "method": "Linear correction", "R2": r2(yv, lc.predict(src_pred[mask, None]))})
        return out

    res = Parallel(n_jobs=n_jobs)(delayed(run)(n, r, c) for n, r, c in tasks)
    return pd.DataFrame([row for rows in res for row in rows])


def permutation_importance_pct(model, X: np.ndarray, y: np.ndarray, n_repeats: int = 10, seed: int = 0) -> np.ndarray:
    """Drop in R² when each sonde variable is permuted, expressed as % of the total drop (Fig. 6e–f)."""
    rng = np.random.default_rng(seed)
    base = r2(y, model.predict(X))
    drops = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        d = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            d.append(base - r2(y, model.predict(Xp)))
        drops[j] = max(0.0, float(np.mean(d)))
    tot = drops.sum()
    return 100 * drops / tot if tot > 0 else drops
