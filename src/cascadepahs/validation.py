"""Validation schemes and the cross-validation runner (Section 2.5, Note S3.5).

Schemes on the 142 Dasha River paired samples
---------------------------------------------
(a) ``folds_random``         stratified random 80:20 splitting, repeated 50 times (reference)
(b) ``folds_lodo``           leave-one-date-out
(c) ``folds_blocks``         contiguous date blocks (primary): eight 4-day summer blocks + one winter block
(d) ``folds_winter_to_summer``  train on winter, test on summer

Sensitivity analyses (82 summer samples as test data, winter always in training)
-----------------------------------------------------------------------------
* block lengths 1, 2, 3, 4, 6, 8 days       → ``folds_blocks(block_days=L, include_winter=False)``
* 1- and 2-day buffers around 4-day blocks  → ``buffer_days=k``
* every possible block starting offset      → ``offset=o``
* station exclusion (Table S30)             → ``folds_station_exclusion``

Hyper-parameter tuning and the cross-fitting of SOLACE happen *inside* each
model's ``fit`` with inner folds grouped by survey date; the held-out outer fold
is never passed to ``fit``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import OPTICAL_VARS, SENSOR_VARS, TARGET


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray
    kind: str = "test"
    info: dict = field(default_factory=dict)


def _summer_mask(df: pd.DataFrame) -> np.ndarray:
    return df["campaign"].astype(str).str.startswith("summer").to_numpy()


def summer_block_ids(df: pd.DataFrame, block_days: int = 4, offset: int = 0) -> np.ndarray:
    """Block index of every summer sample (−1 for non-summer). Blocks are calendar-contiguous.

    With ``offset`` o > 0 the first block is ``block_days − o`` days long.
    """
    summer = _summer_mask(df)
    dates = pd.to_datetime(df["date"])
    start = dates[summer].min()
    day = (dates - start).dt.days.to_numpy()
    ids = np.full(len(df), -1)
    ids[summer] = (day[summer] + offset) // block_days
    return ids


def folds_random(df: pd.DataFrame, n_repeats: int = 50, test_size: float = 0.2, n_bins: int = 5,
                 seed: int = 0) -> list[Fold]:
    """Stratified random splitting (strata: campaign × within-campaign ΣPAHs quantile bin)."""
    from sklearn.model_selection import StratifiedShuffleSplit

    camp = df["campaign"].astype(str).reset_index(drop=True)
    y = df[TARGET].reset_index(drop=True)
    bins = y.groupby(camp).transform(lambda s: pd.qcut(s.rank(method="first"), n_bins, labels=False))
    strata = (camp + "_" + bins.astype(int).astype(str)).to_numpy()
    sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
    return [Fold(f"R{i + 1:02d}", tr, te) for i, (tr, te) in enumerate(sss.split(np.zeros(len(df)), strata))]


def folds_lodo(df: pd.DataFrame, summer_only_test: bool = False) -> list[Fold]:
    dates = df["date"].astype(str).to_numpy()
    summer = _summer_mask(df)
    test_dates = np.unique(dates[summer]) if summer_only_test else np.unique(dates)
    out = []
    for d in test_dates:
        te = np.where(dates == d)[0]
        tr = np.where(dates != d)[0]
        out.append(Fold(d, tr, te, kind="summer" if summer[te[0]] else "winter"))
    return out


def folds_blocks(df: pd.DataFrame, block_days: int = 4, offset: int = 0, buffer_days: int = 0,
                 include_winter: bool = True) -> list[Fold]:
    """Contiguous date blocks. Summer blocks S1…; the winter campaign is one block (W).

    ``buffer_days`` removes from training the samples collected within k days before
    or after the test block. When ``include_winter`` the winter block is held out with
    the models trained on summer data only.
    """
    summer = _summer_mask(df)
    ids = summer_block_ids(df, block_days, offset)
    dates = pd.to_datetime(df["date"])
    out = []
    for b in sorted(np.unique(ids[summer])):
        te = np.where(ids == b)[0]
        keep = ids != b
        if buffer_days > 0:
            lo = dates.iloc[te].min() - pd.Timedelta(days=buffer_days)
            hi = dates.iloc[te].max() + pd.Timedelta(days=buffer_days)
            keep &= ~((dates >= lo) & (dates <= hi)).to_numpy()
        d0, d1 = dates.iloc[te].min(), dates.iloc[te].max()
        out.append(Fold(f"S{b + 1}", np.where(keep)[0], te, kind="summer",
                        info={"dates": f"{d0:%d %b}–{d1:%d %b}"}))
    if include_winter and (~summer).any():
        out.append(Fold("W", np.where(summer)[0], np.where(~summer)[0], kind="winter",
                        info={"dates": "winter campaign"}))
    return out


def folds_winter_to_summer(df: pd.DataFrame) -> list[Fold]:
    summer = _summer_mask(df)
    return [Fold("winter->summer", np.where(~summer)[0], np.where(summer)[0], kind="summer")]


def folds_station_exclusion(df: pd.DataFrame, block_days: int = 4, stations=(1, 2, 3, 4, 5)) -> list[Fold]:
    """Train without the test block AND without any sample from the test station (Table S30)."""
    summer = _summer_mask(df)
    ids = summer_block_ids(df, block_days)
    st = df["station"].to_numpy()
    out = []
    for b in sorted(np.unique(ids[summer])):
        for s in stations:
            te = np.where((ids == b) & (st == s))[0]
            if len(te) == 0:
                continue
            tr = np.where((ids != b) & (st != s))[0]
            out.append(Fold(f"S{b + 1}-st{s}", tr, te, kind="summer", info={"station": s, "block": f"S{b + 1}"}))
    return out


# ----------------------------------------------------------------------------- runner
def _fit_predict(name, factory, fold: Fold, df, optical_vars, diagnostics: bool):
    X = df[SENSOR_VARS].to_numpy(float)
    O = df[optical_vars].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    g = df["date"].astype(str).to_numpy()
    tr, te = fold.train, fold.test
    model = factory()
    model.fit(X[tr], y[tr], g[tr], O[tr])
    O_te = O[te] if getattr(model, "requires_optics_at_predict", False) else None
    res = {f"pred_{name}": model.predict(X[te], O_te)}
    if hasattr(model, "predict_interval"):
        iv = model.predict_interval(X[te])
        res[f"lo_{name}"], res[f"hi_{name}"] = iv[:, 0], iv[:, 1]
    base = getattr(model, "model_", model)
    if diagnostics and hasattr(base, "predict_oracle") and getattr(base, "leaf_source", "") == "optical":
        res[f"oracle_{name}"] = base.predict_oracle(X[te], O[te])
        res[f"leafagree_{name}"] = base.leaf_agreement(X[te], O[te])
        res[f"chance_{name}"] = np.full(len(te), base.chance_agreement_)
    # calibration-domain flag: all seven sensors within the training range
    res["indomain"] = np.all((X[te] >= X[tr].min(0)) & (X[te] <= X[tr].max(0)), axis=1)
    return fold, res


def run_cv(factories: dict, df: pd.DataFrame, folds: list[Fold], *, optical_vars=None, n_jobs: int = 1,
           diagnostics: bool = False, verbose: int = 0) -> pd.DataFrame:
    """Fit every model on every fold; return one row per held-out sample and fold.

    ``factories`` maps a model name to a zero-argument callable returning an
    unfitted model with ``fit(X, y, groups, O)`` and ``predict(X, O=None)``.
    """
    optical_vars = OPTICAL_VARS if optical_vars is None else optical_vars
    df = df.reset_index(drop=True)
    jobs = [(name, fac, f) for f in folds for name, fac in factories.items()]
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_predict)(name, fac, f, df, optical_vars, diagnostics) for name, fac, f in jobs)
    frames = {}
    for fold, res in results:
        key = fold.name
        if key not in frames:
            base = df.iloc[fold.test][["sample_id", "date", "station", "campaign"]].copy()
            base["y"] = df.iloc[fold.test][TARGET].to_numpy()
            base.insert(0, "fold", fold.name)
            base.insert(1, "fold_kind", fold.kind)
            frames[key] = base.reset_index(drop=True)
        for k, v in res.items():
            frames[key][k] = v
    return pd.concat([frames[f.name] for f in folds if f.name in frames], ignore_index=True)


def pooled_metrics(oof: pd.DataFrame, models: list[str], mask=None) -> pd.DataFrame:
    from .metrics import summary

    d = oof if mask is None else oof[mask]
    return pd.DataFrame([{"model": m, **summary(d["y"], d[f"pred_{m}"])} for m in models])


def per_split_metrics(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """For the random scheme: one row per split and model."""
    from .metrics import mae, r2

    rows = []
    for f, g in oof.groupby("fold"):
        for m in models:
            rows.append({"fold": f, "model": m, "R2": r2(g["y"], g[f"pred_{m}"]), "MAE": mae(g["y"], g[f"pred_{m}"])})
    return pd.DataFrame(rows)


def leakage_invariance(factory, df: pd.DataFrame, folds: list[Fold], optical_vars=None, seed: int = 0) -> float:
    """Replace ΣPAHs and optical descriptors of every held-out fold with random values and
    return the maximum absolute change of the held-out predictions (expected: 0)."""
    optical_vars = OPTICAL_VARS if optical_vars is None else optical_vars
    rng = np.random.default_rng(seed)
    base = run_cv({"m": factory}, df, folds, optical_vars=optical_vars)
    worst = 0.0
    for f in folds:
        d2 = df.reset_index(drop=True).copy()
        d2.loc[f.test, TARGET] = rng.uniform(1, 500, len(f.test))
        d2.loc[f.test, optical_vars] = rng.normal(size=(len(f.test), len(optical_vars)))
        pert = run_cv({"m": factory}, d2, [f], optical_vars=optical_vars)
        orig = base[base["fold"] == f.name]["pred_m"].to_numpy()
        worst = max(worst, float(np.max(np.abs(orig - pert["pred_m"].to_numpy()))))
    return worst
