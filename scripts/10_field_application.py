"""Field application and high-frequency reconstruction (Sections 2.5 and 3.5, Fig. 6g, Note S6.2).

    python scripts/10_field_application.py --config configs/fast.yaml

1. SOLACE (quantile final stage + conformal calibration) is refitted on the 142 Dasha
   River samples pooled with the 10 calibration samples of each target site (the
   representative set of script 09 when available) and applied to the sonde readings of
   the application period (22 July–22 August 2024). The predictions are compared with
   the ΣPAHs measured in the same water samples (not used for calibration).
2. The fitted models are applied to the QA/QC-passed high-frequency sonde records:
   Dasha River records → model trained on the 142 Dasha samples; target-site records →
   calibrated models. Each estimate carries a 90 % prediction interval and an
   extrapolation flag (sensor values outside the training range).

Output: results/10_application/
"""
from _common import Timer, get_config, save_json  # noqa: I001

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cascadepahs import plotting, sonde
from cascadepahs.config import resolve_path
from cascadepahs.deploy import reconstruct
from cascadepahs.features import CROSS_SITE_OPTICAL_VARS, OPTICAL_VARS, SENSOR_VARS, TARGET, TARGET_SITES
from cascadepahs.io import load_application, load_hf, load_paired, site_subset, source_domain
from cascadepahs.metrics import coverage, mae, r2
from cascadepahs.models import ConformalIntervalModel
from cascadepahs.pipeline import results_dir, solace


def fit(model, df, optical_vars):
    return model.fit(df[SENSOR_VARS].to_numpy(float), df[TARGET].to_numpy(float), df["date"].astype(str).to_numpy(),
                     df[optical_vars].to_numpy(float))


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "10_application")
    cal_dir = results_dir(cfg, "09_calibration")
    paired = load_paired(resolve_path(cfg, "paired"))
    src = source_domain(paired)
    app = load_application(resolve_path(cfg, "application"))
    rain = pd.read_csv(resolve_path(cfg, "rainfall"), parse_dates=["date"])
    rep_path = cal_dir / "representative_calibration_sets.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    hp_path = cal_dir / "fixed_hyperparameters.json"

    def solace_pi():
        if hp_path.exists():
            hp = json.loads(hp_path.read_text())["SOLACE"]
            base = solace(cfg, "RF", quantile=True)().set_params(
                tune=False, optical_params=hp["optical"], classifier_params=hp["classifier"],
                final_params={k: (tuple(v) if isinstance(v, list) else v) for k, v in hp["final"].items()})
        else:
            base = solace(cfg, "RF", quantile=True)()
        return ConformalIntervalModel(base, alpha=cfg["alpha"], n_folds=cfg["n_inner"])

    models, train_X = {}, {}
    rows, apps = [], []
    rng = np.random.default_rng(cfg["random_state"])
    for site in TARGET_SITES:
        tgt = site_subset(paired, site)
        if site in rep:
            calset = tgt[tgt["sample_id"].isin(rep[site]["sample_ids"])]
        else:                                                  # first sampling day, 10 random samples
            day = sorted(tgt["date"].astype(str).unique())[0]
            pool = tgt[tgt["date"].astype(str) == day]
            calset = pool.iloc[rng.choice(len(pool), 10, replace=False)]
        pooled = pd.concat([src, calset], ignore_index=True)
        with Timer(f"{site}: calibrated SOLACE with 90 % prediction intervals"):
            m = fit(solace_pi(), pooled, CROSS_SITE_OPTICAL_VARS)
        models[site], train_X[site] = m, pooled[SENSOR_VARS].to_numpy(float)
        a = app[app["site"] == site].reset_index(drop=True)
        rec = reconstruct(m, a.assign(timestamp=a["date"]), train_X[site])
        a = pd.concat([a, rec[["SumPAHs_hat", "PI90_low", "PI90_high", "extrapolation"]]], axis=1)
        apps.append(a)
        cv = coverage(a[TARGET], a["PI90_low"], a["PI90_high"])
        rows.append({"site": site, "n": len(a), "R2": r2(a[TARGET], a["SumPAHs_hat"]), "MAE": mae(a[TARGET], a["SumPAHs_hat"]),
                     **cv, "share_extrapolation": float(a["extrapolation"].mean()),
                     "calibration_samples": ";".join(calset["sample_id"])})
    app_pred = pd.concat(apps, ignore_index=True)
    app_pred.to_csv(out / "application_period_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "application_period_performance.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="calibration_samples").round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    plotting.application_timeseries(ax, app_pred, rain)
    ax.set_title("Application period (demo data)" if "demo" in str(resolve_path(cfg, "paired"))
                 else "Application period", fontsize=9)
    plotting.savefig(fig, out / "fig6g_application_period.png")

    with Timer("high-frequency reconstruction"):
        hf = sonde.qaqc(load_hf(resolve_path(cfg, "hf")))
        hf = hf[hf["qc_pass"]].reset_index(drop=True)
        m_src = fit(ConformalIntervalModel(solace(cfg, "RF", quantile=True)(), alpha=cfg["alpha"], n_folds=cfg["n_inner"]),
                    src, OPTICAL_VARS)
        parts = []
        for site, g in hf.groupby("site"):
            m, Xtr = (m_src, src[SENSOR_VARS].to_numpy(float)) if site == "DashaRiver" else (models[site], train_X[site])
            parts.append(reconstruct(m, g.reset_index(drop=True), Xtr))
        hfp = pd.concat(parts, ignore_index=True)
        hfp.to_csv(out / "hf_reconstruction.csv", index=False)
        ex = hfp.groupby("site")["extrapolation"].agg(["size", "mean"]).rename(columns={"size": "records", "mean": "share_extrapolation"})
        ex.to_csv(out / "hf_extrapolation_summary.csv")
        ms = np.mean([p.attrs.get("ms_per_record", np.nan) for p in parts])
        save_json({"records": len(hfp), "ms_per_record": ms, "share_extrapolation": float(hfp["extrapolation"].mean())},
                  out / "hf_reconstruction_summary.json")
        print(ex)
        surveys = hfp["survey_id"].unique()
        fig, axes = plt.subplots(1, len(surveys), figsize=(3.2 * len(surveys), 3.2))
        for ax, sid in zip(np.atleast_1d(axes), surveys):
            sc = plotting.hf_map(ax, hfp[hfp["survey_id"] == sid])
            ax.set_title(sid, fontsize=8)
            ax.tick_params(labelsize=6)
            plt.colorbar(sc, ax=ax, label="ΣPAHs estimate (ng L$^{-1}$)")
        fig.tight_layout()
        plotting.savefig(fig, out / "hf_reconstruction_maps.png")


if __name__ == "__main__":
    main()
