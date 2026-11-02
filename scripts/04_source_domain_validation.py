"""Source-domain (Dasha River) validation of the sensor-only and privileged-information models.

    python scripts/04_source_domain_validation.py --config configs/fast.yaml

* primary scheme: eight 4-day summer blocks + one winter block
  → Table 1 (6 algorithms × {sensor-only, -P}), Figs. 3b–g and 4, Table S18 (block MAE,
    leaf agreement, oracle leaves), Table S24 rows for SORFE/SOLACE, paired date-level
    bootstrap CIs and Wilcoxon / sign tests
* schemes (a) random 80:20, (b) leave-one-date-out, (d) winter→summer → Table S19
* autocorrelation of daily mean SORFE residuals under leave-one-date-out (block-length criterion)

Output: results/04_source_validation/
"""
from _common import Timer, get_config  # noqa: I001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cascadepahs import plotting, stats
from cascadepahs.features import OPTICAL_VARS, SENSOR_VARS, TARGET
from cascadepahs.metrics import blockwise, paired_block_tests, paired_date_bootstrap, summary
from cascadepahs.models import ALGORITHMS
from cascadepahs.pipeline import load_source, results_dir, sensor_only, solace
from cascadepahs.validation import (folds_blocks, folds_lodo, folds_random, folds_winter_to_summer, per_split_metrics, run_cv)

PANEL = "bcdefg"
ORDER = ["RF", "SVR", "XGBoost", "LR", "DNN", "MLP"]


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "04_source_validation")
    df = load_source(cfg)
    v = cfg["validation"]
    nb = v["n_boot"]

    # ------------------------------------------------------------------ primary scheme, 12 models
    facs = {}
    for alg in ALGORITHMS:
        facs[alg] = sensor_only(cfg, alg)
        facs[f"{alg}-P"] = solace(cfg, alg)
    folds = folds_blocks(df, block_days=v["block_days"])
    with Timer(f"primary blocked validation: {len(facs)} models × {len(folds)} folds"):
        oof = run_cv(facs, df, folds, n_jobs=cfg["n_jobs"], diagnostics=True)
    oof.to_csv(out / "oof_primary_scheme.csv", index=False)
    summer = oof["fold_kind"] == "summer"

    rows = []
    for alg in ORDER:
        for name, label in [(alg, alg if alg != "RF" else "RF (SORFE)"),
                            (f"{alg}-P", f"{alg}-P" if alg != "RF" else "RF-P (SOLACE)")]:
            s = summary(oof.loc[summer, "y"], oof.loc[summer, f"pred_{name}"])
            a = summary(oof["y"], oof[f"pred_{name}"])
            rows.append({"model": label, "summer_R2": s["R2"], "summer_MAE": s["MAE"], "pooled_R2": a["R2"],
                         "pooled_MAE": a["MAE"]})
    table1 = pd.DataFrame(rows)
    table1.to_csv(out / "table1_primary_scheme.csv", index=False)
    print(table1.round(3).to_string(index=False))

    # ------------------------------------------------------------------ bootstrap CIs SOLACE vs SORFE
    o2 = oof.rename(columns={"pred_RF-P": "SOLACE", "pred_RF": "SORFE"})
    boot_s = paired_date_bootstrap(o2[summer], ["SOLACE", "SORFE"], n_boot=nb, seed=cfg["random_state"])
    boot_s.insert(0, "subset", "summer")
    boot_a = paired_date_bootstrap(o2, ["SOLACE", "SORFE"], n_boot=nb, seed=cfg["random_state"])
    boot_a.insert(0, "subset", "pooled")
    from cascadepahs.metrics import mae
    boot_m = paired_date_bootstrap(o2[summer], ["SORFE", "SOLACE"], n_boot=nb, seed=cfg["random_state"], metric=mae)
    boot_m.insert(0, "subset", "summer_MAE")
    pd.concat([boot_s, boot_a, boot_m]).to_csv(out / "bootstrap_ci_solace_vs_sorfe.csv", index=False)
    print(boot_s.round(3).to_string(index=False))

    # ------------------------------------------------------------------ Table S18
    o3 = oof.assign(SORFE=oof["pred_RF"], SOLACE=oof["pred_RF-P"], Oracle=oof["oracle_RF-P"])
    bt = blockwise(o3, ["SORFE", "SOLACE", "Oracle"])
    agree = oof.groupby("fold", sort=False).agg(leaf_agreement=("leafagree_RF-P", "mean"),
                                                chance_agreement=("chance_RF-P", "mean")).reset_index(drop=True)
    bt = pd.concat([bt, agree], axis=1)
    bt["dMAE"] = bt["MAE_SORFE"] - bt["MAE_SOLACE"]
    s_rows = o3[summer]
    bt.loc[len(bt)] = {"block": "All summer", "n": int(summer.sum()),
                       "MAE_SORFE": summary(s_rows["y"], s_rows["SORFE"])["MAE"],
                       "MAE_SOLACE": summary(s_rows["y"], s_rows["SOLACE"])["MAE"],
                       "MAE_Oracle": summary(s_rows["y"], s_rows["Oracle"])["MAE"],
                       "leaf_agreement": s_rows["leafagree_RF-P"].mean(),
                       "chance_agreement": s_rows["chance_RF-P"].mean(),
                       "dMAE": summary(s_rows["y"], s_rows["SORFE"])["MAE"] - summary(s_rows["y"], s_rows["SOLACE"])["MAE"]}
    bt.to_csv(out / "tableS18_blocks_leaf_agreement.csv", index=False)
    tests = paired_block_tests(blockwise(o3[summer], ["SORFE", "SOLACE"]), "SOLACE", "SORFE")
    pd.DataFrame([tests]).to_csv(out / "block_tests.csv", index=False)
    possible = bt.iloc[-1]["MAE_SORFE"] - bt.iloc[-1]["MAE_Oracle"]          # reduction possible with oracle leaves
    frac = (bt.iloc[-1]["MAE_SORFE"] - bt.iloc[-1]["MAE_SOLACE"]) / possible if possible > 0.5 else float("nan")
    print(bt.round(3).to_string(index=False))
    print(f"  share of the oracle MAE reduction achieved by SOLACE: {100 * frac:.0f} % "
          f"(undefined when oracle leaves do not reduce the MAE);  block tests: {tests}")

    # ------------------------------------------------------------------ winter errors (Table S24 rows)
    w = oof[~summer]
    if len(w):
        rows = []
        drop = w["y"].idxmax()
        for name, lab in [("RF", "SORFE"), ("RF-P", "SOLACE")]:
            s = summary(w["y"], w[f"pred_{name}"])
            s2 = summary(w.drop(index=drop)["y"], w.drop(index=drop)[f"pred_{name}"])
            rows.append({"model": lab, "winter_MAE": s["MAE"], "winter_RMSE": s["RMSE"], "winter_bias": s["bias"],
                         "winter_median_prediction": s["median_pred"], "winter_MAE_without_max": s2["MAE"],
                         "share_in_calibration_domain": float(w["indomain"].mean())})
        pd.DataFrame(rows).to_csv(out / "winter_block_errors.csv", index=False)

    # ------------------------------------------------------------------ schemes a, b, d (Table S19)
    two = {"SORFE": sensor_only(cfg, "RF"), "SOLACE": solace(cfg, "RF")}
    rows = []
    with Timer("scheme (a) stratified random 80:20"):
        fr = folds_random(df, n_repeats=v["n_random_splits"], test_size=v["test_size"], seed=cfg["random_state"])
        oa = run_cv(two, df, fr, n_jobs=cfg["n_jobs"])
        ps = per_split_metrics(oa, list(two))
        wide = ps.pivot(index="fold", columns="model", values="R2")
        for mname in two:
            g = ps[ps["model"] == mname]
            rows.append({"scheme": "(a) random 80:20", "model": mname, "R2": g["R2"].median(),
                         "R2_lo": g["R2"].quantile(.025), "R2_hi": g["R2"].quantile(.975), "MAE": g["MAE"].median()})
        dd = wide["SOLACE"] - wide["SORFE"]
        rows.append({"scheme": "(a) random 80:20", "model": "dR2", "R2": dd.median(), "R2_lo": dd.quantile(.025),
                     "R2_hi": dd.quantile(.975)})
    for label, flist in [("(b) leave-one-date-out", folds_lodo(df)), ("(d) train winter, test summer",
                                                                      folds_winter_to_summer(df))]:
        with Timer(f"scheme {label}"):
            o = run_cv(two, df, flist, n_jobs=cfg["n_jobs"])
            o.to_csv(out / f"oof_{label.split()[0].strip('()')}.csv", index=False)
            b = paired_date_bootstrap(o, ["pred_SOLACE", "pred_SORFE"], n_boot=nb, seed=cfg["random_state"])
            for mname in two:
                s = summary(o["y"], o[f"pred_{mname}"])
                r_ = b[b["model"] == f"pred_{mname}"].iloc[0]
                rows.append({"scheme": label, "model": mname, "R2": s["R2"], "R2_lo": r_["ci_low"],
                             "R2_hi": r_["ci_high"], "MAE": s["MAE"]})
            r_ = b.iloc[-1]
            rows.append({"scheme": label, "model": "dR2", "R2": r_["estimate"], "R2_lo": r_["ci_low"],
                         "R2_hi": r_["ci_high"]})
            if label.startswith("(b)"):
                sm = o[o["campaign"].str.startswith("summer")]
                res = (sm["y"] - sm["pred_SORFE"]).groupby(sm["date"]).mean()
                stats.residual_acf(res, 5).to_csv(out / "sorfe_lodo_residual_acf.csv", index=False)
    for mname, col in [("SORFE", "pred_RF"), ("SOLACE", "pred_RF-P")]:
        for sub, mask in [("(c) blocks, summer", summer), ("(c) blocks, pooled", np.ones(len(oof), bool))]:
            s = summary(oof.loc[mask, "y"], oof.loc[mask, col])
            rows.append({"scheme": sub, "model": mname, "R2": s["R2"], "MAE": s["MAE"]})
    pd.DataFrame(rows).to_csv(out / "tableS19_validation_schemes.csv", index=False)

    # ------------------------------------------------------------------ figures 3b–g and 4
    X, O, y = df[SENSOR_VARS].to_numpy(float), df[OPTICAL_VARS].to_numpy(float), df[TARGET].to_numpy(float)
    g = df["date"].astype(str).to_numpy()
    smask_df = df["campaign"].str.startswith("summer").to_numpy()
    lim = float(np.percentile(oof.loc[summer, "y"], 100)) * 1.15
    for suffix, fig_name, ttl in [("", "fig3_sensor_only_scatter.png", ""), ("-P", "fig4_privileged_scatter.png", "-P")]:
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for ax, alg, pnl in zip(axes.ravel(), ORDER, PANEL):
            name = alg + suffix
            m = facs[name]().fit(X, y, g, O)                             # training fit, for the "Train" points
            lab = {"RF": "RF (SORFE)", "RF-P": "RF-P (SOLACE)"}.get(name, name)
            plotting.measured_vs_predicted(ax, oof.loc[summer, "y"], oof.loc[summer, f"pred_{name}"], label=lab,
                                           y_train=y[smask_df], p_train=m.predict(X[smask_df]), lim=lim)
            ax.set_title(pnl, loc="left", fontweight="bold")
        h, lab = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        plotting.savefig(fig, out / fig_name)
    t4 = pd.DataFrame({"algorithm": ORDER,
                       "R2_sensor": [summary(oof.loc[summer, "y"], oof.loc[summer, f"pred_{a}"])["R2"] for a in ORDER],
                       "R2_P": [summary(oof.loc[summer, "y"], oof.loc[summer, f"pred_{a}-P"])["R2"] for a in ORDER],
                       "MAE_sensor": [summary(oof.loc[summer, "y"], oof.loc[summer, f"pred_{a}"])["MAE"] for a in ORDER],
                       "MAE_P": [summary(oof.loc[summer, "y"], oof.loc[summer, f"pred_{a}-P"])["MAE"] for a in ORDER]})
    fig, ax = plt.subplots(figsize=(8, 4))
    plotting.r2_mae_by_algorithm(ax, t4)
    plotting.savefig(fig, out / "fig4a_r2_mae_by_algorithm.png")


if __name__ == "__main__":
    main()
