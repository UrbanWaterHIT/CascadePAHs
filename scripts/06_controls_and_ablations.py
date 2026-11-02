"""Complexity-matched controls, alternative uses of the optical data, ablations and reference models
(Section 2.4, Note S4.3, Tables S17 and S24).

    python scripts/06_controls_and_ablations.py --config configs/fast.yaml

All models are evaluated under the primary blocked validation on the same held-out samples.
ΔR² = R²(SOLACE) − R²(model), with paired date-level bootstrap 95 % CIs.

Output: results/06_controls/
"""
from _common import Timer, get_config  # noqa: I001

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from cascadepahs.features import OPTICAL_GROUPS, OPTICAL_VARS, TARGET
from cascadepahs.metrics import paired_date_bootstrap, r2, summary
from cascadepahs.pipeline import control_factories, load_source, results_dir, solace
from cascadepahs.validation import folds_blocks, run_cv

DEPLOYABLE = {"Measured optics only": False, "Sensors + measured optics": False}


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "06_controls")
    df = load_source(cfg)
    folds = folds_blocks(df, cfg["validation"]["block_days"])
    nb, seed = cfg["validation"]["n_boot"], cfg["random_state"]

    facs = control_factories(cfg)
    with Timer(f"controls and reference models: {len(facs)} models × {len(folds)} folds"):
        oof = run_cv(facs, df, folds, n_jobs=cfg["n_jobs"], diagnostics=True)
    # ablations: remove one optical group from the optical stage of SOLACE
    for grp, cols in OPTICAL_GROUPS.items():
        keep = [c for c in OPTICAL_VARS if c not in cols]
        name = f"SOLACE without {grp}"
        with Timer(f"ablation: {name}"):
            o = run_cv({name: solace(cfg, "RF")}, df, folds, optical_vars=keep, n_jobs=cfg["n_jobs"], diagnostics=True)
            for k in ("pred", "oracle", "leafagree", "chance"):
                oof[f"{k}_{name}"] = o[f"{k}_{name}"].to_numpy()
    oof.to_csv(out / "oof_controls.csv", index=False)

    summer = oof["fold_kind"] == "summer"
    # leaf diagnostics (oracle leaves, agreement) of SOLACE, the shuffled control and the ablations
    diag = []
    for c in [c for c in oof.columns if c.startswith("leafagree_")]:
        m = c[len("leafagree_"):]
        diag.append({"model": m, "summer_leaf_agreement": oof.loc[summer, c].mean(),
                     "summer_chance_agreement": oof.loc[summer, f"chance_{m}"].mean(),
                     "summer_MAE": summary(oof.loc[summer, "y"], oof.loc[summer, f"pred_{m}"])["MAE"],
                     "summer_MAE_oracle_leaves": summary(oof.loc[summer, "y"], oof.loc[summer, f"oracle_{m}"])["MAE"],
                     "winter_leaf_agreement": oof.loc[~summer, c].mean() if (~summer).any() else np.nan})
    pd.DataFrame(diag).to_csv(out / "leaf_diagnostics.csv", index=False)
    print(pd.DataFrame(diag).round(3).to_string(index=False))
    names = [c[5:] for c in oof.columns if c.startswith("pred_")]
    rows = []
    for m in names:
        bs = paired_date_bootstrap(oof[summer], ["pred_SOLACE", f"pred_{m}"] if m != "SOLACE" else ["pred_SOLACE"],
                                   n_boot=nb, seed=seed)
        bp = paired_date_bootstrap(oof, [f"pred_{m}"], n_boot=nb, seed=seed)
        row = {"model": m, "deployable": DEPLOYABLE.get(m, True),
               "summer_R2": bs.iloc[0 if m == "SOLACE" else 1]["estimate"],
               "summer_R2_lo": bs.iloc[0 if m == "SOLACE" else 1]["ci_low"],
               "summer_R2_hi": bs.iloc[0 if m == "SOLACE" else 1]["ci_high"],
               "pooled_R2": bp.iloc[0]["estimate"], "pooled_R2_lo": bp.iloc[0]["ci_low"],
               "pooled_R2_hi": bp.iloc[0]["ci_high"]}
        if m != "SOLACE":
            row.update({"dR2_summer": bs.iloc[-1]["estimate"], "dR2_lo": bs.iloc[-1]["ci_low"],
                        "dR2_hi": bs.iloc[-1]["ci_high"]})
        w = oof[~summer]
        if len(w):
            s = summary(w["y"], w[f"pred_{m}"])
            row.update({"winter_MAE": s["MAE"], "winter_RMSE": s["RMSE"], "winter_bias": s["bias"],
                        "winter_median_pred": s["median_pred"]})
        a = summary(oof["y"], oof[f"pred_{m}"])
        row.update({"all_MAE": a["MAE"]})
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "tableS17_S24_controls.csv", index=False)
    print(t[["model", "summer_R2", "dR2_summer", "dR2_lo", "dR2_hi", "pooled_R2"]].round(3).to_string(index=False))

    # negative control: remove the three least important optical descriptors vs random triplets
    with Timer("negative control: least-important vs random descriptor removals"):
        rf = RandomForestRegressor(n_estimators=300, random_state=seed).fit(df[OPTICAL_VARS], df[TARGET])
        least = list(pd.Series(rf.feature_importances_, OPTICAL_VARS).sort_values().index[:3])
        base = r2(oof.loc[summer, "y"], oof.loc[summer, "pred_SOLACE"])
        rng = np.random.default_rng(seed)
        sets = [("least important", least)] + [
            (f"random {i + 1}", list(rng.choice(OPTICAL_VARS, 3, replace=False)))
            for i in range(cfg["controls"]["negative_control_random_removals"])]
        rows = []
        for label, rem in sets:
            keep = [c for c in OPTICAL_VARS if c not in rem]
            o = run_cv({"m": solace(cfg, "RF")}, df, [f for f in folds if f.kind == "summer"], optical_vars=keep,
                       n_jobs=cfg["n_jobs"])
            rows.append({"set": label, "removed": ",".join(rem), "summer_R2": r2(o["y"], o["pred_m"]),
                         "reduction": base - r2(o["y"], o["pred_m"])})
        nc = pd.DataFrame(rows)
        nc.to_csv(out / "negative_control_descriptor_removal.csv", index=False)
        print(f"  least important {least}: reduction {nc.iloc[0]['reduction']:.3f}; "
              f"median of random removals {nc.iloc[1:]['reduction'].median():.3f}")


if __name__ == "__main__":
    main()
