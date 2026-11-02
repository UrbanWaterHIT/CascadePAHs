"""Sensitivity analyses of the source-domain comparison (Note S3.5, Tables S20 and S30).

    python scripts/05_sensitivity_analyses.py --config configs/fast.yaml

* summer test data (82 samples), winter always in training:
  block lengths 1 (LODO), 2, 3, 4, 6, 8 days; 4-day blocks + 1/2-day buffers;
  every possible block starting offset
* station exclusion: test station removed from training (Table S30)
* random-seed replicates of SOLACE (LODO vs 4-day blocks)
* leakage-invariance test (held-out ΣPAHs and optical data replaced by random values)

Output: results/05_sensitivity/
"""
from _common import Timer, get_config  # noqa: I001

import numpy as np
import pandas as pd

from cascadepahs.metrics import paired_date_bootstrap, r2, summary
from cascadepahs.pipeline import load_source, results_dir, sensor_only, solace
from cascadepahs.validation import folds_blocks, folds_lodo, folds_station_exclusion, leakage_invariance, run_cv


def compare(two, df, folds, cfg, label):
    o = run_cv(two, df, folds, n_jobs=cfg["n_jobs"])
    b = paired_date_bootstrap(o, ["pred_SOLACE", "pred_SORFE"], n_boot=cfg["validation"]["n_boot"],
                              seed=cfg["random_state"])
    s1, s2 = summary(o["y"], o["pred_SORFE"]), summary(o["y"], o["pred_SOLACE"])
    d = b.iloc[-1]
    return o, {"scheme": label, "test_groups": len(folds), "mean_training_n": float(np.mean([len(f.train) for f in folds])),
               "R2_SORFE": s1["R2"], "R2_SOLACE": s2["R2"], "dR2": d["estimate"], "dR2_lo": d["ci_low"],
               "dR2_hi": d["ci_high"], "MAE_SORFE": s1["MAE"], "MAE_SOLACE": s2["MAE"]}


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "05_sensitivity")
    df = load_source(cfg)
    v = cfg["validation"]
    two = {"SORFE": sensor_only(cfg, "RF"), "SOLACE": solace(cfg, "RF")}

    rows = []
    with Timer("block lengths and buffers (summer test data)"):
        for L in v["block_lengths"]:
            folds = folds_lodo(df, summer_only_test=True) if L == 1 else folds_blocks(df, L, include_winter=False)
            label = "Leave-one-date-out" if L == 1 else f"{L}-day blocks"
            _, r = compare(two, df, folds, cfg, label)
            rows.append(r)
            print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
        for bfr in v["buffers"]:
            _, r = compare(two, df, folds_blocks(df, v["block_days"], buffer_days=bfr, include_winter=False), cfg,
                           f"{v['block_days']}-day blocks + {bfr}-day buffer")
            rows.append(r)
    pd.DataFrame(rows).to_csv(out / "tableS20_block_lengths_buffers.csv", index=False)

    if v.get("offsets", False):
        with Timer("all block starting offsets"):
            rows = []
            for L in [x for x in v["block_lengths"] if x > 1]:
                for off in range(L):
                    _, r = compare(two, df, folds_blocks(df, L, offset=off, include_winter=False), cfg,
                                   f"{L}-day blocks, offset {off}")
                    rows.append({**r, "block_days": L, "offset": off})
            offs = pd.DataFrame(rows)
            offs.to_csv(out / "block_offsets.csv", index=False)
            offs.groupby("block_days")[["R2_SORFE", "R2_SOLACE", "dR2"]].agg(["mean", "min", "max"]).to_csv(
                out / "block_offsets_summary.csv")

    with Timer("station exclusion (Table S30)"):
        o, r = compare(two, df, folds_station_exclusion(df, v["block_days"]), cfg, "station excluded")
        st = []
        for s, g in o.groupby("station"):
            st.append({"station": s, "n": len(g), "R2_SORFE": r2(g["y"], g["pred_SORFE"]),
                       "R2_SOLACE": r2(g["y"], g["pred_SOLACE"])})
        st = pd.DataFrame(st)
        st["dR2"] = st["R2_SOLACE"] - st["R2_SORFE"]
        st.loc[len(st)] = {"station": "pooled", "n": len(o), "R2_SORFE": r["R2_SORFE"], "R2_SOLACE": r["R2_SOLACE"],
                           "dR2": r["dR2"]}
        st.to_csv(out / "tableS30_station_exclusion.csv", index=False)
        print(st.round(3).to_string(index=False))

    with Timer("random-seed replicates of SOLACE"):
        reps = []
        for seed in range(v["seeds_replicates"]):
            fac = {"SOLACE": solace(cfg, "RF", seed=seed)}
            r_l = r2(*run_cv(fac, df, folds_lodo(df, summer_only_test=True), n_jobs=cfg["n_jobs"])[["y", "pred_SOLACE"]].T.values)
            r_b = r2(*run_cv(fac, df, folds_blocks(df, v["block_days"], include_winter=False),
                             n_jobs=cfg["n_jobs"])[["y", "pred_SOLACE"]].T.values)
            reps.append({"seed": seed, "R2_LODO": r_l, "R2_4day_blocks": r_b})
        reps = pd.DataFrame(reps)
        reps.to_csv(out / "seed_replicates.csv", index=False)
        print(reps.describe().loc[["mean", "std"]].round(3))

    with Timer("leakage-invariance test"):
        worst = leakage_invariance(solace(cfg, "RF"), df, folds_blocks(df, v["block_days"]), seed=cfg["random_state"])
        pd.DataFrame([{"max_abs_prediction_change": worst}]).to_csv(out / "leakage_invariance.csv", index=False)
        print(f"  max |Δ prediction| after randomising held-out ΣPAHs and optics: {worst:.3e}")


if __name__ == "__main__":
    main()
