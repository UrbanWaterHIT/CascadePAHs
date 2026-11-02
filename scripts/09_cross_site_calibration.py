"""Direct transfer and few-shot calibration at the Yangyong River and Shenzhen Bay
(Sections 2.5 and 3.5, Note S6, Tables S21, S25, S26, Figs. 6a–f, S12–S15).

    python scripts/09_cross_site_calibration.py --config configs/fast.yaml

Architecture and hyper-parameters of SOLACE/SORFE are fixed from the source domain;
in every draw both models are refitted on the 142 Dasha River samples pooled with the
same 10 calibration samples (14 optical descriptors; PARAFAC components excluded).

Output: results/09_calibration/
"""
from _common import Timer, get_config, save_json  # noqa: I001

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone

from cascadepahs import calibration as cal
from cascadepahs import plotting
from cascadepahs.config import resolve_path
from cascadepahs.features import CROSS_SITE_OPTICAL_VARS, OPTICAL_VARS, SENSOR_VARS, TARGET, TARGET_SITES
from cascadepahs.io import load_paired, site_subset, source_domain
from cascadepahs.metrics import describe_draws
from cascadepahs.models import ConformalIntervalModel
from cascadepahs.pipeline import results_dir, sensor_only, solace

NAMES = {"YangyongRiver": "Yangyong River", "ShenzhenBay": "Shenzhen Bay"}


def frozen(factory, df, optical_vars):
    m = factory().fit(df[SENSOR_VARS].to_numpy(float), df[TARGET].to_numpy(float), df["date"].astype(str).to_numpy(),
                      df[optical_vars].to_numpy(float))
    return m


def main():
    cfg = get_config(__doc__)
    c = cfg["calibration"]
    out = results_dir(cfg, "09_calibration")
    paired = load_paired(resolve_path(cfg, "paired"))
    src = source_domain(paired)
    seed, nj = cfg["random_state"], cfg["n_jobs"]

    with Timer("source-domain models (direct transfer; fixed hyper-parameters for calibration)"):
        src_sorfe = frozen(sensor_only(cfg, "RF"), src, OPTICAL_VARS)
        src_solace = frozen(solace(cfg, "RF"), src, OPTICAL_VARS)
        fx_solace = frozen(solace(cfg, "RF"), src, CROSS_SITE_OPTICAL_VARS)     # 14 descriptors
        # Unfitted prototypes with the tuned hyper-parameters fixed. The factories close over these
        # small objects only, so parallel workers never receive pickled fitted forests.
        proto_sorfe, proto_solace = src_sorfe.frozen_clone(), fx_solace.frozen_clone()
        alpha, n_inner = cfg["alpha"], cfg["n_inner"]
        fixed = {"SORFE": lambda: clone(proto_sorfe), "SOLACE": lambda: clone(proto_solace)}
        fixed_pi = {"SORFE": lambda: clone(proto_sorfe),
                    "SOLACE": lambda: ConformalIntervalModel(clone(proto_solace).set_params(quantile=True),
                                                             alpha=alpha, n_folds=n_inner)}
        save_json({"SOLACE": {"optical": fx_solace.optical_params_, "classifier": fx_solace.classifier_params_,
                              "final": fx_solace.final_params_}, "SORFE": src_sorfe.best_params_},
                  out / "fixed_hyperparameters.json")

    all_A, all_B, rep_sets, spear = [], [], {}, []
    for site in TARGET_SITES:
        tgt = site_subset(paired, site)
        lab = NAMES[site]
        dt, errs = cal.direct_transfer({"SORFE": src_sorfe, "SOLACE": src_solace}, tgt)
        dt.insert(0, "site", lab)
        dt.to_csv(out / f"direct_transfer_{site}.csv", index=False)
        errs.to_csv(out / f"direct_transfer_abs_errors_{site}.csv", index=False)
        print(dt.round(3).to_string(index=False))

        with Timer(f"{lab}: day-separated calibration ({c['n_draws']} draws per day)"):
            draws, preds = cal.fewshot_calibration(fixed, src, tgt, n_cal=c["n_cal"], n_draws=c["n_draws"], mode="day",
                                                   seed=seed, n_jobs=nj, importance_for="SOLACE",
                                                   importance_repeats=c["importance_repeats"])
        with Timer(f"{lab}: random-selection reference with prediction intervals"):
            rdraws, _ = cal.fewshot_calibration(fixed_pi, src, tgt, n_cal=c["n_cal"], n_draws=c["n_draws"],
                                                mode="random", seed=seed + 1, n_jobs=nj, keep_predictions=False)
        both = pd.concat([draws, rdraws], ignore_index=True)
        both.insert(0, "site", lab)
        both.to_csv(out / f"draws_{site}.csv", index=False)
        preds.to_csv(out / f"predictions_{site}.csv", index=False)
        A, B = cal.summarize_calibration(both.drop(columns="site"))
        A.insert(0, "site", lab)
        B.insert(0, "site", lab)
        all_A.append(A)
        all_B.append(B)
        spear.append({"site": lab, **cal.range_coverage_spearman(rdraws)})

        # representative draw (Fig. 6a–d) and its calibration set (used in script 10)
        setting, d = cal.representative_draw(draws)
        sel = draws[(draws["setting"] == setting) & (draws["draw"] == d) & (draws["model"] == "SOLACE")].iloc[0]
        rep_sets[site] = {"setting": setting, "draw": d, "sample_ids": sel["cal_ids"].split(";")}
        imp = draws[draws["model"] == "SOLACE"][[f"imp_{v}" for v in SENSOR_VARS]]
        imp_t = pd.DataFrame([{"feature": v, **describe_draws(imp[f"imp_{v}"])} for v in SENSOR_VARS])
        imp_t.to_csv(out / f"permutation_importance_{site}.csv", index=False)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        med = A[A["setting"] != "Random"].pivot(index="setting", columns="model", values="R2_median")
        for ax, mname, col in [(axes[0], "SOLACE", plotting.ROSE), (axes[1], "SORFE", plotting.BLUE)]:
            pr = preds[(preds["setting"] == setting) & (preds["draw"] == d) & (preds["model"] == mname)]
            plotting.measured_vs_predicted(ax, pr["y"], pr["pred"], label=mname, prefix="")
            ax.text(0.03, 0.80, "median R² by day:\n" + ", ".join(f"{s.split()[-1]}: {v:.2f}" for s, v in med[mname].items()),
                    transform=ax.transAxes, fontsize=7, va="top")
            ax.set_title(f"{lab}: {mname} ({setting}, draw {d})", fontsize=9)
        plotting.importance_bars(axes[2], imp_t.assign(feature=[{"Sal": "Salinity", "Turb": "Turbidity", "Chla": "Chl-a"}.get(v, v)
                                                                for v in imp_t["feature"]]))
        axes[2].set_title("Permutation importance (calibrated SOLACE)", fontsize=9)
        fig.tight_layout()
        plotting.savefig(fig, out / f"fig6_calibration_{site}.png")

        with Timer(f"{lab}: calibration-size curve (Fig. S15)"):
            curve = cal.calibration_size_curve(fixed["SOLACE"], src_solace, src, tgt, sizes=c["sizes"],
                                               n_draws=c["n_draws_size"], seed=seed + 2, n_jobs=nj)
            curve.insert(0, "site", lab)
            curve.to_csv(out / f"calibration_size_curve_{site}.csv", index=False)
            cs = curve.groupby(["method", "n_cal"])["R2"].median().unstack(0)
            print(cs.round(3))
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for mth, col in zip(cs.columns, [plotting.ROSE, "#4F9D69", plotting.BLUE]):
                ax.plot(cs.index, cs[mth], "-o", color=col, label=mth)
            ax.axhline(dt.loc[dt["model"] == "SOLACE", "R2"].iloc[0], ls="--", color=plotting.GREY, label="no calibration")
            ax.set_ylim(max(-1, min(-0.2, cs.values.min() - 0.1)), 1)
            ax.set_xlabel("Number of calibration samples")
            ax.set_ylabel("Median validation $R^2$")
            ax.set_title(lab)
            ax.legend(frameon=False, fontsize=7)
            plotting.savefig(fig, out / f"figS15_size_curve_{site}.png")

    A = pd.concat(all_A, ignore_index=True)
    B = pd.concat(all_B, ignore_index=True)
    A.to_csv(out / "tableS26A_performance.csv", index=False)
    B.to_csv(out / "tableS26B_paired_difference.csv", index=False)
    A[A["setting"] == "Random"].to_csv(out / "tableS25_random_reference.csv", index=False)
    pd.DataFrame(spear).to_csv(out / "range_coverage_spearman.csv", index=False)
    save_json(rep_sets, out / "representative_calibration_sets.json")
    print(A[["site", "setting", "model", "R2_median", "R2_q1", "R2_q3", "MAE_median"]].round(3).to_string(index=False))
    print(B.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    plotting.calibration_intervals(ax, A[A["setting"] != "Random"].assign(site_setting=lambda t: t["site"] + " / " + t["setting"]))
    plotting.savefig(fig, out / "figS14_single_day_calibration.png")


if __name__ == "__main__":
    main()
