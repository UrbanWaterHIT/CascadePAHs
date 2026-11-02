"""SHAP attributions for the models in the cascade and predictability of the optical descriptors (Fig. 5).

    python scripts/08_shap_interpretation.py --config configs/fast.yaml

a  first-stage optical forest (17 descriptors), TreeExplainer
b  SORFE (7 sonde variables), TreeExplainer
c  complete SOLACE deployment pipeline (7 sonde variables), permutation explainer
d  R² of predicting each optical descriptor from the 7 sonde variables (blocked validation)

Held-out samples of the summer blocks; background = training samples of each fold.
Output: results/08_shap/  (complete SHAP matrices as CSV + fig5_shap.png)
"""
from _common import Timer, get_config  # noqa: I001

import matplotlib.pyplot as plt

from cascadepahs import plotting
from cascadepahs.features import OPTICAL_LABELS, OPTICAL_VARS, SENSOR_VARS
from cascadepahs.interpret import HAS_SHAP, descriptor_predictability, mean_abs_shap, shap_over_folds
from cascadepahs.pipeline import load_source, results_dir, sensor_only, solace
from cascadepahs.validation import folds_blocks

SENSOR_SHORT = {"T": "T", "pH": "pH", "DO": "DO", "Sal": "Salinity", "Turb": "Turbidity", "Chla": "Chl-a",
                "FDOM": "FDOM"}


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "08_shap")
    df = load_source(cfg)
    folds = folds_blocks(df, cfg["validation"]["block_days"])
    print(f"  shap package available: {HAS_SHAP} (otherwise exact / sampled interventional Shapley values)")
    with Timer("SHAP over the summer blocks"):
        res = shap_over_folds(df, folds, sorfe_factory=sensor_only(cfg, "RF"), solace_factory=solace(cfg, "RF"),
                              optical_vars=OPTICAL_VARS, max_background=cfg["shap"]["max_background"],
                              seed=cfg["random_state"])
    for k, t in res.items():
        t.to_csv(out / f"shap_{k}.csv", index=False)
        print(f"  mean |SHAP| {k}:", mean_abs_shap(t).round(2).to_dict())
    with Timer("sensor-to-optical predictability"):
        pr = descriptor_predictability(df, folds, OPTICAL_VARS, {"n_estimators": cfg["fixed_params"]["RF"]["n_estimators"]},
                                       n_jobs=cfg.get("model_n_jobs", 1))
        pr.to_csv(out / "descriptor_predictability.csv", index=False)
        print(pr.round(3).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    plotting.shap_beeswarm(axes[0, 0], res["optical"], OPTICAL_VARS, OPTICAL_LABELS)
    plotting.shap_beeswarm(axes[0, 1], res["sorfe"], SENSOR_VARS, SENSOR_SHORT)
    plotting.shap_beeswarm(axes[1, 0], res["solace"], SENSOR_VARS, SENSOR_SHORT)
    lim = max(abs(res["sorfe"].filter(like="shap_").values).max(), abs(res["solace"].filter(like="shap_").values).max()) * 1.05
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xlim(-lim, lim)
    plotting.descriptor_r2_bars(axes[1, 1], pr, OPTICAL_LABELS)
    for ax, p in zip(axes.ravel(), "abcd"):
        ax.set_title(p, loc="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.62, 0.012, 0.25])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.set_label("Feature value")
    plotting.savefig(fig, out / "fig5_shap.png")


if __name__ == "__main__":
    main()
