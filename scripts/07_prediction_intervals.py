"""90 % prediction intervals, calibration domain and screening (Section 2.5, Note S3.6).

    python scripts/07_prediction_intervals.py --config configs/fast.yaml

Final stage fitted as a quantile regression forest (5th/95th percentiles), calibrated
by conformalized quantile regression with date-grouped out-of-fold conformity scores,
recomputed inside the training data of every validation block.

Output: results/07_intervals/
  oof_intervals.csv, interval_coverage.csv, screening.csv, triage.csv
"""
from _common import Timer, get_config  # noqa: I001

import numpy as np
import pandas as pd

from cascadepahs.metrics import coverage, interval_triage, screening
from cascadepahs.pipeline import load_source, results_dir, sensor_only, solace, with_intervals
from cascadepahs.validation import folds_blocks, run_cv


def width_at_coverage(y, lo, hi, target=0.90):
    e = np.maximum(lo - y, y - hi)
    delta = np.quantile(e, target)
    return float(np.mean(hi - lo + 2 * delta))


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "07_intervals")
    df = load_source(cfg)
    facs = {"SORFE": with_intervals(sensor_only(cfg, "RF", quantile=True), cfg),
            "SOLACE": with_intervals(solace(cfg, "RF", quantile=True), cfg)}
    with Timer("blocked validation with conformal quantile-forest intervals"):
        oof = run_cv(facs, df, folds_blocks(df, cfg["validation"]["block_days"]), n_jobs=cfg["n_jobs"])
    oof.to_csv(out / "oof_intervals.csv", index=False)
    summer = oof["fold_kind"] == "summer"

    rows = []
    for m in facs:
        for sub, mask in [("summer", summer), ("winter", ~summer), ("all", np.ones(len(oof), bool))]:
            if mask.sum() == 0:
                continue
            c = coverage(oof.loc[mask, "y"], oof.loc[mask, f"lo_{m}"], oof.loc[mask, f"hi_{m}"])
            rows.append({"model": m, "subset": sub, "n": int(mask.sum()), **c,
                         "share_in_calibration_domain": float(oof.loc[mask, "indomain"].mean())})
        s = oof[summer]
        rows.append({"model": m, "subset": "summer, width at 90 % empirical coverage", "n": int(summer.sum()),
                     "mean_width": width_at_coverage(s["y"].to_numpy(), s[f"lo_{m}"].to_numpy(), s[f"hi_{m}"].to_numpy())})
    cov = pd.DataFrame(rows)
    cov.to_csv(out / "interval_coverage.csv", index=False)
    print(cov.round(3).to_string(index=False))

    ys = oof.loc[summer, "y"]
    thresholds = {"upper tertile": float(np.quantile(ys, 2 / 3)), "summer median": float(np.median(ys))}
    rows, tri = [], []
    for name, thr in thresholds.items():
        for m in facs:
            rows.append({"threshold_name": name, "model": m, **screening(ys, oof.loc[summer, f"pred_{m}"], thr)})
            tri.append({"threshold_name": name, "model": m,
                        **interval_triage(ys, oof.loc[summer, f"lo_{m}"], oof.loc[summer, f"hi_{m}"], thr)})
    pd.DataFrame(rows).to_csv(out / "screening.csv", index=False)
    pd.DataFrame(tri).to_csv(out / "triage.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(tri).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
