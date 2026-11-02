"""Sensor-subset benchmark: 127 non-empty subsets of the 7 sonde variables × 6 algorithms (Fig. 3a).

    python scripts/03_sensor_subset_benchmark.py --config configs/fast.yaml

Each configuration is evaluated under stratified random 80:20 splitting (reference
scheme) and summarised by its mean held-out R² across splits. The analysis only
shows how performance changes with the number of predictors; no subset is selected.

Output results/03_subsets/: subset_benchmark.csv, fig3a_subset_benchmark.png
"""
from _common import Timer, get_config  # noqa: I001

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cascadepahs import plotting
from cascadepahs.features import SENSOR_VARS, TARGET
from cascadepahs.metrics import r2
from cascadepahs.models import SensorOnlyModel
from cascadepahs.pipeline import fixed_params, load_source, results_dir
from cascadepahs.config import grid_from_cfg
from cascadepahs.validation import folds_random


def evaluate(cfg, alg, subset, X, y, groups, folds, tune):
    cols = list(subset)
    scores = []
    for f in folds:
        m = SensorOnlyModel(algorithm=alg, params=fixed_params(cfg, alg), grid=grid_from_cfg(cfg, alg), tune=tune,
                            n_inner=cfg["n_inner"], random_state=cfg["random_state"])
        m.fit(X[f.train][:, cols], y[f.train], groups[f.train])
        scores.append(r2(y[f.test], m.predict(X[f.test][:, cols])))
    return {"algorithm": alg, "subset": "+".join(SENSOR_VARS[c] for c in cols), "subset_size": len(cols),
            "mean_R2": float(np.mean(scores)), "sd_R2": float(np.std(scores))}


def main():
    cfg = get_config(__doc__)
    sb = cfg["subset_benchmark"]
    out = results_dir(cfg, "03_subsets")
    df = load_source(cfg)
    X, y = df[SENSOR_VARS].to_numpy(float), df[TARGET].to_numpy(float)
    groups = df["date"].astype(str).to_numpy()
    folds = folds_random(df, n_repeats=sb["n_splits"], test_size=cfg["validation"]["test_size"],
                         seed=cfg["random_state"])
    subsets = [c for k in range(1, 8) for c in itertools.combinations(range(7), k)]
    print(f"  {len(subsets)} subsets × {len(sb['algorithms'])} algorithms × {len(folds)} splits")
    with Timer("subset benchmark"):
        res = Parallel(n_jobs=cfg["n_jobs"], verbose=0)(
            delayed(evaluate)(cfg, alg, s, X, y, groups, folds, bool(sb.get("tune", False)))
            for alg in sb["algorithms"] for s in subsets)
    res = pd.DataFrame(res)
    res.to_csv(out / "subset_benchmark.csv", index=False)
    summ = res.groupby("subset_size")["mean_R2"].agg(median="median", p90=lambda v: np.percentile(v, 90))
    summ.to_csv(out / "subset_benchmark_summary.csv")
    print(summ.round(3))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotting.subset_benchmark(ax, res)
    plotting.savefig(fig, out / "fig3a_subset_benchmark.png")


if __name__ == "__main__":
    main()
