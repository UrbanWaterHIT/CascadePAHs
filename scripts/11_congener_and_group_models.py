"""Models for individual congeners and molecular-weight groups (Section 2.3, Note S4.5, Table S16).

    python scripts/11_congener_and_group_models.py --config configs/fast.yaml

Each of the 16 congeners (non-detects at ½·LOD) and the LMW (2–3 rings), MMW (4 rings)
and HMW (5–6 rings) sums is predicted with SOLACE in the same framework and evaluated on
the summer blocks of the primary scheme, for comparison with ΣPAHs.

Output: results/11_congeners/tableS16_congener_group_models.csv
"""
from _common import Timer, get_config  # noqa: I001

import pandas as pd

from cascadepahs import pahs
from cascadepahs.config import resolve_path
from cascadepahs.features import PAH_CONGENERS, PAH_COLS, TARGET
from cascadepahs.metrics import r2
from cascadepahs.pipeline import load_source, results_dir, solace
from cascadepahs.validation import folds_blocks, run_cv


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "11_congeners")
    df = load_source(cfg)
    if not set(PAH_COLS) <= set(df.columns):
        print("  congener columns not found in the paired file – nothing to do")
        return
    lod = pd.read_csv(resolve_path(cfg, "paired").parent / "pah_lod.csv").set_index("congener")["LOD_ng_L"].to_dict()
    conc = pahs.substitute_nondetects(df, lod)
    groups = pahs.group_sums(df, lod)
    targets = {name: conc[c] for c, name, _ in PAH_CONGENERS}
    targets.update({"LMW PAHs (2–3 rings)": groups["LMW"], "MMW PAHs (4 rings)": groups["MMW"],
                    "HMW PAHs (5–6 rings)": groups["HMW"], "ΣPAHs": df[TARGET]})
    folds = [f for f in folds_blocks(df, cfg["validation"]["block_days"]) if f.kind == "summer"]
    rows = []
    for name, values in targets.items():
        with Timer(name):
            d = df.copy()
            d[TARGET] = values.to_numpy()
            o = run_cv({"SOLACE": solace(cfg, "RF")}, d, folds, n_jobs=cfg["n_jobs"])
            rows.append({"target": name, "R2_summer_blocks": r2(o["y"], o["pred_SOLACE"])})
    t = pd.DataFrame(rows)
    t.to_csv(out / "tableS16_congener_group_models.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
