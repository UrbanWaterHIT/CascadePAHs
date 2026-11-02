"""Write the demo data to data/demo/.

    python scripts/00_demo_data.py [--config configs/fast.yaml]

The demo values are simulated (not measurements). To analyse the measured data,
use the Supplementary Data in the same format (data/README.md) and skip this step.
"""
from _common import ROOT, Timer, get_config  # noqa: I001

import numpy as np
import pandas as pd

from cascadepahs.config import resolve_path
from cascadepahs.demo_data import PAH_LOD, make_application, make_hf, make_paired, make_raw_spectra



def write_csv(df: pd.DataFrame, path, header: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(df)} rows)")


def main():
    cfg = get_config(__doc__)
    seed = cfg["random_state"]
    with Timer("paired samples, on-station readings, rainfall"):
        paired, onstation, rain = make_paired(seed=2024)
        write_csv(paired, resolve_path(cfg, "paired"))
        onstation["timestamp"] = onstation["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        write_csv(onstation.round(4), resolve_path(cfg, "onstation"))
        rain_out = rain.assign(date=rain["date"].dt.strftime("%Y-%m-%d"))
        write_csv(rain_out, resolve_path(cfg, "rainfall"))
    with Timer("application-period samples"):
        write_csv(make_application(rain, seed=7), resolve_path(cfg, "application"))
    with Timer("high-frequency sonde records"):
        hf = make_hf(rain, seed=11)
        hf["timestamp"] = hf["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        write_csv(hf, resolve_path(cfg, "hf"))
    with Timer("raw UV–vis and EEM spectra"):
        d = make_raw_spectra(n=30, seed=5)
        out = resolve_path(cfg, "raw_spectra")
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "eem_cube.npz", ex=d["ex"], em=d["em"], eem=d["eem"].astype(np.float32),
                            blank=d["blank"].astype(np.float32), sample_id=d["sample_id"])
        absb = pd.DataFrame(d["absorbance"].round(6), columns=[f"A{int(w)}" for w in d["wl"]])
        absb.insert(0, "sample_id", d["sample_id"])
        rng = np.random.default_rng(seed)
        absb.insert(1, "DOC_mg_L", (2.0 + 1.6 * d["true_scores"][:, 1] + rng.normal(0, 0.3, len(absb))).round(3))
        write_csv(absb, out / "absorbance_doc.csv")
        pd.DataFrame(d["true_scores"], columns=["C1_true", "C2_true", "C3_true"]).assign(
            sample_id=d["sample_id"]).to_csv(out / "true_component_scores.csv", index=False)
    pd.DataFrame({"congener": list(PAH_LOD), "LOD_ng_L": list(PAH_LOD.values())}).to_csv(
        ROOT / "data" / "demo" / "pah_lod.csv", index=False)
    (ROOT / "data" / "demo" / "README.txt").write_text(
        "Demo data for running the code. The values are simulated (not measurements) and do not\n"
        "reproduce the results of the paper. Measured data: Supplementary Data of the article.\n"
        "For data requests, please contact congchaozhang@stu.hit.edu.cn.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
