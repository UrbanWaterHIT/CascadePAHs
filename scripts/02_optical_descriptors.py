"""Optical descriptors from raw UV–vis and EEM spectra, and PARAFAC (Notes S1.3 and S2.1).

    python scripts/02_optical_descriptors.py --config configs/fast.yaml

Input  data/demo/raw_spectra/{absorbance_doc.csv, eem_cube.npz}
Output results/02_optics/
  optical_descriptors_from_raw.csv   17 descriptors per sample
  parafac_components.csv             excitation/emission maxima of C1–C3
  parafac_split_half.json            Tucker congruence between split halves
  parafac_components.png             component loadings
"""
from _common import Timer, get_config, save_json  # noqa: I001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cascadepahs import optics, parafac
from cascadepahs.config import resolve_path
from cascadepahs.pipeline import results_dir


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "02_optics")
    raw = resolve_path(cfg, "raw_spectra")
    ab = pd.read_csv(raw / "absorbance_doc.csv")
    wl = np.array([float(c[1:]) for c in ab.columns if c.startswith("A")])
    A = ab[[c for c in ab.columns if c.startswith("A")]].to_numpy(float)
    cube = np.load(raw / "eem_cube.npz", allow_pickle=False)
    ex, em, eem, blank = cube["ex"], cube["em"], cube["eem"].astype(float), cube["blank"].astype(float)

    with Timer("EEM pre-processing (blank, inner filter, Raman units, scatter)"):
        ra = optics.raman_area(blank, ex, em)
        E = np.array([optics.preprocess_eem(eem[i], blank, ex, em, wl, A[i], raman=ra) for i in range(len(eem))])
        print(f"  Raman area of the blank: {ra:.1f}")

    with Timer("PARAFAC (3 components) and split-half validation"):
        model = parafac.nn_parafac(E, 3, n_init=3, seed=cfg["random_state"])
        print(f"  explained variance: {model['explained']:.4f}")
        peaks = pd.DataFrame(parafac.component_peaks(model, ex, em))
        peaks.to_csv(out / "parafac_components.csv", index=False)
        sh = parafac.split_half(E, 3, seed=cfg["random_state"], n_init=2)
        save_json({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sh.items()},
                  out / "parafac_split_half.json")
        print(peaks)
        print("  split-half:", sh)

    with Timer("UV–vis and EEM descriptors"):
        tab = optics.descriptor_table(ab["sample_id"], wl, A, E, ex, em, doc=ab["DOC_mg_L"].to_numpy(), fmax=model["A"])
        tab.to_csv(out / "optical_descriptors_from_raw.csv", index=False)
        true_path = raw / "true_component_scores.csv"
        if true_path.exists():
            t = pd.read_csv(true_path)
            for r in range(3):
                print(f"  C{r + 1}: r(Fmax, simulated score) = {np.corrcoef(tab[f'C{r + 1}'], t[f'C{r + 1}_true'])[0, 1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for r, ax in enumerate(axes):
        ax.contourf(em, ex, np.outer(model["B"][:, r], model["C"][:, r]), levels=20, cmap="viridis")
        ax.set_title(f"C{r + 1}")
        ax.set_xlabel("Emission (nm)")
        ax.set_ylabel("Excitation (nm)")
    fig.tight_layout()
    fig.savefig(out / "parafac_components.png", dpi=200)


if __name__ == "__main__":
    main()
