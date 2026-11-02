"""Dissolved ΣPAHs and composition from congener concentrations (Section 2.3).

* non-detects (NaN or values below the LOD) are set to 0.5 × LOD;
* optional recovery correction (concentration / recovery fraction);
* ΣPAHs = sum of the 16 US EPA priority PAHs;
* molecular-weight groups: LMW (2–3 rings), MMW (4 rings), HMW (5–6 rings);
* comparison with EU environmental quality standards (Directive 2013/39/EU).
"""
from __future__ import annotations

import pandas as pd

from .features import PAH_CONGENERS, PAH_COLS

RINGS = {c: r for c, _, r in PAH_CONGENERS}
MW_GROUPS = {
    "LMW": [c for c in PAH_COLS if RINGS[c] <= 3],
    "MMW": [c for c in PAH_COLS if RINGS[c] == 4],
    "HMW": [c for c in PAH_COLS if RINGS[c] >= 5],
}

# EU EQS (ng L⁻¹): AA = annual average; MAC = maximum allowable concentration
# (MAC for benzo[a]pyrene: 27 ng L⁻¹ in transitional and coastal waters, 270 ng L⁻¹ inland)
EU_EQS = {
    "Fla": {"AA": 6.3, "MAC": 120.0},
    "BaP": {"AA": 0.17, "MAC_other": 27.0, "MAC_inland": 270.0},
}


def substitute_nondetects(df: pd.DataFrame, lod: dict[str, float], factor: float = 0.5) -> pd.DataFrame:
    out = df[PAH_COLS].astype(float).copy()
    for c in PAH_COLS:
        nd = out[c].isna() | (out[c] < lod[c])
        out.loc[nd, c] = factor * lod[c]
    return out


def recovery_correct(df: pd.DataFrame, recovery: dict[str, float] | float) -> pd.DataFrame:
    """Divide concentrations by recovery fractions (e.g. 0.91 for 91 %)."""
    out = df[PAH_COLS].astype(float).copy()
    for c in PAH_COLS:
        r = recovery[c] if isinstance(recovery, dict) else recovery
        out[c] = out[c] / r
    return out


def sum_pahs(df: pd.DataFrame, lod: dict[str, float], factor: float = 0.5) -> pd.Series:
    """Dissolved ΣPAHs (ng L⁻¹) with non-detects at ``factor`` × LOD."""
    return substitute_nondetects(df, lod, factor).sum(axis=1).round(3)


def group_sums(df: pd.DataFrame, lod: dict[str, float]) -> pd.DataFrame:
    c = substitute_nondetects(df, lod)
    out = pd.DataFrame({g: c[cols].sum(axis=1) for g, cols in MW_GROUPS.items()})
    total = out.sum(axis=1)
    for g in MW_GROUPS:
        out[f"{g}_share"] = out[g] / total
    return out


def detection_frequency(df: pd.DataFrame, lod: dict[str, float], by: str | None = None) -> pd.DataFrame:
    det = pd.DataFrame({c: (df[c].notna() & (df[c] >= lod[c])) for c in PAH_COLS})
    if by is None:
        return det.agg(["sum", "count", "mean"]).T
    det[by] = df[by].values
    return det.groupby(by).mean().T


def eqs_exceedance(df: pd.DataFrame, lod: dict[str, float]) -> pd.DataFrame:
    """Share of samples exceeding the EU AA-EQS / MAC-EQS for fluoranthene and benzo[a]pyrene."""
    rows = []
    for c, lim in EU_EQS.items():
        x = df[c]
        det = x.notna() & (x >= lod[c])
        for k, v in lim.items():
            rows.append({"compound": c, "standard": k, "limit_ng_L": v, "n": len(x), "detected": int(det.sum()),
                         "exceed_n": int((x[det] > v).sum()), "exceed_share": float((x[det] > v).sum() / len(x))})
    return pd.DataFrame(rows)
