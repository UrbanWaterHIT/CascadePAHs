"""DOM optical descriptors from UV–vis absorbance and EEM fluorescence (Note S1.3, Table S2).

UV–vis (1-cm quartz cell, 240–600 nm)
-------------------------------------
* a(λ) = 2.303 · A(λ) / L  (Napierian absorption coefficient, m⁻¹; L in m)
* a254, a350
* S275–295, S350–400: spectral slopes (linear fit of ln a(λ) vs λ; Helms et al., 2008)
* SR = S275–295 / S350–400
* SUVA254 = a254 / DOC (L mg C⁻¹ m⁻¹)
  (``suva_decadic=True`` gives the conventional A254/L/DOC instead)

EEM (Ex 200–500 nm, Em 240–600 nm)
----------------------------------
Pre-processing: blank subtraction → inner-filter correction (absorbance-based)
→ Raman normalisation (water Raman peak at Ex 350 nm, Em 371–428 nm) → removal of
first/second-order Rayleigh and Raman scatter → interpolation of the removed bands.

Indices:
* FI  = F(Ex 370, Em 470) / F(Ex 370, Em 520)                     (McKnight et al., 2001)
* BIX = F(Ex 310, Em 380) / F(Ex 310, Em 430)
* β:α = F(Ex 310, Em 380) / max F(Ex 310, Em 420–435)              (Parlanti et al., 2000)
* HIX = ΣF(Ex 254, Em 438–480) / [ΣF(Em 300–345) + ΣF(Em 438–480)]  (Ohno, 2002; ``hix_form='ohno'``)
  or the H/L ratio (Zsolnay et al., 1999; ``hix_form='ratio'``)
* protein-like fluorescence: Ex 220–300 nm / Em 280–380 nm (peaks B and T)
* humic-like fluorescence:   Ex 220–400 nm / Em 380–550 nm (peaks A, C and M)
* TotalF: sum of the PARAFAC component maxima (Fmax)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

PROTEIN_WINDOW = {"ex": (220.0, 300.0), "em": (280.0, 380.0)}      # peaks B and T
HUMIC_WINDOW = {"ex": (220.0, 400.0), "em": (380.0, 550.0)}        # peaks A, C and M


# ============================================================================ UV–vis
def absorption_coefficient(A: np.ndarray, path_m: float = 0.01) -> np.ndarray:
    """Napierian absorption coefficient a(λ) = 2.303 A(λ) / L (m⁻¹)."""
    return 2.303 * np.asarray(A, float) / path_m


def spectral_slope(wl: np.ndarray, a: np.ndarray, lo: float, hi: float) -> float:
    """Spectral slope (nm⁻¹) from a linear fit of ln a(λ) against λ over [lo, hi]."""
    wl = np.asarray(wl, float)
    m = (wl >= lo) & (wl <= hi)
    aa = np.asarray(a, float)[m]
    ok = aa > 0
    if ok.sum() < 3:
        return np.nan
    slope = np.polyfit(wl[m][ok], np.log(aa[ok]), 1)[0]
    return float(-slope)


def uv_descriptors(wl, A, doc=None, path_m: float = 0.01, baseline_nm: tuple | None = (590, 600),
                   suva_decadic: bool = False) -> dict:
    """UV–vis descriptors of one spectrum (decadic absorbance A)."""
    wl, A = np.asarray(wl, float), np.asarray(A, float)
    if baseline_nm is not None:                                  # null-point correction
        m = (wl >= baseline_nm[0]) & (wl <= baseline_nm[1])
        A = A - A[m].mean()
    a = absorption_coefficient(A, path_m)
    out = {
        "a254": float(np.interp(254, wl, a)),
        "a350": float(np.interp(350, wl, a)),
        "S275_295": spectral_slope(wl, a, 275, 295),
        "S350_400": spectral_slope(wl, a, 350, 400),
    }
    out["SR"] = out["S275_295"] / out["S350_400"]
    if doc is not None:
        num = np.interp(254, wl, A) / path_m if suva_decadic else out["a254"]
        out["SUVA254"] = float(num / doc)
    return out


# ============================================================================ EEM pre-processing
def _interp_spectrum(x_new, x, y):
    return np.interp(x_new, x, y, left=y[0], right=y[-1])


def inner_filter_correction(eem: np.ndarray, ex, em, wl, A) -> np.ndarray:
    """F_corr = F_obs · 10^((A_ex + A_em)/2), absorbance for a 1-cm cell."""
    Aex = _interp_spectrum(ex, wl, A)
    Aem = _interp_spectrum(em, wl, A)
    return eem * 10 ** ((Aex[:, None] + Aem[None, :]) / 2)


def raman_area(blank: np.ndarray, ex, em, ex_raman: float = 350.0, em_range=(371.0, 428.0)) -> float:
    """Integrated water Raman peak (a.u.·nm) of the blank at Ex 350 nm (baseline-corrected)."""
    ex, em = np.asarray(ex, float), np.asarray(em, float)
    row = np.array([np.interp(ex_raman, ex, blank[:, j]) for j in range(len(em))])
    m = (em >= em_range[0]) & (em <= em_range[1])
    x, y = em[m], row[m]
    base = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    return float(np.trapezoid(y - base, x)) if hasattr(np, "trapezoid") else float(np.trapz(y - base, x))


def scatter_mask(ex, em, rayleigh1: float = 15.0, rayleigh2: float = 15.0, raman1: float = 12.0,
                 below_rayleigh: bool = True) -> np.ndarray:
    """Boolean mask (True = remove) of first/second-order Rayleigh and Raman scatter bands."""
    E, M = np.meshgrid(np.asarray(ex, float), np.asarray(em, float), indexing="ij")
    raman_em = 1.0 / (1.0 / E - 3400e-7)
    mask = (np.abs(M - E) <= rayleigh1) | (np.abs(M - 2 * E) <= rayleigh2) | (np.abs(M - raman_em) <= raman1)
    if below_rayleigh:
        mask |= M < E - rayleigh1                     # emission shorter than excitation (no fluorescence)
    return mask


def remove_scatter(eem: np.ndarray, ex, em, interpolate: bool = True, **mask_kw) -> np.ndarray:
    """Set scatter bands to NaN, then (optionally) fill them by interpolation along emission.

    Values at emission < excitation are set to zero (physically no fluorescence).
    """
    E, M = np.meshgrid(np.asarray(ex, float), np.asarray(em, float), indexing="ij")
    mask = scatter_mask(ex, em, **mask_kw)
    out = np.where(mask, np.nan, eem).astype(float)
    if not interpolate:
        return out
    below = M < E - mask_kw.get("rayleigh1", 15.0)
    out[below] = 0.0
    em = np.asarray(em, float)
    for i in range(out.shape[0]):
        row = out[i]
        bad = np.isnan(row)
        if bad.all():
            out[i] = 0.0
        elif bad.any():
            out[i, bad] = np.interp(em[bad], em[~bad], row[~bad])
    return np.clip(out, 0, None)


def preprocess_eem(raw: np.ndarray, blank: np.ndarray, ex, em, wl=None, A=None, raman: float | None = None,
                   interpolate: bool = True, **mask_kw) -> np.ndarray:
    """Blank subtraction, inner-filter correction, Raman normalisation (R.U.) and scatter removal."""
    x = np.asarray(raw, float) - np.asarray(blank, float)
    if wl is not None and A is not None:
        x = inner_filter_correction(x, ex, em, wl, A)
    ra = raman_area(blank, ex, em) if raman is None else raman
    x = x / ra
    return remove_scatter(x, ex, em, interpolate=interpolate, **mask_kw)


# ============================================================================ EEM indices
def eem_value(eem, ex, em, ex0: float, em0: float) -> float:
    f = RegularGridInterpolator((np.asarray(ex, float), np.asarray(em, float)), eem, bounds_error=False,
                                fill_value=np.nan)
    return float(f([[ex0, em0]])[0])


def emission_scan(eem, ex, em, ex0: float, em_grid) -> np.ndarray:
    f = RegularGridInterpolator((np.asarray(ex, float), np.asarray(em, float)), eem, bounds_error=False,
                                fill_value=np.nan)
    em_grid = np.asarray(em_grid, float)
    return f(np.column_stack([np.full(len(em_grid), ex0), em_grid]))


def fluorescence_index(eem, ex, em) -> float:
    return eem_value(eem, ex, em, 370, 470) / eem_value(eem, ex, em, 370, 520)


def biological_index(eem, ex, em) -> float:
    return eem_value(eem, ex, em, 310, 380) / eem_value(eem, ex, em, 310, 430)


def freshness_index(eem, ex, em) -> float:
    """β:α = F(310/380) / max F(310, 420–435)."""
    alpha = np.nanmax(emission_scan(eem, ex, em, 310, np.arange(420, 436, 1.0)))
    return eem_value(eem, ex, em, 310, 380) / alpha


def humification_index(eem, ex, em, form: str = "ohno", high=(438.0, 480.0), low=(300.0, 345.0)) -> float:
    grid_h = np.arange(high[0], high[1] + 0.5, 1.0)
    grid_l = np.arange(low[0], low[1] + 0.5, 1.0)
    H = np.nansum(emission_scan(eem, ex, em, 254, grid_h))
    L = np.nansum(emission_scan(eem, ex, em, 254, grid_l))
    return float(H / (H + L)) if form == "ohno" else float(H / L)


def regional_integral(eem, ex, em, window: dict) -> float:
    """Integral of the EEM (R.U.·nm²) over an Ex/Em window."""
    ex, em = np.asarray(ex, float), np.asarray(em, float)
    mi = (ex >= window["ex"][0]) & (ex <= window["ex"][1])
    mj = (em >= window["em"][0]) & (em <= window["em"][1])
    sub = np.nan_to_num(eem[np.ix_(mi, mj)])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(trap(sub, em[mj], axis=1), ex[mi]))


def regional_mean(eem, ex, em, window: dict) -> float:
    """Mean intensity (R.U.) over an Ex/Em window."""
    ex, em = np.asarray(ex, float), np.asarray(em, float)
    mi = (ex >= window["ex"][0]) & (ex <= window["ex"][1])
    mj = (em >= window["em"][0]) & (em <= window["em"][1])
    return float(np.nanmean(eem[np.ix_(mi, mj)]))


def eem_descriptors(eem, ex, em, *, hix_form: str = "ohno", protein_window=PROTEIN_WINDOW,
                    humic_window=HUMIC_WINDOW, regional: str = "mean") -> dict:
    """Index-type EEM descriptors of one Raman-normalised, scatter-free EEM."""
    reg = regional_mean if regional == "mean" else regional_integral
    return {
        "FI": fluorescence_index(eem, ex, em),
        "BIX": biological_index(eem, ex, em),
        "BetaAlpha": freshness_index(eem, ex, em),
        "HIX": humification_index(eem, ex, em, form=hix_form),
        "ProteinF": reg(eem, ex, em, protein_window),
        "HumicF": reg(eem, ex, em, humic_window),
    }


def descriptor_table(sample_ids, wl, A, eems, ex, em, doc=None, fmax: np.ndarray | None = None,
                     **kw) -> pd.DataFrame:
    """Combine UV–vis, EEM and (optionally) PARAFAC Fmax into the 17-descriptor table."""
    rows = []
    for i, sid in enumerate(sample_ids):
        row = {"sample_id": sid}
        row.update(uv_descriptors(wl, A[i], None if doc is None else doc[i]))
        row.update(eem_descriptors(eems[i], ex, em, **kw))
        if fmax is not None:
            for r in range(fmax.shape[1]):
                row[f"C{r + 1}"] = float(fmax[i, r])
            row["TotalF"] = float(fmax[i].sum())
        if doc is not None:
            row["DOC"] = float(doc[i])
        rows.append(row)
    return pd.DataFrame(rows)
