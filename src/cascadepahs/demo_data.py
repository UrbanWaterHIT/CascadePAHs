"""Demo data for running the code.

The values are simulated (not measurements). They follow the sampling design of the
study (Tables S12–S13) and the approximate variable ranges (Tables S3–S5, S23).
The measured data are provided as Supplementary Data with the article.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OPTICAL_VARS, PAH_CONGENERS, SENSOR_VARS

SUMMER_START = pd.Timestamp("2024-07-22")
SUMMER_DATES = pd.date_range("2024-07-22", "2024-08-22", freq="D")        # 32 days
WINTER_DATES = pd.date_range("2025-01-14", "2025-01-18", freq="D")        # 5 days
YANGYONG_DATES = pd.to_datetime(["2024-05-21", "2024-06-04", "2024-06-18"])
SHENZHEN_DATES = pd.to_datetime(["2024-05-14", "2024-05-28", "2024-06-11", "2024-06-25"])
SUMMER_PER_STATION = {1: 15, 2: 17, 3: 8, 4: 20, 5: 22}                     # Table S13 (82 samples)

# Approximate station coordinates (Fig. 1; WGS 84)
DASHA_STATIONS = {
    1: (22.52194, 113.95056), 2: (22.52083, 113.95111), 3: (22.51944, 113.95167), 4: (22.52139, 113.95250),
    5: (22.51972, 113.95000), 6: (22.52111, 113.95028), 7: (22.51833, 113.95000), 8: (22.52167, 113.95139),
    9: (22.52000, 113.95111), 10: (22.51889, 113.95111), 11: (22.52056, 113.95222), 12: (22.52111, 113.95333),
}
SHENZHEN_STATIONS = {
    1: (22.4975, 113.9405), 2: (22.4985, 113.9725), 3: (22.4985, 114.0010), 4: (22.5050, 114.0255),
    5: (22.4960, 114.0450), 6: (22.4880, 114.0220), 7: (22.4830, 113.9950), 8: (22.4800, 113.9680),
    9: (22.4790, 113.9300), 10: (22.4660, 113.9420), 11: (22.4810, 114.0550), 12: (22.4760, 114.0280),
    13: (22.4700, 113.9950), 14: (22.4590, 113.9720), 15: (22.4460, 113.9455), 16: (22.4340, 113.9235),
}
YANGYONG_STATIONS = {i: (22.80265 - 0.0007 * (i % 2), 113.87415 + 0.00045 * (i - 1)) for i in range(1, 15)}

# LOD (ng L⁻¹) per congener, within the 0.06–0.17 range reported in Table S14
PAH_LOD = dict(zip([c for c, _, _ in PAH_CONGENERS],
                   [0.17, 0.12, 0.10, 0.08, 0.06, 0.07, 0.06, 0.06, 0.09, 0.08, 0.10, 0.09, 0.06, 0.12, 0.15, 0.11]))


# ---------------------------------------------------------------------------
# daily drivers
# ---------------------------------------------------------------------------
def _rainfall(rng) -> pd.DataFrame:
    n = len(SUMMER_DATES)
    base = rng.exponential(1.2, n) * (rng.random(n) < 0.45)
    dasha = base.copy()
    for d, v in {"2024-07-27": 35.2, "2024-07-28": 11.0, "2024-07-31": 48.6, "2024-08-01": 9.4,
                 "2024-08-14": 24.8, "2024-08-17": 21.6}.items():
        dasha[SUMMER_DATES.get_loc(pd.Timestamp(d))] = v
    target = base * 1.4
    for d, v in {"2024-07-26": 4.0, "2024-07-27": 70.0, "2024-07-28": 27.0, "2024-07-30": 6.5, "2024-07-31": 9.0,
                 "2024-08-11": 3.0, "2024-08-12": 18.0, "2024-08-14": 68.0, "2024-08-15": 21.0,
                 "2024-08-17": 50.0, "2024-08-18": 18.0, "2024-08-19": 7.0, "2024-08-21": 5.5}.items():
        target[SUMMER_DATES.get_loc(pd.Timestamp(d))] = v
    return pd.DataFrame({"date": SUMMER_DATES, "rain_dasha_mm": dasha.round(1), "rain_target_mm": target.round(1)})


def _runoff_index(rain: np.ndarray, weights=(0.55, 1.0, 0.55, 0.25, 0.1), scale=35.0) -> np.ndarray:
    r = np.zeros_like(rain, dtype=float)
    for k, w in enumerate(weights):
        r[k:] += w * rain[: len(rain) - k]
    return np.log1p(r / scale * 3.0)


def _ar1(rng, n, phi=0.6, sd=0.5):
    x = np.zeros(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2))
    return x


def _summer_schedule(rng) -> list[tuple[pd.Timestamp, int]]:
    """Analysed (date, station) pairs: per-station counts of Table S13, ≥1 sample per day."""
    days = list(SUMMER_DATES)
    quota = dict(SUMMER_PER_STATION)
    chosen: set[tuple[int, int]] = set()
    order = rng.permutation(len(days))
    for di in order:                                             # one sample on every day
        st = rng.choice(list(quota), p=np.array(list(quota.values())) / sum(quota.values()))
        chosen.add((di, st))
        quota[st] -= 1
    for st, q in quota.items():                                  # fill the remaining quota
        free = [di for di in range(len(days)) if (di, st) not in chosen]
        w = np.array([0.35 if days[di].strftime("%m-%d") in {"07-27", "07-28", "07-31", "08-01"} else 1.0
                      for di in free])
        for di in rng.choice(free, size=q, replace=False, p=w / w.sum()):
            chosen.add((di, st))
    return sorted((days[di], st) for di, st in chosen)


# ---------------------------------------------------------------------------
# latent model -> sensors, optics, ΣPAHs
# ---------------------------------------------------------------------------
def _site_params(site: str, campaign: str) -> dict:
    if site == "DashaRiver" and campaign == "summer2024":
        return dict(T=29.0, Tsd=1.1, Sal=6.2, Salk=5.5, FDOM=23.5, FDOMk=7.0, Turb=20.0, Chla=5.5, DO=6.6, pH=7.92,
                    DOC=6.9, DOCk=3.0, C1=1.32, C2=2.45, C3=0.41, FI=1.52, BIX=0.85, HIX=0.60, a254=10.0, a350=1.99,
                    S275=0.0197, S350=0.0183, y=72.0, ylo=25.7, yhi=209.4, ysd=0.30, bdom=0.50, baro=0.10,
                    bfresh=0.05, bsal=0.0, bnl=0.60, sensor_noise=0.5)
    if site == "DashaRiver":                                    # winter 2025
        return dict(T=17.5, Tsd=1.0, Sal=4.5, Salk=4.0, FDOM=24.0, FDOMk=7.5, Turb=21.0, Chla=6.0, DO=7.4, pH=7.88,
                    DOC=7.0, DOCk=3.2, C1=1.35, C2=2.50, C3=0.42, FI=1.50, BIX=0.84, HIX=0.61, a254=10.2, a350=2.0,
                    S275=0.0196, S350=0.0182, y=36.0, ylo=28.8, yhi=60.0, ysd=0.05, bdom=0.05, baro=0.03,
                    bfresh=0.02, bsal=-0.01, bnl=0.0, sensor_noise=0.5)
    if site == "YangyongRiver":
        return dict(T=21.6, Tsd=0.6, Sal=0.0, Salk=0.0, FDOM=13.3, FDOMk=0.9, Turb=5.0, Chla=1.2, DO=6.4, pH=7.74,
                    DOC=12.5, DOCk=3.0, C1=1.71, C2=2.06, C3=2.94, FI=1.97, BIX=0.98, HIX=0.61, a254=17.6, a350=3.42,
                    S275=0.0209, S350=0.0275, y=95.0, ylo=40.9, yhi=402.0, ysd=0.22, bdom=0.55, baro=0.15,
                    bfresh=0.05, bsal=0.0, bnl=0.45, sensor_noise=0.6)
    if site == "ShenzhenBay":
        return dict(T=26.9, Tsd=2.5, Sal=12.0, Salk=5.5, FDOM=19.3, FDOMk=5.5, Turb=14.0, Chla=7.0, DO=6.4, pH=8.15,
                    DOC=3.8, DOCk=1.0, C1=1.17, C2=0.97, C3=0.38, FI=1.29, BIX=0.82, HIX=0.73, a254=12.0, a350=3.0,
                    S275=0.0189, S350=0.0150, y=22.0, ylo=12.1, yhi=55.4, ysd=0.18, bdom=0.35, baro=0.10,
                    bfresh=0.05, bsal=-0.08, bnl=0.30, sensor_noise=0.6)
    raise ValueError(site)


def _simulate(rng, site, campaign, run, drift, bio, tide, station_grad, n=None):
    """Simulate sensors, optics and ΣPAHs for vectors of daily drivers (one value per row)."""
    p = _site_params(site, campaign)
    n = len(run)
    e = rng.normal(size=(n, 12))
    u_dom = 0.75 * run + 0.45 * drift + 0.25 * station_grad + 0.35 * e[:, 0]
    u_aro = 0.45 * u_dom + 0.85 * e[:, 1]
    u_fresh = 0.5 * e[:, 2] + 0.4 * bio
    u_bio = bio + 0.5 * e[:, 3]
    u_sal = -0.7 * run + tide - 0.6 * station_grad + 0.3 * e[:, 4]

    sn = p.get("sensor_noise", 1.0)
    e[:, 5:12] *= sn
    s = {}
    s["T"] = p["T"] + p["Tsd"] * (0.6 * drift - 0.5 * run) + 0.3 * e[:, 5]
    s["Sal"] = np.clip(p["Sal"] + p["Salk"] * u_sal + (0.8 if p["Salk"] else 0.0) * e[:, 6], 0.0 if site == "YangyongRiver" else 0.2, 28.3)
    if site == "YangyongRiver":                                 # freshwater: 0.00 PSU (Table S5)
        s["Sal"] = np.zeros(n)
    s["FDOM"] = np.clip(p["FDOM"] + p["FDOMk"] * (u_dom + 0.3 * u_aro) + 0.45 * p["FDOMk"] * e[:, 7], 1.0, 45.0)
    s["Turb"] = np.clip(np.exp(np.log(p["Turb"]) + 0.45 * run + 0.25 * u_dom + 0.45 * e[:, 8]), 1.5, 480.0)
    s["Chla"] = np.clip(np.exp(np.log(p["Chla"]) + 0.55 * u_bio + 0.15 * u_dom + 0.35 * e[:, 9]), 0.05, 49.0)
    s["DO"] = np.clip(p["DO"] - 0.45 * u_dom + 0.45 * u_bio + 0.3 * e[:, 10], 4.4, 11.7)
    s["pH"] = p["pH"] + 0.07 * u_bio - 0.03 * u_dom + 0.04 * e[:, 11]
    s["Cond"] = np.clip(650 + 1000 * s["Sal"] + 60 * rng.normal(size=n), 600, None)
    if site == "YangyongRiver":
        s["Cond"] = 750 + 25 * u_dom + 20 * rng.normal(size=n)
    s["TDS"] = 0.65 * s["Cond"] + 5 * rng.normal(size=n)

    f = rng.normal(size=(n, 14))
    o = {}
    o["DOC"] = np.clip(p["DOC"] + p["DOCk"] * (u_dom + 0.25 * u_aro) + 0.15 * p["DOCk"] * f[:, 0], 0.5, None)
    o["C2"] = np.clip(p["C2"] * np.exp(0.5 * u_dom + 0.15 * u_fresh) + 0.06 * p["C2"] * f[:, 1], 0.05, None)
    o["C1"] = np.clip(p["C1"] * (1 + 0.30 * u_aro + 0.05 * u_dom) + 0.04 * p["C1"] * f[:, 2], 0.05, None)
    o["C3"] = np.clip(p["C3"] * (1 + 0.45 * u_fresh + 0.15 * u_dom) + 0.08 * p["C3"] * f[:, 3], 0.01, None)
    o["TotalF"] = o["C1"] + o["C2"] + o["C3"]                    # sum of PARAFAC Fmax
    o["ProteinF"] = 0.93 * (o["C2"] + o["C3"]) * (1 + 0.03 * f[:, 4])
    o["HumicF"] = 1.05 * o["C1"] * (1 + 0.03 * f[:, 5])
    o["FI"] = p["FI"] + 0.10 * u_fresh + 0.03 * u_dom + 0.02 * f[:, 6]
    o["BIX"] = p["BIX"] + 0.045 * u_fresh + 0.010 * f[:, 7]
    o["BetaAlpha"] = 0.975 * o["BIX"] + 0.008 * f[:, 8]
    o["HIX"] = np.clip(p["HIX"] + 0.07 * u_aro - 0.03 * u_fresh + 0.012 * f[:, 9], 0.3, 0.9)
    o["a254"] = np.clip(p["a254"] * (1 + 0.24 * u_dom + 0.09 * u_aro) + 0.025 * p["a254"] * f[:, 10], 1.0, None)
    o["a350"] = np.clip(p["a350"] * (1 + 0.27 * u_dom + 0.15 * u_aro) + 0.04 * p["a350"] * f[:, 11], 0.2, None)
    o["S275_295"] = p["S275"] * (1 - 0.095 * u_aro + 0.03 * u_fresh) + 0.0003 * f[:, 12]
    o["S350_400"] = p["S350"] * (1 - 0.085 * u_aro) + 0.0005 * f[:, 13]
    o["SR"] = o["S275_295"] / o["S350_400"]
    o["SUVA254"] = o["a254"] / o["DOC"]

    logy = (np.log(p["y"]) + p["bdom"] * u_dom + p["baro"] * u_aro + p["bfresh"] * u_fresh + p["bsal"] * u_sal
            + p["bnl"] * np.clip(u_dom - 0.8, 0, None) + p["ysd"] * rng.normal(size=n))
    y = np.clip(np.exp(logy), p["ylo"], p["yhi"])
    return pd.DataFrame(s), pd.DataFrame(o), y


def _congeners(rng, total: np.ndarray, site: str, campaign: str) -> pd.DataFrame:
    """Split ΣPAHs into 16 congeners with site/season-specific ring composition (Fig. S6)."""
    rings = np.array([r for _, _, r in PAH_CONGENERS])
    grp = np.where(rings <= 3, 0, np.where(rings == 4, 1, 2))
    share = {"DashaRiver/summer2024": (0.30, 0.25, 0.45), "DashaRiver/winter2025": (0.46, 0.27, 0.27),
             "YangyongRiver/calib2024": (0.25, 0.275, 0.475), "ShenzhenBay/calib2024": (0.436, 0.30, 0.264)}
    g = np.array(share[f"{site}/{campaign}"])
    within = np.array([0.30, 0.04, 0.08, 0.16, 0.32, 0.10,      # LMW (2–3 rings)
                       0.32, 0.28, 0.18, 0.22,                  # MMW (4 rings)
                       0.26, 0.16, 0.18, 0.15, 0.05, 0.20])     # HMW (5–6 rings)
    for k in range(3):
        within[grp == k] /= within[grp == k].sum()
    mean_frac = g[grp] * within
    comp = rng.dirichlet(mean_frac * 80, size=len(total))
    vals = comp * total[:, None]
    df = pd.DataFrame(vals, columns=[c for c, _, _ in PAH_CONGENERS])
    for c, lod in PAH_LOD.items():
        nd = df[c] < lod * rng.uniform(1.0, 3.0, len(df))        # non-detects near the LOD
        df.loc[nd & (rng.random(len(df)) < 0.7), c] = np.nan
    return df


def _onstation(rng, sensors: pd.DataFrame, meta: pd.DataFrame, riverine: bool) -> pd.DataFrame:
    """Raw sonde readings recorded on station during sampling (two deployments for riverine samples)."""
    rows = []
    sd = {"T": 0.08, "pH": 0.01, "DO": 0.06, "Sal": 0.15, "Turb": 1.2, "Chla": 0.35, "FDOM": 0.6,
          "Cond": 40.0, "TDS": 26.0}
    for i in range(len(sensors)):
        k = rng.integers(14, 30) if riverine else rng.integers(5, 11)
        t0 = pd.Timestamp(meta["date"].iloc[i]) + pd.Timedelta(hours=9) + pd.Timedelta(minutes=int(rng.integers(0, 300)))
        minutes = np.sort(rng.uniform(0, rng.uniform(10, 30) if riverine else 3, k))
        for j in range(k):
            r = {"sample_id": meta["sample_id"].iloc[i], "site": meta["site"].iloc[i], "station": meta["station"].iloc[i],
                 "timestamp": t0 + pd.Timedelta(minutes=float(minutes[j])),
                 "deployment": 1 + int(riverine and j >= k // 2), "depth_m": 0.5 + 0.05 * rng.normal()}
            for v in sd:
                scale = 0.0 if (v == "Sal" and sensors["Sal"].iloc[i] == 0) else 1.0
                r[v] = sensors[v].iloc[i] + sd[v] * rng.normal() * scale
            rows.append(r)
    out = pd.DataFrame(rows)
    out["Sal"] = out["Sal"].clip(lower=0)
    out["Turb"] = out["Turb"].clip(lower=0.1)
    out["Chla"] = out["Chla"].clip(lower=0.01)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def make_paired(seed: int = 2024):
    """Return (paired_samples, onstation_readings, rainfall)."""
    rng = np.random.default_rng(seed)
    rain = _rainfall(rng)
    run_s = _runoff_index(rain["rain_dasha_mm"].to_numpy())
    drift_s, bio_s, tide_s = _ar1(rng, 32, 0.6, 0.55), _ar1(rng, 32, 0.5, 0.6), _ar1(rng, 32, 0.3, 0.5)

    blocks = []
    # Dasha River summer (82 samples)
    sched = _summer_schedule(rng)
    di = np.array([SUMMER_DATES.get_loc(d) for d, _ in sched])
    st = np.array([s for _, s in sched])
    meta = pd.DataFrame({"site": "DashaRiver", "campaign": "summer2024", "date": [d for d, _ in sched], "station": st})
    blocks.append((meta, run_s[di], drift_s[di], bio_s[di], tide_s[di], (st - 3) / 2.0))
    # Dasha River winter (12 stations x 5 days)
    wd = np.repeat(np.arange(5), 12)
    wst = np.tile(np.arange(1, 13), 5)
    dr_w, bio_w, tide_w = _ar1(rng, 5, 0.3, 0.25), _ar1(rng, 5, 0.3, 0.4), _ar1(rng, 5, 0.3, 0.6)
    meta = pd.DataFrame({"site": "DashaRiver", "campaign": "winter2025", "date": WINTER_DATES[wd], "station": wst})
    blocks.append((meta, np.zeros(60) - 0.2, dr_w[wd], bio_w[wd], tide_w[wd], (wst - 6.5) / 5.5))
    # Yangyong River (14 stations x 3 days)
    yd = np.repeat(np.arange(3), 14)
    yst = np.tile(np.arange(1, 15), 3)
    run_y = np.array([0.15, 1.35, 0.5])[yd]
    meta = pd.DataFrame({"site": "YangyongRiver", "campaign": "calib2024", "date": YANGYONG_DATES[yd], "station": yst})
    blocks.append((meta, run_y, np.array([0.2, -0.3, 0.1])[yd], rng.normal(0, .4, 3)[yd], np.zeros(42),
                   (yst - 7.5) / 6.5))
    # Shenzhen Bay (16 stations x 4 days, 6 samples lost)
    missing = {0: [11], 1: [12], 2: [13, 14], 3: [15, 16]}
    rows = [(d, s) for d in range(4) for s in range(1, 17) if s not in missing[d]]
    sd_, sst = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    meta = pd.DataFrame({"site": "ShenzhenBay", "campaign": "calib2024", "date": SHENZHEN_DATES[sd_], "station": sst})
    blocks.append((meta, np.array([0.3, 0.05, 0.6, 0.2])[sd_], np.array([0.1, -0.2, 0.3, -0.1])[sd_],
                   rng.normal(0, .5, 4)[sd_], rng.normal(0, .3, 4)[sd_], (sst - 8.5) / 7.5))

    out, onst = [], []
    for meta, run, drift, bio, tide, grad in blocks:
        site, camp = meta["site"].iloc[0], meta["campaign"].iloc[0]
        sens, opt, y = _simulate(rng, site, camp, run, drift, bio, tide, grad)
        if site == "DashaRiver" and camp == "winter2025":          # one high winter sample (station 8, 17 Jan)
            k = np.where((meta["station"] == 8) & (meta["date"] == pd.Timestamp("2025-01-17")))[0][0]
            y[k] = 226.1
        cong = _congeners(rng, y, site, camp)
        coords = {"DashaRiver": DASHA_STATIONS, "ShenzhenBay": SHENZHEN_STATIONS, "YangyongRiver": YANGYONG_STATIONS}[site]
        meta = meta.copy()
        meta["lat"] = [coords[s][0] for s in meta["station"]]
        meta["lon"] = [coords[s][1] for s in meta["station"]]
        df = pd.concat([meta.reset_index(drop=True), sens, opt, cong], axis=1)
        df["_y_true"] = y
        out.append(df)
    paired = pd.concat(out, ignore_index=True)
    paired = paired.sort_values(["site", "date", "station"]).reset_index(drop=True)
    paired.insert(0, "sample_id", [f"{s[:2].upper()}-{d:%Y%m%d}-S{st:02d}" for s, d, st in
                                   zip(paired["site"], paired["date"], paired["station"])])

    # on-station readings; the paired sensor vector is their mean (Section 2.2)
    for site in paired["site"].unique():
        sub = paired[paired["site"] == site]
        onst.append(_onstation(rng, sub, sub, riverine=(site != "ShenzhenBay")))
    onstation = pd.concat(onst, ignore_index=True)
    means = onstation.groupby("sample_id")[SENSOR_VARS + ["Cond", "TDS"]].mean()
    paired = paired.set_index("sample_id")
    paired.loc[means.index, SENSOR_VARS + ["Cond", "TDS"]] = means
    paired = paired.reset_index()
    paired["n_sonde_readings"] = paired["sample_id"].map(onstation.groupby("sample_id").size())

    from .pahs import sum_pahs
    paired["SumPAHs"] = sum_pahs(paired, PAH_LOD)
    paired = paired.drop(columns="_y_true")
    paired["date"] = paired["date"].dt.strftime("%Y-%m-%d")
    for c in SENSOR_VARS + ["Cond", "TDS"] + OPTICAL_VARS:
        paired[c] = paired[c].astype(float).round(5 if c.startswith("S") and c != "Sal" and c != "SUVA254" else 4)
    return paired, onstation, rain


def make_application(rain: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Daily samples (ΣPAHs only) + sonde readings at sampling, 22 Jul–22 Aug 2024."""
    rng = np.random.default_rng(seed)
    r = rain["rain_target_mm"].to_numpy()
    out = []
    for site, w, k in [("YangyongRiver", (0.6, 1.0, 0.5, 0.25, 0.1), 0.8), ("ShenzhenBay", (0.2, 0.5, 0.7, 0.5, 0.3), 0.7)]:
        run = _runoff_index(r, w, scale=45.0) * k * 1.25
        n = len(r)
        sens, opt, y = _simulate(rng, site, "calib2024", run, _ar1(rng, n, .6, .3), _ar1(rng, n, .5, .4),
                                 _ar1(rng, n, .3, .3), np.zeros(n))
        df = pd.DataFrame({"site": site, "date": rain["date"].dt.strftime("%Y-%m-%d"), "station": 7 if site == "YangyongRiver" else 8})
        df = pd.concat([df, sens[SENSOR_VARS + ["Cond", "TDS"]].round(4)], axis=1)
        df["SumPAHs"] = np.round(y, 2)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def _lawnmower(n, lat0, lon0, dlat, dlon, legs=8):
    t = np.linspace(0, legs, n)
    leg = np.floor(t).astype(int).clip(max=legs - 1)
    frac = t - leg
    lon = lon0 + dlon * np.where(leg % 2 == 0, frac, 1 - frac)
    lat = lat0 + dlat * (leg + frac * 0.15) / legs
    return lat, lon


def make_hf(rain: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """A subset of high-frequency sonde survey records (1 Hz), with QA/QC artefacts."""
    rng = np.random.default_rng(seed)
    run_s = _runoff_index(rain["rain_dasha_mm"].to_numpy())
    surveys = [("DashaRiver", "summer2024", "2024-07-28", 900, (22.5183, 113.9496, 0.0042, 0.0042)),
               ("DashaRiver", "summer2024", "2024-08-06", 900, (22.5183, 113.9496, 0.0042, 0.0042)),
               ("DashaRiver", "summer2024", "2024-08-15", 900, (22.5183, 113.9496, 0.0042, 0.0042)),
               ("DashaRiver", "winter2025", "2025-01-16", 900, (22.5183, 113.9496, 0.0042, 0.0042)),
               ("YangyongRiver", "calib2024", "2024-06-04", 600, (22.8020, 113.8735, 0.0012, 0.0075)),
               ("ShenzhenBay", "calib2024", "2024-05-28", 1200, (22.4400, 113.9300, 0.0650, 0.1150))]
    out = []
    for site, camp, day, n, (lat0, lon0, dlat, dlon) in surveys:
        d = pd.Timestamp(day)
        if camp == "summer2024":
            run = run_s[SUMMER_DATES.get_loc(d)]
        else:
            run = {"winter2025": -0.2, "calib2024": 0.5}[camp]
        lat, lon = _lawnmower(n, lat0, lon0, dlat, dlon)
        # smooth spatial field: higher DOM near the river mouth / upstream end
        xs = (lon - lon0) / dlon
        ys = (lat - lat0) / dlat
        field = 0.6 * np.exp(-((xs - 0.1) ** 2 + (ys - 0.9) ** 2) / 0.25) - 0.3 * xs
        sens, _, _ = _simulate(rng, site, camp, np.full(n, run) + 0.4 * field, np.full(n, rng.normal(0, .3)),
                               np.full(n, rng.normal(0, .3)) + 0.1 * np.sin(np.linspace(0, 6, n)),
                               np.full(n, rng.normal(0, .3)), field)
        # sensor time series are smoother than independent draws
        sens = sens.rolling(15, min_periods=1, center=True).mean()
        df = pd.DataFrame({"timestamp": d + pd.Timedelta(hours=10) + pd.to_timedelta(np.arange(n), unit="s"),
                           "site": site, "campaign": camp, "survey_id": f"{site[:2].upper()}-{d:%Y%m%d}",
                           "lat": lat.round(6), "lon": lon.round(6), "depth_m": (0.5 + 0.04 * rng.normal(size=n)).round(3)})
        df = pd.concat([df, sens.round(4)], axis=1)
        # QA/QC artefacts: stabilisation (first 40 s), air exposure, spikes
        df.loc[:39, "Turb"] = df.loc[:39, "Turb"] * np.linspace(3, 1, 40)
        air = rng.choice(np.arange(60, n), 6, replace=False)
        df.loc[air, "depth_m"] = 0.02
        df.loc[air, ["Sal", "FDOM", "Turb"]] = 0.0
        spikes = rng.choice(np.arange(60, n), 4, replace=False)
        df.loc[spikes, "Turb"] = df.loc[spikes, "Turb"] * 25
        out.append(df)
    return pd.concat(out, ignore_index=True)


def make_raw_spectra(n: int = 30, seed: int = 5) -> dict:
    """Example UV–vis absorbance spectra and raw EEMs (with scatter and inner-filter effects).

    Returns a dict with ``wl`` (240–600 nm), ``absorbance`` (n x wl, decadic, 1 cm),
    ``ex`` (200–450 nm, 5 nm), ``em`` (250–600 nm, 2 nm), ``eem`` (n x ex x em, raw a.u.),
    ``blank`` (ex x em, ultrapure water), ``true_scores`` (n x 3, R.U. at the component maxima)
    and ``raman_area`` (blank Raman integral used for normalisation).
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(240, 601, 1.0)
    ex = np.arange(200, 451, 5.0)
    em = np.arange(250, 601, 2.0)
    E, M = np.meshgrid(ex, em, indexing="ij")

    def g(x, mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2)

    # component loadings (C1 humic-like, C2 tryptophan-like, C3 tyrosine-like)
    exl = [0.9 * g(ex, 250, 22) + 0.45 * g(ex, 330, 30), 0.8 * g(ex, 228, 10) + 0.5 * g(ex, 280, 12),
           0.9 * g(ex, 225, 9) + 0.45 * g(ex, 275, 10)]
    eml = [g(em, 455, 62), g(em, 345, 24), g(em, 305, 14)]
    exl = [x / x.max() for x in exl]
    eml = [x / x.max() for x in eml]
    scores = np.column_stack([rng.uniform(0.6, 2.2, n), rng.uniform(0.8, 6.5, n), rng.uniform(0.1, 1.2, n)])
    true = np.einsum("nr,ri,rj->nij", scores, np.array(exl), np.array(eml))           # R.U.

    # absorbance: two-slope CDOM model, decadic A for 1-cm path
    a350 = 0.6 * scores[:, 0] + 0.25 * scores[:, 1] + rng.uniform(0.2, 0.6, n)          # m⁻¹ (Napierian)
    S1 = rng.uniform(0.016, 0.024, n)
    S2 = rng.uniform(0.014, 0.020, n)
    a = (a350[:, None] * 0.6 * np.exp(-S1[:, None] * (wl - 350)) + a350[:, None] * 0.4 * np.exp(-S2[:, None] * (wl - 350)))
    A = a * 0.01 / 2.303 + 0.0004 * rng.normal(size=a.shape)

    # scatter (water Raman + Rayleigh) and inner-filter attenuation
    raman_em = 1.0 / (1.0 / E - 3400e-7)
    raman_area = 1250.0                                         # a.u.·nm of the blank Raman peak at Ex 350 nm
    amp = raman_area / (6.0 * np.sqrt(2 * np.pi))               # Gaussian band (sd 6 nm) with that integral
    blank = amp * (g(M, raman_em, 6.0) + 8.0 * g(M, E, 4.0) + 3.0 * g(M, 2 * E, 6.0)) + 0.5 * rng.normal(size=E.shape)
    Aex = np.array([np.interp(ex, wl, A[i], left=A[i, 0]) for i in range(n)])
    Aem = np.array([np.interp(em, wl, A[i], left=A[i, 0]) for i in range(n)])
    ife = 10 ** (-(Aex[:, :, None] + Aem[:, None, :]) / 2)
    eem = true * raman_area * ife + blank[None] + 1.0 * rng.normal(size=true.shape)
    return {"wl": wl, "absorbance": A, "ex": ex, "em": em, "eem": eem, "blank": blank,
            "true_scores": scores, "raman_area": raman_area,
            "sample_id": np.array([f"RAW-{i + 1:03d}" for i in range(n)])}
