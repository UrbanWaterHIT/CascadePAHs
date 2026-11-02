"""Variable definitions shared by every module.

Column names follow the CSV schema documented in ``data/README.md``.
They correspond one-to-one to the variables in Supplementary Table S2.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Seven deployment-available sonde predictors (YSI EXO2)
# --------------------------------------------------------------------------
SENSOR_VARS: list[str] = ["T", "pH", "DO", "Sal", "Turb", "Chla", "FDOM"]

SENSOR_LABELS: dict[str, str] = {
    "T": "Temperature (°C)",
    "pH": "pH",
    "DO": "Dissolved oxygen (mg L⁻¹)",
    "Sal": "Salinity (PSU)",
    "Turb": "Turbidity (FNU)",
    "Chla": "Chlorophyll-a (µg L⁻¹)",
    "FDOM": "FDOM (QSU)",
}

# Recorded by the sonde but excluded before modelling (collinear with Sal)
EXCLUDED_SONDE_VARS: list[str] = ["Cond", "TDS"]

# --------------------------------------------------------------------------
# 17 laboratory optical descriptors (training-only privileged information)
# --------------------------------------------------------------------------
EEM_VARS: list[str] = [
    "C1", "C2", "C3", "TotalF", "ProteinF", "HumicF", "FI", "BIX", "HIX", "BetaAlpha",
]
UVVIS_VARS: list[str] = ["a254", "a350", "S275_295", "S350_400", "SR"]
DOC_VARS: list[str] = ["DOC", "SUVA254"]

OPTICAL_VARS: list[str] = EEM_VARS + UVVIS_VARS + DOC_VARS          # 17 descriptors
PARAFAC_VARS: list[str] = ["C1", "C2", "C3"]
# PARAFAC components were derived separately for each site, so they are
# excluded from the cross-site refit (14 descriptors remain; Table S21).
CROSS_SITE_OPTICAL_VARS: list[str] = [v for v in OPTICAL_VARS if v not in PARAFAC_VARS]

OPTICAL_GROUPS: dict[str, list[str]] = {
    "EEM": EEM_VARS,      # 10 descriptors
    "UVVIS": UVVIS_VARS,  # 5 descriptors
    "DOC": DOC_VARS,      # 2 descriptors
}

OPTICAL_LABELS: dict[str, str] = {
    "C1": "C$_1$", "C2": "C$_2$", "C3": "C$_3$", "TotalF": "F$_{total}$",
    "ProteinF": "Protein-like fluorescence", "HumicF": "Humic-like fluorescence",
    "FI": "FI", "BIX": "BIX", "HIX": "HIX", "BetaAlpha": "β:α",
    "a254": "a$_{254}$", "a350": "a$_{350}$", "S275_295": "S$_{275–295}$",
    "S350_400": "S$_{350–400}$", "SR": "S$_R$", "SUVA254": "SUVA$_{254}$", "DOC": "DOC",
}

# --------------------------------------------------------------------------
# Response and metadata
# --------------------------------------------------------------------------
TARGET: str = "SumPAHs"        # dissolved ΣPAHs (sum of 16 US EPA PAHs), ng L⁻¹
META_VARS: list[str] = ["sample_id", "site", "campaign", "date", "station", "lat", "lon"]

SITES = {"DashaRiver": "Dasha River", "ShenzhenBay": "Shenzhen Bay", "YangyongRiver": "Yangyong River"}
SOURCE_SITE = "DashaRiver"
TARGET_SITES = ["YangyongRiver", "ShenzhenBay"]

# 16 US EPA priority PAHs: (column, full name, rings)
PAH_CONGENERS: list[tuple[str, str, int]] = [
    ("NaP", "Naphthalene", 2),
    ("Acy", "Acenaphthylene", 3),
    ("Ace", "Acenaphthene", 3),
    ("Flu", "Fluorene", 3),
    ("Phe", "Phenanthrene", 3),
    ("Ant", "Anthracene", 3),
    ("Fla", "Fluoranthene", 4),
    ("Pyr", "Pyrene", 4),
    ("BaA", "Benz[a]anthracene", 4),
    ("Chr", "Chrysene", 4),
    ("BbF", "Benzo[b]fluoranthene", 5),
    ("BkF", "Benzo[k]fluoranthene", 5),
    ("BaP", "Benzo[a]pyrene", 5),
    ("IcdP", "Indeno[1,2,3-cd]pyrene", 6),
    ("DahA", "Dibenz[a,h]anthracene", 5),
    ("BghiP", "Benzo[g,h,i]perylene", 6),
]
PAH_COLS: list[str] = [c for c, _, _ in PAH_CONGENERS]
