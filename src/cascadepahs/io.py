"""Loading and validating the paired, high-frequency and application datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import OPTICAL_VARS, SENSOR_VARS, SOURCE_SITE, TARGET

REQUIRED_PAIRED = ["sample_id", "site", "campaign", "date", "station"] + SENSOR_VARS + OPTICAL_VARS + [TARGET]
REQUIRED_HF = ["timestamp", "site", "lat", "lon"] + SENSOR_VARS
REQUIRED_APPLICATION = ["site", "date"] + SENSOR_VARS + [TARGET]


def _check(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}. See data/README.md for the schema.")


def load_paired(path: str | Path) -> pd.DataFrame:
    """Complete PAH–optical–sensor paired observations (242 rows in the study)."""
    df = pd.read_csv(path)
    _check(df, REQUIRED_PAIRED, "paired samples")
    df["date"] = pd.to_datetime(df["date"])
    if df[SENSOR_VARS + OPTICAL_VARS + [TARGET]].isna().any().any():
        raise ValueError("paired samples: complete paired observations must not contain missing values")
    return df.reset_index(drop=True)


def load_hf(path: str | Path) -> pd.DataFrame:
    """High-frequency sonde records (unlabelled; used only for deployment)."""
    df = pd.read_csv(path)
    _check(df, REQUIRED_HF, "high-frequency records")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_application(path: str | Path) -> pd.DataFrame:
    """Application-period samples (ΣPAHs only + sonde readings at sampling)."""
    df = pd.read_csv(path)
    _check(df, REQUIRED_APPLICATION, "application-period samples")
    df["date"] = pd.to_datetime(df["date"])
    return df


def source_domain(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["site"] == SOURCE_SITE].reset_index(drop=True)


def site_subset(df: pd.DataFrame, site: str) -> pd.DataFrame:
    return df[df["site"] == site].reset_index(drop=True)


def arrays(df: pd.DataFrame, optical_vars: list[str] | None = None):
    """Return (X_sensor, O_optical, y, groups=date string) as numpy arrays."""
    optical_vars = OPTICAL_VARS if optical_vars is None else optical_vars
    X = df[SENSOR_VARS].to_numpy(float)
    O = df[optical_vars].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    groups = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    return X, O, y, groups


def is_summer(df: pd.DataFrame) -> np.ndarray:
    return (df["campaign"].astype(str).str.startswith("summer")).to_numpy()
