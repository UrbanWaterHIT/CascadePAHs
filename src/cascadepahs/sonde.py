"""Sonde record QA/QC, sensor–sample matching and calibration-domain flags (Sections 2.2 and 2.5).

* ``qaqc`` removes records affected by sensor stabilisation, exposure to air,
  values outside plausible ranges and implausible signal discontinuities (spikes).
* ``match_onstation`` builds the paired sensor vector of each sample as the mean of
  the readings recorded on station during the sampling deployments.
* ``calibration_domain`` flags records whose seven sensor values fall outside the
  range of the samples used to train the model applied to them (extrapolations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SENSOR_VARS

# plausible ranges (inclusive) for the EXO2 variables used here
DEFAULT_RANGES = {
    "T": (0.0, 40.0), "pH": (5.0, 10.5), "DO": (0.0, 20.0), "Sal": (0.0, 40.0),
    "Turb": (0.0, 1000.0), "Chla": (0.0, 400.0), "FDOM": (0.0, 300.0),
}


def qaqc(df: pd.DataFrame, *, survey_col: str = "survey_id", time_col: str = "timestamp",
         depth_col: str = "depth_m", min_depth: float = 0.2, stabilization_s: float = 40.0,
         ranges: dict | None = None, spike_vars=("Turb", "FDOM", "Chla"), spike_window: int = 11,
         spike_k: float = 8.0, air_vars=("Sal", "FDOM", "Turb")) -> pd.DataFrame:
    """Return a copy of ``df`` with boolean flag columns and ``qc_pass``.

    Flags: ``flag_stabilization`` (first ``stabilization_s`` seconds of each survey),
    ``flag_air`` (depth < ``min_depth`` or simultaneous zeros in ``air_vars``),
    ``flag_range`` (outside plausible range), ``flag_spike`` (|x − rolling median| >
    ``spike_k`` × rolling MAD).
    """
    ranges = DEFAULT_RANGES if ranges is None else ranges
    out = df.copy()
    t = pd.to_datetime(out[time_col])
    grp = out[survey_col] if survey_col in out else pd.Series(0, index=out.index)
    t0 = t.groupby(grp).transform("min")
    out["flag_stabilization"] = (t - t0).dt.total_seconds() < stabilization_s
    air = pd.Series(False, index=out.index)
    if depth_col in out:
        air |= out[depth_col] < min_depth
    air |= (out[list(air_vars)] == 0).all(axis=1)
    out["flag_air"] = air
    rng_flag = pd.Series(False, index=out.index)
    for v, (lo, hi) in ranges.items():
        if v in out:
            rng_flag |= (out[v] < lo) | (out[v] > hi)
    out["flag_range"] = rng_flag
    spike = pd.Series(False, index=out.index)
    for v in spike_vars:
        if v not in out:
            continue
        x = out[v].where(~air)
        med = x.groupby(grp).transform(lambda s: s.rolling(spike_window, center=True, min_periods=3).median())
        mad = (x - med).abs().groupby(grp).transform(
            lambda s: s.rolling(spike_window * 3, center=True, min_periods=3).median())
        spike |= ((x - med).abs() > spike_k * (1.4826 * mad + 1e-6)).fillna(False)
    out["flag_spike"] = spike
    out["qc_pass"] = ~(out["flag_stabilization"] | out["flag_air"] | out["flag_range"] | out["flag_spike"])
    return out


def match_onstation(readings: pd.DataFrame, variables=None, key: str = "sample_id") -> pd.DataFrame:
    """Mean of the on-station readings of each sampling event (+ number of readings)."""
    variables = SENSOR_VARS if variables is None else list(variables)
    g = readings.groupby(key)
    out = g[variables].mean()
    out["n_sonde_readings"] = g.size()
    if "timestamp" in readings:
        ts = pd.to_datetime(readings["timestamp"])
        span = ts.groupby(readings[key]).agg(lambda s: (s.max() - s.min()).total_seconds() / 60)
        out["window_min"] = span
    return out.reset_index()


def calibration_domain(X_train: np.ndarray, X: np.ndarray, names=None) -> pd.DataFrame:
    """In-domain flag (all variables within the training range) and the out-of-range variables."""
    X_train, X = np.asarray(X_train, float), np.asarray(X, float)
    lo, hi = X_train.min(0), X_train.max(0)
    out = (X < lo) | (X > hi)
    names = SENSOR_VARS if names is None else list(names)
    return pd.DataFrame({
        "in_domain": ~out.any(axis=1),
        "n_out_of_range": out.sum(axis=1),
        "out_of_range_vars": [",".join(np.array(names)[row]) for row in out],
    })
