"""High-frequency surrogate reconstruction from sonde records (Section 2.5, Note S6.2).

For each record the seven sonde variables are passed through the fitted model
(SOLACE: sensor-to-leaf classifier → leaf encoding → final regressor) to give a
point estimate and a 90 % conformal prediction interval. Records whose sensor
values fall outside the range of the training samples are flagged as
extrapolations (the coverage guarantee does not hold for them). Timestamp,
coordinates and domain identifier are retained.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .features import SENSOR_VARS
from .sonde import calibration_domain


def reconstruct(model, records: pd.DataFrame, X_train: np.ndarray, keep_cols=("timestamp", "site", "survey_id",
                                                                             "lat", "lon", "depth_m")) -> pd.DataFrame:
    X = records[SENSOR_VARS].to_numpy(float)
    t0 = time.perf_counter()
    pred = model.predict(X)
    elapsed = time.perf_counter() - t0
    out = records[[c for c in keep_cols if c in records]].copy().reset_index(drop=True)
    out["SumPAHs_hat"] = pred
    if hasattr(model, "predict_interval"):
        iv = model.predict_interval(X)
        out["PI90_low"], out["PI90_high"] = np.clip(iv[:, 0], 0, None), iv[:, 1]
    dom = calibration_domain(X_train, X)
    out["extrapolation"] = ~dom["in_domain"].to_numpy()
    out["out_of_range_vars"] = dom["out_of_range_vars"].to_numpy()
    out.attrs["seconds"] = elapsed
    out.attrs["ms_per_record"] = 1000 * elapsed / max(1, len(X))
    return out
