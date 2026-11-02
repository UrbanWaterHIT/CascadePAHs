"""Minimal use of SOLACE and SORFE on a paired dataset.

    python examples/quickstart.py [path/to/paired_samples.csv]

Fits both models on the Dasha River summer samples except the last four survey
days, predicts those days from the seven sonde variables only, and prints R², MAE
and 90 % prediction intervals. With no argument the demo file is used.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascadepahs.features import OPTICAL_VARS, SENSOR_VARS, TARGET
from cascadepahs.io import load_paired
from cascadepahs.metrics import coverage, mae, r2
from cascadepahs.models import SOLACE, SORFE, ConformalIntervalModel

path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "demo" / "paired_samples.csv"
df = load_paired(path)
df = df[df["campaign"] == "summer2024"].reset_index(drop=True)

last4 = sorted(df["date"].unique())[-4:]
test = df["date"].isin(last4).to_numpy()
X, O, y = df[SENSOR_VARS].to_numpy(float), df[OPTICAL_VARS].to_numpy(float), df[TARGET].to_numpy(float)
groups = df["date"].dt.strftime("%Y-%m-%d").to_numpy()

solace = ConformalIntervalModel(
    SOLACE(optical_params={"n_estimators": 100, "min_samples_leaf": 8}, classifier_params={"n_estimators": 100},
           final_params={"n_estimators": 200}, quantile=True, n_inner=5, random_state=42),
    alpha=0.10)
solace.fit(X[~test], y[~test], groups[~test], O[~test])           # optical data: training only
sorfe = SORFE(params={"n_estimators": 200}, random_state=42).fit(X[~test], y[~test], groups[~test])

p_sol = solace.predict(X[test])                                     # sonde variables only
p_sor = sorfe.predict(X[test])
pi = solace.predict_interval(X[test])
print(f"held-out days: {', '.join(d.strftime('%d %b') for d in last4)}  (n = {test.sum()})")
print(f"SORFE   R² = {r2(y[test], p_sor):.3f}   MAE = {mae(y[test], p_sor):.1f} ng/L")
print(f"SOLACE  R² = {r2(y[test], p_sol):.3f}   MAE = {mae(y[test], p_sol):.1f} ng/L")
print("SOLACE 90 % PI:", {k: round(v, 3) for k, v in coverage(y[test], pi[:, 0], pi[:, 1]).items()})
print("leaf agreement with measured-optics leaves:", np.round(solace.leaf_agreement(X[test], O[test]).mean(), 3),
      "(chance:", round(solace.chance_agreement_, 3), ")")
