"""Build model factories and load data from a configuration (shared by the scripts)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import grid_from_cfg, resolve_path
from .features import OPTICAL_VARS
from .io import load_paired, source_domain
from .models import (ConformalIntervalModel, GeneralizedDistillation, MultiTaskMLP, OpticsOnlyRF, SensorOnlyModel,
                     SensorsPlusOpticsRF, SOLACE, SVRPlus, TwoStageRegression)


def _tuplify(params: dict | None) -> dict:
    out = {}
    for k, v in (params or {}).items():
        out[k] = tuple(v) if isinstance(v, list) else v
    return out


def fixed_params(cfg, alg: str) -> dict:
    return _tuplify(cfg.get("fixed_params", {}).get(alg, {}))


def sensor_only(cfg, alg: str = "RF", quantile: bool = False, seed: int | None = None):
    rs = cfg["random_state"] if seed is None else seed

    def factory():
        return SensorOnlyModel(algorithm=alg, params=fixed_params(cfg, alg), grid=grid_from_cfg(cfg, alg),
                               tune=bool(cfg.get("tune", False)), n_inner=cfg["n_inner"], quantile=quantile,
                               random_state=rs, n_jobs=cfg.get("model_n_jobs", 1))
    return factory


def solace(cfg, final_alg: str = "RF", quantile: bool = False, seed: int | None = None, **overrides):
    s = cfg["solace"]
    tune = bool(cfg.get("tune", False))
    rf_grid = grid_from_cfg(cfg, "RF")
    grids = {}
    if tune and s.get("tune_optical", True):
        grids["optical"] = rf_grid
    if tune and s.get("tune_classifier", True):
        grids["classifier"] = rf_grid
    if tune and s.get("tune_final", True):
        grids["final"] = grid_from_cfg(cfg, final_alg)
    final_params = _tuplify(s.get("final_params")) if final_alg == "RF" else fixed_params(cfg, final_alg)
    rs = cfg["random_state"] if seed is None else seed

    def factory():
        kw = dict(final_algorithm=final_alg, optical_params=_tuplify(s.get("optical_params")),
                  classifier_params=_tuplify(s.get("classifier_params")), final_params=final_params,
                  grids=grids or None, tune=bool(grids), n_inner=cfg["n_inner"],
                  leaf_targets=s.get("leaf_targets", "crossfit"), quantile=quantile,
                  random_state=rs, n_jobs=cfg.get("model_n_jobs", 1))
        kw.update(overrides)
        return SOLACE(**kw)
    return factory


def with_intervals(factory, cfg):
    def f():
        return ConformalIntervalModel(factory(), alpha=cfg["alpha"], n_folds=cfg["n_inner"])
    return f


def control_factories(cfg) -> dict:
    """Factories of Table S17 (except the ablations, which change the optical columns)."""
    c = cfg["controls"]
    rf = fixed_params(cfg, "RF")
    rs, nj, ni = cfg["random_state"], cfg.get("model_n_jobs", 1), cfg["n_inner"]
    tune = bool(cfg.get("tune", False))
    rf_grid = grid_from_cfg(cfg, "RF")
    return {
        "SORFE": sensor_only(cfg, "RF"),
        "SOLACE": solace(cfg, "RF"),
        "Capacity control": solace(cfg, "RF", leaf_source="sensor"),
        "Shuffled control": solace(cfg, "RF", shuffle_leaf_targets=True),
        "Two-stage regression": lambda: TwoStageRegression(params=rf, grid={"n_estimators": c["two_stage"]["n_estimators"]},
                                                           tune=tune, n_inner=ni, random_state=rs, n_jobs=nj),
        "Generalized distillation": lambda: GeneralizedDistillation(params=rf, lambdas=tuple(c["distillation"]["lambdas"]),
                                                                    tune=True, n_inner=ni, random_state=rs, n_jobs=nj),
        "SVR+": lambda: SVRPlus(C_grid=tuple(c["svr_plus"]["C"]), epsilon_grid=tuple(c["svr_plus"]["epsilon"]),
                                gamma_priv=c["svr_plus"].get("gamma_priv", 1.0), tune=True, n_inner=ni),
        "Multi-task learning": lambda: MultiTaskMLP(hidden_grid=tuple(tuple(h) for h in c["multitask"]["hidden"]),
                                                    tune=True, n_inner=ni, random_state=rs),
        "Measured optics only": lambda: OpticsOnlyRF(params=rf, grid=rf_grid, tune=tune, n_inner=ni,
                                                     random_state=rs, n_jobs=nj),
        "Sensors + measured optics": lambda: SensorsPlusOpticsRF(params=rf, grid=rf_grid, tune=tune, n_inner=ni,
                                                                 random_state=rs, n_jobs=nj),
    }


def load_source(cfg) -> pd.DataFrame:
    return source_domain(load_paired(resolve_path(cfg, "paired")))


def results_dir(cfg, sub: str) -> Path:
    p = resolve_path(cfg, "results") / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def optical_vars_for(df: pd.DataFrame) -> list[str]:
    return [v for v in OPTICAL_VARS if v in df.columns]
