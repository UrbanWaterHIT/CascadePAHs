from .base import ALGORITHMS, HAS_XGBOOST, make_estimator, tune
from .conformal import ConformalIntervalModel, conformal_quantile
from .controls import (GeneralizedDistillation, MultiTaskMLP, OpticsOnlyRF, SensorsPlusOpticsRF,
                       TwoStageRegression, capacity_control, shuffled_control)
from .qrf import QuantileRandomForestRegressor
from .solace import SOLACE, chance_agreement
from .sorfe import SORFE, SensorOnlyModel
from .svr_plus import SVRPlus, SVRPlusCore

__all__ = [
    "ALGORITHMS", "HAS_XGBOOST", "make_estimator", "tune", "ConformalIntervalModel", "conformal_quantile",
    "GeneralizedDistillation", "MultiTaskMLP", "OpticsOnlyRF", "SensorsPlusOpticsRF", "TwoStageRegression",
    "capacity_control", "shuffled_control", "QuantileRandomForestRegressor", "SOLACE", "chance_agreement",
    "SORFE", "SensorOnlyModel", "SVRPlus", "SVRPlusCore",
]
