import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def paired():
    from cascadepahs.demo_data import make_paired

    df, _, _ = make_paired(seed=2024)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="session")
def dasha(paired):
    return paired[paired["site"] == "DashaRiver"].reset_index(drop=True)


@pytest.fixture
def small_solace():
    from cascadepahs.models import SOLACE

    def make(**kw):
        base = dict(optical_params={"n_estimators": 20, "min_samples_leaf": 8}, classifier_params={"n_estimators": 20},
                    final_params={"n_estimators": 30}, n_inner=3, random_state=0)
        base.update(kw)
        return SOLACE(**base)
    return make
