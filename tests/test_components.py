import numpy as np
import pytest

from cascadepahs import metrics, optics, parafac, stats
from cascadepahs.interpret import exact_interventional_shap, sampling_interventional_shap
from cascadepahs.models import QuantileRandomForestRegressor, SVRPlus, conformal_quantile
from cascadepahs.models.controls import GeneralizedDistillation, MultiTaskMLP, TwoStageRegression


# ------------------------------------------------------------------ optics
def test_spectral_slope_and_uv_descriptors():
    wl = np.arange(240, 601, 1.0)
    a = 3.0 * np.exp(-0.018 * (wl - 350))
    A = a * 0.01 / 2.303
    d = optics.uv_descriptors(wl, A, doc=2.0, baseline_nm=None)
    assert abs(d["a350"] - 3.0) < 1e-6
    assert abs(d["S275_295"] - 0.018) < 1e-6 and abs(d["S350_400"] - 0.018) < 1e-6
    assert abs(d["SR"] - 1.0) < 1e-6
    assert abs(d["SUVA254"] - d["a254"] / 2.0) < 1e-9


def test_eem_indices_on_analytic_eem():
    ex = np.arange(200, 451, 5.0)
    em = np.arange(250, 601, 2.0)
    E, M = np.meshgrid(ex, em, indexing="ij")
    eem = np.exp(-((M - 450) / 60) ** 2) * np.ones_like(E)
    fi = optics.fluorescence_index(eem, ex, em)
    assert abs(fi - np.exp(-((470 - 450) / 60) ** 2) / np.exp(-((520 - 450) / 60) ** 2)) < 1e-3
    hix = optics.humification_index(eem, ex, em)
    assert 0 < hix < 1
    mask = optics.scatter_mask(ex, em)
    assert mask[np.searchsorted(ex, 300), np.searchsorted(em, 300)]          # Rayleigh band masked


def test_parafac_recovers_rank3():
    rng = np.random.default_rng(0)
    B, C = rng.random((20, 3)), rng.random((25, 3))
    A = rng.random((40, 3)) * 3
    X = np.einsum("ir,jr,kr->ijk", A, B, C)
    m = parafac.nn_parafac(X, 3, n_init=2, seed=0, n_iter=3000)
    assert m["explained"] > 0.999
    order, tcc = parafac.match_components(C, m["C"])
    assert np.all(tcc > 0.99)


# ------------------------------------------------------------------ intervals
def test_qrf_quantiles_and_conformal_quantile():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (400, 2))
    y = 5 * X[:, 0] + rng.normal(0, 1, 400)
    q = QuantileRandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=0).fit(X[:300], y[:300])
    qq = q.predict_quantiles(X[300:], (0.05, 0.95))
    cov = np.mean((y[300:] >= qq[:, 0]) & (y[300:] <= qq[:, 1]))
    assert 0.7 < cov <= 1.0
    s = np.arange(1, 101, dtype=float)
    assert conformal_quantile(s, 0.1) == 91.0


# ------------------------------------------------------------------ LUPI comparison models
@pytest.fixture(scope="module")
def lupi_data():
    rng = np.random.default_rng(1)
    n = 90
    z = rng.normal(size=(n, 3))
    X = np.column_stack([z[:, 0] + 0.3 * rng.normal(size=n), z[:, 1] + 0.3 * rng.normal(size=n), rng.normal(size=n)])
    O = z + 0.05 * rng.normal(size=(n, 3))
    y = 2 * z[:, 0] - z[:, 1] + 0.3 * rng.normal(size=n)
    g = np.repeat(np.arange(15), 6).astype(str)
    return X, O, y, g


def test_svr_plus(lupi_data):
    X, O, y, g = lupi_data
    m = SVRPlus(C_grid=(1.0,), epsilon_grid=(0.1,), tune=False, C=1.0).fit(X[:70], y[:70], g[:70], O[:70])
    assert metrics.r2(y[70:], m.predict(X[70:])) > 0.6


def test_other_lupi_models(lupi_data):
    X, O, y, g = lupi_data
    for m in [TwoStageRegression(params={"n_estimators": 30}, n_inner=3),
              GeneralizedDistillation(params={"n_estimators": 30}, lambdas=(0.25, 0.75), n_inner=3),
              MultiTaskMLP(hidden_grid=((16,),), n_inner=3, max_iter=500)]:
        m.fit(X[:70], y[:70], g[:70], O[:70])
        p = m.predict(X[70:])
        assert p.shape == (20,) and np.all(np.isfinite(p))


# ------------------------------------------------------------------ Shapley fallback
def test_exact_shapley_efficiency_and_linear_case():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(30, 4))
    X = rng.normal(size=(5, 4))
    w = np.array([1.0, -2.0, 0.5, 0.0])
    f = lambda Z: Z @ w + 0.3 * Z[:, 0] * Z[:, 1]                     # noqa: E731
    phi = exact_interventional_shap(f, X, B)
    np.testing.assert_allclose(phi.sum(1), f(X) - f(B).mean(), atol=1e-10)   # efficiency
    lin = lambda Z: Z @ w                                              # noqa: E731
    np.testing.assert_allclose(exact_interventional_shap(lin, X, B), (X - B.mean(0)) * w, atol=1e-10)
    approx = sampling_interventional_shap(lin, X, B, n_permutations=8)
    np.testing.assert_allclose(approx, (X - B.mean(0)) * w, atol=1e-10)


# ------------------------------------------------------------------ statistics
def test_statistics_helpers():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(10), 5)
    v = rng.normal(size=10)[g] * 3 + rng.normal(size=50)
    r = stats.anova_icc(v, g)
    assert r["ICC"] > 0.5 and r["p"] < 0.01
    k = stats.kruskal_epsilon2(np.r_[rng.normal(0, 1, 30), rng.normal(2, 1, 30)], np.repeat(["a", "b"], 30))
    assert abs(k["epsilon2"] - k["H"] / 59) < 1e-12
    X = np.r_[rng.normal(0, 1, (30, 3)), rng.normal(1, 1, (30, 3))]
    t = stats.manova(X, np.repeat([0, 1], 30))
    assert (t["p"] < 0.001).all()


def test_metrics_bootstrap_and_screening():
    import pandas as pd

    rng = np.random.default_rng(0)
    y = rng.uniform(10, 200, 60)
    df = pd.DataFrame({"date": np.repeat(np.arange(12), 5).astype(str), "y": y,
                       "A": y + rng.normal(0, 10, 60), "B": y + rng.normal(0, 30, 60)})
    b = metrics.paired_date_bootstrap(df, ["A", "B"], n_boot=200, seed=0)
    assert b.iloc[-1]["estimate"] > 0 and b.iloc[-1]["ci_low"] <= b.iloc[-1]["estimate"] <= b.iloc[-1]["ci_high"]
    s = metrics.screening(y, y, 100)
    assert s["sensitivity"] == 1 and s["specificity"] == 1
    t = metrics.interval_triage(y, y - 1, y + 1, 100)
    assert t["accuracy_confident"] == 1
