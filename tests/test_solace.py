import numpy as np

from cascadepahs.features import OPTICAL_VARS, SENSOR_VARS, TARGET
from cascadepahs.models import SORFE, capacity_control, shuffled_control
from cascadepahs.models.solace import _majority_map, chance_agreement


def _arrays(df):
    return (df[SENSOR_VARS].to_numpy(float), df[TARGET].to_numpy(float), df["date"].astype(str).to_numpy(),
            df[OPTICAL_VARS].to_numpy(float))


def test_fit_predict_sensor_only(dasha, small_solace):
    X, y, g, O = _arrays(dasha)
    m = small_solace().fit(X, y, g, O)
    p = m.predict(X[:10])                       # no optical data needed
    assert p.shape == (10,) and np.all(np.isfinite(p))
    L = m.predict_leaves(X[:10])
    assert L.shape == (10, 20)
    agree = m.leaf_agreement(X[:10], O[:10])
    assert np.all((agree >= 0) & (agree <= 1))
    assert 0 < m.chance_agreement_ < 1
    assert m.predict_oracle(X[:5], O[:5]).shape == (5,)


def test_predictions_ignore_optics_argument(dasha, small_solace):
    X, y, g, O = _arrays(dasha)
    m = small_solace().fit(X, y, g, O)
    rng = np.random.default_rng(1)
    np.testing.assert_array_equal(m.predict(X[:20]), m.predict(X[:20], O=rng.normal(size=(20, O.shape[1]))))


def test_leaf_targets_modes_and_controls(dasha, small_solace):
    X, y, g, O = _arrays(dasha)
    for kw in [dict(leaf_targets="insample"), dict(leaf_targets="crossfit")]:
        m = small_solace(**kw).fit(X, y, g, O)
        assert m.leaf_targets_.shape == (len(y), 20)
        # every target leaf belongs to the reference forest's leaf vocabulary
        ref = m.optical_forest_.apply(O)
        for t in range(3):
            leaves_t = np.flatnonzero(m.optical_forest_.estimators_[t].tree_.children_left == -1)
            assert set(np.unique(m.leaf_targets_[:, t])) <= set(leaves_t)
            assert set(np.unique(ref[:, t])) <= set(leaves_t)
    cap = capacity_control(optical_params={"n_estimators": 10, "min_samples_leaf": 8},
                           classifier_params={"n_estimators": 10}, final_params={"n_estimators": 10}, n_inner=3)
    cap.fit(X, y, g, None)                       # leaves from sensors: optics not needed
    assert cap.predict(X[:3]).shape == (3,)
    sh = shuffled_control(optical_params={"n_estimators": 10, "min_samples_leaf": 8},
                          classifier_params={"n_estimators": 10}, final_params={"n_estimators": 10}, n_inner=3)
    sh.fit(X, y, g, O)
    assert sh.predict(X[:3]).shape == (3,)


def test_privileged_variants_and_quantiles(dasha, small_solace):
    X, y, g, O = _arrays(dasha)
    for alg in ["LR", "SVR"]:
        m = small_solace(final_algorithm=alg, final_params={}).fit(X, y, g, O)
        assert np.all(np.isfinite(m.predict(X[:5])))
    q = small_solace(quantile=True).fit(X, y, g, O)
    qq = q.predict_quantiles(X[:15], (0.05, 0.5, 0.95))
    assert np.all(qq[:, 0] <= qq[:, 1]) and np.all(qq[:, 1] <= qq[:, 2])


def test_frozen_clone_keeps_hyperparameters(dasha, small_solace):
    X, y, g, O = _arrays(dasha)
    grids = {"optical": {"max_depth": [None, 5]}, "classifier": {"min_samples_split": [2, 5]},
             "final": {"max_depth": [None, 5]}}
    m = small_solace(tune=True, grids=grids).fit(X, y, g, O)
    f = m.frozen_clone()
    assert f.tune is False
    assert f.optical_params["max_depth"] == m.optical_params_["max_depth"]
    assert f.final_params == m.final_params_


def test_sorfe(dasha):
    X, y, g, O = _arrays(dasha)
    m = SORFE(params={"n_estimators": 30}, quantile=True).fit(X, y, g)
    assert m.predict(X[:4]).shape == (4,)
    assert m.predict_quantiles(X[:4]).shape == (4, 2)


def test_majority_map_and_chance():
    keys_fit = np.array([1, 1, 1, 2, 2, 3])
    vals_fit = np.array([7, 7, 8, 9, 9, 4])
    np.testing.assert_array_equal(_majority_map(keys_fit, vals_fit, np.array([2, 1, 3])), [9, 7, 4])
    L = np.array([[0, 1], [0, 1], [1, 1], [1, 1]])
    assert abs(chance_agreement(L) - (0.5 + 1.0) / 2) < 1e-12
