"""SOLACE – Sensor-to-Optical Leaf-Mapping Cascaded Estimator (Section 2.4, Notes S3–S4).

Training (optical descriptors O available)::

    stage 1  optical random forest            O  ->  ΣPAHs        (defines the leaf vocabulary)
             cross-fitted leaf targets L       (date-grouped inner folds)
    stage 2  multi-output sensor-to-leaf RF   X  ->  L
             out-of-fold predicted leaves L̂    (same date-grouped inner folds)
    stage 3  final regressor                  [X, onehot(L̂)] -> ΣPAHs

Deployment (sensors only)::

    X -> L̂ = classifier(X) -> [X, onehot(L̂)] -> ΣPAHs

``O`` is never touched by :meth:`predict`. Replacing the optical data or the
response of held-out samples therefore cannot change their predictions
(leakage-invariance test, ``tests/test_leakage.py``).

Variants used in the paper
--------------------------
* ``final_algorithm`` ∈ {RF, SVR, XGBoost, LR, DNN, MLP}  → the "-P" models of Table 1
  (RF-P = SOLACE).
* ``leaf_source="sensor"``          → capacity control (leaves built from sensor data).
* ``shuffle_leaf_targets=True``     → shuffled control (leaf targets permuted).
* ``optical_cols`` subset (ablations) is handled by the caller via the columns of ``O``.
* ``quantile=True`` (RF final stage) → quantile regression forest for prediction intervals.

Cross-fitted leaf targets
-------------------------
Leaves of forests fitted on different inner folds live in different index
spaces. To combine the out-of-fold memberships into one vocabulary, the
reference optical forest (fitted on all training samples) defines the leaf
vocabulary, and each held-out sample receives, for every tree t, the reference
leaf held by the majority of its co-members in tree t of the inner-fold forest
fitted without its date group (``leaf_targets="crossfit"``). ``"insample"``
uses the reference forest's leaves directly. See docs/METHODS.md.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import fit_quiet, inner_cv, make_estimator, set_core_params, tune

warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def leaf_accuracy_scorer(estimator, X, Y) -> float:
    """Mean per-tree accuracy of predicted leaf memberships (classifier tuning score)."""
    P = estimator.predict(X)
    return float(np.mean(P == Y))


def _majority_map(keys_fit: np.ndarray, vals_fit: np.ndarray, keys_query: np.ndarray) -> np.ndarray:
    """For each query key return the most frequent value among fitting rows with that key."""
    pairs, counts = np.unique(np.column_stack([keys_fit, vals_fit]), axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))       # key asc, count desc, value asc
    pairs = pairs[order]
    first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    keys_u, vals_u = pairs[first, 0], pairs[first, 1]
    pos = np.clip(np.searchsorted(keys_u, keys_query), 0, len(keys_u) - 1)
    out = vals_u[pos]
    miss = keys_u[pos] != keys_query
    if miss.any():                                                  # cannot happen for RF leaves; safe fallback
        vals, cnt = np.unique(vals_fit, return_counts=True)
        out[miss] = vals[np.argmax(cnt)]
    return out


def chance_agreement(leaves: np.ndarray) -> float:
    """Mean over trees of Σ_l p_l² (agreement expected under random assignment)."""
    vals = []
    for t in range(leaves.shape[1]):
        _, c = np.unique(leaves[:, t], return_counts=True)
        p = c / c.sum()
        vals.append(np.sum(p ** 2))
    return float(np.mean(vals))


# ----------------------------------------------------------------------------
# SOLACE
# ----------------------------------------------------------------------------
class SOLACE(BaseEstimator, RegressorMixin):
    requires_optics_at_predict = False

    def __init__(self, final_algorithm: str = "RF", optical_params: dict | None = None,
                 classifier_params: dict | None = None, final_params: dict | None = None,
                 grids: dict | None = None, tune: bool = False, n_inner: int = 5,
                 leaf_targets: str = "crossfit", leaf_source: str = "optical",
                 shuffle_leaf_targets: bool = False, quantile: bool = False,
                 batch_size: int = 2000, random_state: int = 0, n_jobs: int = 1):
        self.final_algorithm = final_algorithm
        self.optical_params = optical_params
        self.classifier_params = classifier_params
        self.final_params = final_params
        self.grids = grids
        self.tune = tune
        self.n_inner = n_inner
        self.leaf_targets = leaf_targets
        self.leaf_source = leaf_source
        self.shuffle_leaf_targets = shuffle_leaf_targets
        self.quantile = quantile
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, groups, O=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        groups = np.asarray(groups)
        if self.leaf_source == "optical":
            if O is None:
                raise ValueError("SOLACE needs the optical descriptors O during training")
            P = np.asarray(O, float)
        elif self.leaf_source == "sensor":          # capacity control
            P = X
        else:
            raise ValueError("leaf_source must be 'optical' or 'sensor'")
        grids = self.grids or {}
        cv = inner_cv(groups, self.n_inner)
        self.inner_splits_ = list(cv.split(X, y, groups))

        self.scaler_ = StandardScaler().fit(X)
        self.X_train_min_ = X.min(axis=0)
        self.X_train_max_ = X.max(axis=0)

        # ---- stage 1: optical forest (leaf vocabulary) ----------------------
        base = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
        base.set_params(**(self.optical_params or {}))
        if self.tune and grids.get("optical"):
            search = GridSearchCV(base, grids["optical"], cv=self.inner_splits_,
                                  scoring="neg_mean_absolute_error", n_jobs=1)
            search.fit(P, y)
            opt_params = {**(self.optical_params or {}), **search.best_params_}
        else:
            opt_params = dict(self.optical_params or {})
        self.optical_params_ = opt_params
        template = clone(base).set_params(**opt_params)
        self.optical_forest_ = clone(template).fit(P, y)
        L_ref = self.optical_forest_.apply(P)
        self.chance_agreement_ = chance_agreement(L_ref)

        # ---- cross-fitted leaf targets --------------------------------------
        if self.leaf_targets == "crossfit":
            L = np.empty_like(L_ref)
            for tr, te in self.inner_splits_:
                Fk = clone(template).fit(P[tr], y[tr])
                lk_tr, lk_te = Fk.apply(P[tr]), Fk.apply(P[te])
                for t in range(L_ref.shape[1]):
                    L[te, t] = _majority_map(lk_tr[:, t], L_ref[tr, t], lk_te[:, t])
        elif self.leaf_targets == "insample":
            L = L_ref.copy()
        else:
            raise ValueError("leaf_targets must be 'crossfit' or 'insample'")
        if self.shuffle_leaf_targets:                # shuffled control
            rng = np.random.default_rng(self.random_state)
            L = L[rng.permutation(len(L))]
        self.leaf_targets_ = L

        # ---- stage 2: multi-output sensor-to-leaf classifier ----------------
        cbase = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        cbase.set_params(**(self.classifier_params or {}))
        if self.tune and grids.get("classifier"):
            search = GridSearchCV(cbase, grids["classifier"], cv=self.inner_splits_,
                                  scoring=leaf_accuracy_scorer, n_jobs=1)
            search.fit(X, L)
            clf_params = {**(self.classifier_params or {}), **search.best_params_}
        else:
            clf_params = dict(self.classifier_params or {})
        self.classifier_params_ = clf_params
        ctemplate = clone(cbase).set_params(**clf_params)
        self.classifier_ = clone(ctemplate).fit(X, L)

        # out-of-fold predicted leaves for the final regressor
        L_oof = np.empty_like(L)
        for tr, te in self.inner_splits_:
            ck = clone(ctemplate).fit(X[tr], L[tr])
            L_oof[te] = self._predict_leaves_with(ck, X[te])
        self.leaf_oof_ = L_oof

        cats = [np.unique(L[:, t]) for t in range(L.shape[1])]
        self.encoder_ = OneHotEncoder(categories=cats, handle_unknown="ignore", sparse_output=True,
                                      dtype=np.float64).fit(L)
        Z = self._design(X, L_oof)

        # ---- stage 3: final regressor ---------------------------------------
        final = make_estimator(self.final_algorithm, scale_X=False, random_state=self.random_state,
                               n_jobs=self.n_jobs, quantile=self.quantile and self.final_algorithm == "RF")
        if self.tune and grids.get("final"):
            self.final_, fparams = tune(final, grids["final"], Z, y, groups, n_inner=self.n_inner)
            self.final_params_ = {**(self.final_params or {}), **fparams}
        else:
            if self.final_params:
                set_core_params(final, self.final_params)
            self.final_ = fit_quiet(final, Z, y)
            self.final_params_ = dict(self.final_params or {})
        return self

    # ------------------------------------------------------------ internals
    def _predict_leaves_with(self, clf, X) -> np.ndarray:
        X = np.asarray(X, float)
        out = []
        for s in range(0, len(X), self.batch_size):
            out.append(np.asarray(clf.predict(X[s:s + self.batch_size])))
        P = np.vstack(out)
        return P.reshape(len(X), -1).astype(np.int64)

    def _design(self, X, leaves) -> sp.csr_matrix:
        Xs = sp.csr_matrix(self.scaler_.transform(np.asarray(X, float)))
        return sp.hstack([Xs, self.encoder_.transform(leaves)], format="csr")

    # --------------------------------------------------------------- public
    def predict_leaves(self, X) -> np.ndarray:
        """Optical leaf memberships predicted from the seven sonde variables."""
        return self._predict_leaves_with(self.classifier_, X)

    def optical_leaves(self, O) -> np.ndarray:
        """Leaves obtained from measured optical descriptors (oracle; not deployable)."""
        if self.leaf_source != "optical":
            raise RuntimeError("oracle leaves are defined only for leaf_source='optical'")
        return self.optical_forest_.apply(np.asarray(O, float))

    def predict_from_leaves(self, X, leaves) -> np.ndarray:
        return self._final_predict(self._design(X, leaves))

    def _final_predict(self, Z) -> np.ndarray:
        out = [self.final_.predict(Z[s:s + self.batch_size]) for s in range(0, Z.shape[0], self.batch_size)]
        return np.concatenate(out)

    def predict(self, X, O=None) -> np.ndarray:
        """Sensor-only prediction. ``O`` is ignored (kept for a uniform interface)."""
        X = np.asarray(X, float)
        return self.predict_from_leaves(X, self.predict_leaves(X))

    def predict_oracle(self, X, O) -> np.ndarray:
        """Final regressor given the leaves of the measured optical descriptors (not deployable)."""
        return self.predict_from_leaves(X, self.optical_leaves(O))

    def predict_quantiles(self, X, quantiles=(0.05, 0.95)) -> np.ndarray:
        if not (self.quantile and self.final_algorithm == "RF"):
            raise RuntimeError("quantiles require final_algorithm='RF' and quantile=True")
        Z = self._design(X, self.predict_leaves(X))
        return self.final_.named_steps["est"].predict_quantiles(Z, quantiles)

    def leaf_agreement(self, X, O) -> np.ndarray:
        """Per-sample share of trees whose predicted leaf equals the measured-optics leaf."""
        return np.mean(self.predict_leaves(X) == self.optical_leaves(O), axis=1)

    @property
    def n_leaf_features_(self) -> int:
        return int(sum(len(c) for c in self.encoder_.categories_))

    def frozen_clone(self) -> "SOLACE":
        """Unfitted copy with the tuned hyper-parameters fixed (no further tuning)."""
        m = clone(self)
        return m.set_params(tune=False, optical_params=dict(self.optical_params_),
                            classifier_params=dict(self.classifier_params_),
                            final_params=dict(self.final_params_))
