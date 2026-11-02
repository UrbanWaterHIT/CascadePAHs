"""Quantile regression forest (Meinshausen, 2006).

A drop-in subclass of ``RandomForestRegressor``: ``predict`` returns the usual
conditional mean, ``predict_quantiles`` returns conditional quantiles obtained
from the forest weights w_i(x) = mean_t 1[x and x_i share a leaf in tree t] / |leaf|.
Works with dense or sparse (CSR) input.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor


class QuantileRandomForestRegressor(RandomForestRegressor):
    """Random forest that additionally stores the training targets and leaves."""

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        y = np.asarray(y, dtype=float).ravel()
        order = np.argsort(y, kind="mergesort")
        self._qrf_y_sorted = y[order]
        leaves = self.apply(X)[order]                     # (n_train, n_trees), rows sorted by y
        self._qrf_n_train = leaves.shape[0]
        # One sparse (n_train x n_leaves_t) normalised membership matrix per tree
        self._qrf_maps = []
        for t in range(leaves.shape[1]):
            uniq, inv = np.unique(leaves[:, t], return_inverse=True)
            counts = np.bincount(inv).astype(float)
            M = sp.csr_matrix((1.0 / counts[inv], (np.arange(len(inv)), inv)), shape=(len(inv), len(uniq)))
            self._qrf_maps.append((uniq, M.T.tocsr()))   # (n_leaves_t x n_train)
        return self

    def _weights(self, X) -> np.ndarray:
        leaves = self.apply(X)                            # (m, n_trees)
        m = leaves.shape[0]
        W = np.zeros((m, self._qrf_n_train))
        for t, (uniq, MT) in enumerate(self._qrf_maps):
            pos = np.searchsorted(uniq, leaves[:, t])
            pos = np.clip(pos, 0, len(uniq) - 1)
            hit = uniq[pos] == leaves[:, t]
            rows = MT[pos]                                # (m x n_train) sparse
            if not hit.all():                             # leaf without training samples (cannot occur for RF)
                rows = sp.diags(hit.astype(float)) @ rows
            W += rows.toarray()
        W /= len(self._qrf_maps)
        return W

    def predict_quantiles(self, X, quantiles=(0.05, 0.95), batch_size: int = 2000) -> np.ndarray:
        """Return an array of shape (n_samples, len(quantiles))."""
        quantiles = np.atleast_1d(quantiles)
        n = X.shape[0]
        out = np.empty((n, len(quantiles)))
        for start in range(0, n, batch_size):
            sl = slice(start, min(n, start + batch_size))
            W = self._weights(X[sl])
            cdf = np.cumsum(W, axis=1)
            cdf /= cdf[:, -1:]
            for j, q in enumerate(quantiles):
                idx = (cdf < q).sum(axis=1)
                idx = np.clip(idx, 0, self._qrf_n_train - 1)
                out[sl, j] = self._qrf_y_sorted[idx]
        return out
