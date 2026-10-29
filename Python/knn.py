"""Exact kNN implemented with NumPy, without a library neighbour estimator.

Training examples are stored as contiguous arrays. Queries use a bounded batch
of squared Euclidean distances and a top-k index table selected by argpartition.
No query-by-training matrix for the entire evaluation cohort is materialized.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _KNNBase(BaseEstimator):
    def __init__(self, n_neighbors=15, weights='uniform', batch_size=256):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.batch_size = batch_size

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        if not isinstance(self.n_neighbors, Integral) or not 1 <= self.n_neighbors <= len(X):
            raise ValueError('n_neighbors must be an integer between 1 and training size.')
        if self.weights not in ('uniform', 'distance'):
            raise ValueError('weights must be uniform or distance.')
        if not isinstance(self.batch_size, Integral) or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        self.X_ = np.array(X, dtype=np.float64, order='C', copy=True)
        self.y_ = np.array(y, copy=True)
        self.n_features_in_ = X.shape[1]
        self.squared_norms_ = np.einsum('ij,ij->i', self.X_, self.X_)
        return self

    def iter_neighbors(self, X):
        """Yield (query slice, distances, indices), sorted by distance then row.

        Exact distance ties at the kth boundary select smaller fitted row indices.
        The norm identity avoids a batch × training × features temporary array.
        Near-zero distances are recomputed directly to avoid cancellation for
        identical observations (important for inverse-distance weighting).
        """
        check_is_fitted(self, 'X_')
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Query feature count differs from fitted data.')
        k = self.n_neighbors
        for start in range(0, len(X), self.batch_size):
            query = X[start:start + self.batch_size]
            qnorm = np.einsum('ij,ij->i', query, query)
            squared = qnorm[:, None] + self.squared_norms_[None, :] - 2 * query @ self.X_.T
            np.maximum(squared, 0, out=squared)
            candidates = np.argpartition(squared, k - 1, axis=1)[:, :k]
            values = np.take_along_axis(squared, candidates, axis=1)
            for row in range(len(query)):
                boundary = values[row].max()
                ties = np.flatnonzero(squared[row] == boundary)
                if len(ties) > 1:
                    below = np.flatnonzero(squared[row] < boundary)
                    candidates[row] = np.concatenate([below, ties[:k - len(below)]])
                near = np.flatnonzero(squared[row, candidates[row]] < 1e-10)
                if len(near):
                    delta = self.X_[candidates[row, near]] - query[row]
                    squared[row, candidates[row, near]] = np.einsum('ij,ij->i', delta, delta)
                order = np.lexsort((candidates[row], squared[row, candidates[row]]))
                candidates[row] = candidates[row, order]
            distances = np.sqrt(np.take_along_axis(squared, candidates, axis=1))
            yield slice(start, start + len(query)), distances, candidates

    def kneighbors(self, X, return_distance=True):
        """Materialize only the query × k output, never all query distances."""
        batches = list(self.iter_neighbors(X))
        distances = np.concatenate([d for _, d, _ in batches])
        indices = np.concatenate([i for _, _, i in batches])
        return (distances, indices) if return_distance else indices

    @staticmethod
    def neighbor_weights(distances, weights):
        if weights == 'uniform':
            return np.ones_like(distances)
        if weights != 'distance':
            raise ValueError('weights must be uniform or distance.')
        zero = distances == 0
        has_zero = zero.any(axis=1)
        result = np.divide(1., distances, out=np.zeros_like(distances), where=~zero)
        result[has_zero] = zero[has_zero]
        return result


class KNNRegressorScratch(RegressorMixin, _KNNBase):
    """Uniform or inverse-distance average of continuous neighbour targets."""

    def fit(self, X, y):
        super().fit(X, y)
        self.y_ = self.y_.astype(float)
        if not np.isfinite(self.y_).all():
            raise ValueError('Regression targets must be finite.')
        return self

    def predict_from_neighbors(self, distances, indices, n_neighbors=None, weights=None):
        k = n_neighbors or self.n_neighbors
        weight = self.neighbor_weights(distances[:, :k], weights or self.weights)
        return (weight * self.y_[indices[:, :k]]).sum(axis=1) / weight.sum(axis=1)

    def predict(self, X):
        check_is_fitted(self, 'X_')
        return np.concatenate([self.predict_from_neighbors(d, i) for _, d, i in self.iter_neighbors(X)])


class KNNClassifierScratch(ClassifierMixin, _KNNBase):
    """Neighbour vote fractions; equal votes select the smallest sorted class.

    If a query matches fitted rows exactly, distance weighting uses only those
    zero-distance neighbours. Probabilities are local vote fractions, not
    calibrated estimates of future default risk.
    """

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_, self.encoded_y_ = np.unique(self.y_, return_inverse=True)
        return self

    def predict_proba_from_neighbors(self, distances, indices, n_neighbors=None, weights=None):
        k = n_neighbors or self.n_neighbors
        weight = self.neighbor_weights(distances[:, :k], weights or self.weights)
        labels = self.encoded_y_[indices[:, :k]]
        votes = np.stack([(weight * (labels == c)).sum(axis=1)
                          for c in range(len(self.classes_))], axis=1)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        check_is_fitted(self, 'classes_')
        return np.concatenate([self.predict_proba_from_neighbors(d, i)
                               for _, d, i in self.iter_neighbors(X)])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
