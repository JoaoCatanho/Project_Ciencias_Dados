"""Independent brute-force checks, edge cases, and sklearn pipeline integration."""
import unittest
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from Python.knn import KNNClassifierScratch, KNNRegressorScratch


class KNNTests(unittest.TestCase):
    def test_neighbors_match_independent_direct_distance(self):
        rng = np.random.default_rng(42)
        X, Q = rng.normal(size=(37, 4)), rng.normal(size=(11, 4))
        model = KNNRegressorScratch(n_neighbors=7, batch_size=3).fit(X, np.arange(37))
        distance, indices = model.kneighbors(Q)
        reference = np.linalg.norm(Q[:, None, :] - X[None, :, :], axis=2)
        expected = np.argsort(reference, axis=1, kind='stable')[:, :7]
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_allclose(distance, np.take_along_axis(reference, expected, axis=1))
        np.testing.assert_allclose(model.predict(Q), expected.mean(axis=1))

    def test_exact_duplicates_and_distance_weighting(self):
        X, y = [[1.], [1.], [3.]], [2., 4., 100.]
        model = KNNRegressorScratch(3, 'distance').fit(X, y)
        self.assertAlmostEqual(model.predict([[1.]])[0], 3.)
        classifier = KNNClassifierScratch(3, 'distance').fit(X, [0, 2, 2])
        np.testing.assert_allclose(classifier.predict_proba([[1.]]), [[.5, .5]])
        self.assertEqual(classifier.predict([[1.]])[0], 0)

    def test_boundary_and_vote_ties(self):
        model = KNNClassifierScratch(2).fit([[-1.], [1.], [-1.], [1.]], [2, 0, 2, 2])
        np.testing.assert_array_equal(model.kneighbors([[0.]])[1], [[0, 1]])
        self.assertEqual(model.predict([[0.]])[0], 0)

    def test_invalid_inputs_and_pipeline(self):
        with self.assertRaises(ValueError):
            KNNRegressorScratch(3).fit([[0.], [1.]], [0, 1])
        with self.assertRaises(ValueError):
            KNNClassifierScratch(1).fit([[np.nan]], [0])
        pipe = make_pipeline(StandardScaler(), KNNClassifierScratch(1))
        cloned = clone(pipe).fit([[0.], [2.]], [0, 2])
        np.testing.assert_array_equal(cloned.predict([[0.], [2.]]), [0, 2])
        with self.assertRaises(ValueError):
            cloned[-1].predict([[0., 1.]])

    def test_weighted_averaging_and_class_probabilities(self):
        model = KNNRegressorScratch(2, 'distance').fit([[0.], [4.]], [10., 30.])
        self.assertAlmostEqual(model.predict([[1.]])[0], 15.)
        classifier = KNNClassifierScratch(2, 'distance').fit([[0.], [4.]], [0, 2])
        np.testing.assert_allclose(classifier.predict_proba([[1.]]), [[.75, .25]])


if __name__ == '__main__':
    unittest.main()
