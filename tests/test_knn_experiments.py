"""Check reproducible rare-class sampling and fold-local preprocessing."""
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from Python.knn_experiments import KNNExperimentConfig, KNNExperimentRunner


class ExperimentTests(unittest.TestCase):
    def test_sampling_keeps_rare_class_without_duplicates(self):
        data = pd.DataFrame({'target_class': [0] * 900 + [1] * 2 + [2] * 98})
        with tempfile.TemporaryDirectory() as out:
            runner = KNNExperimentRunner(None, out, KNNExperimentConfig(training_cap=100))
            a = runner.sample_indices(data, 'classification', 42)
            b = runner.sample_indices(data, 'classification', 42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 100)
        self.assertEqual(len(np.unique(a)), 100)
        self.assertEqual(set(data.iloc[a].target_class), {0, 1, 2})
        self.assertLess((data.iloc[a].target_class == 1).sum(), 5)

    def test_preprocessing_learns_only_training_and_prefix_reuse(self):
        setup = SimpleNamespace(predictor_plan={'regression': {'numeric': ['value'], 'categorical': []}},
                                predictors=lambda task: ['value'])
        training = pd.DataFrame({'value': [0., 2., 4., 6.], 'int_rate': [1., 3., 5., 7.]})
        evaluation = pd.DataFrame({'value': [10000., np.nan], 'int_rate': [8., 4.]})
        with tempfile.TemporaryDirectory() as out:
            runner = KNNExperimentRunner(setup, out, KNNExperimentConfig(training_cap=4, neighbors=(1, 3)))
            _, model, pre, predicted = runner._fit_score(
                'regression', training, evaluation, 'cv_1', [(1, 'uniform'), (3, 'distance')])
        numeric = pre._pipeline.named_steps['pre'].named_transformers_['num']
        self.assertAlmostEqual(numeric.named_steps['imputer'].statistics_[0], 3.)
        self.assertAlmostEqual(numeric.named_steps['scaler'].mean_[0], 3.)
        model.n_neighbors, model.weights = 1, 'uniform'
        np.testing.assert_allclose(model.predict(pre.transform(evaluation[['value']])), predicted[(1, 'uniform')])
        model.n_neighbors, model.weights = 3, 'distance'
        np.testing.assert_allclose(model.predict(pre.transform(evaluation[['value']])), predicted[(3, 'distance')])


if __name__ == '__main__':
    unittest.main()
