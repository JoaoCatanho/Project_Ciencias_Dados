"""Checks for bagging, boosting budgets, and three-class score alignment."""
import tempfile
import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from Python.ensemble_models import EnsembleExperimentConfig, EnsembleModelFactory, EnsembleExperimentRunner


class EnsembleTests(unittest.TestCase):
    def test_required_families_and_declared_search_budget(self):
        factory = EnsembleModelFactory()
        for task, count, classes in [('regression', 6, (RandomForestRegressor, HistGradientBoostingRegressor)),
                                     ('classification', 8, (RandomForestClassifier, HistGradientBoostingClassifier))]:
            choices = factory.candidates(task)
            self.assertEqual(len(choices), count)
            self.assertEqual({name for name, _ in choices}, {'random_forest', 'hist_gradient_boosting'})
            for family, params in choices:
                model = factory.build(task, family, params)
                self.assertIsInstance(model, classes)
                if family == 'random_forest':
                    self.assertTrue(model.bootstrap)
                    self.assertEqual(model.max_samples, .5)
                else:
                    self.assertFalse(model.early_stopping)
        with self.assertRaises(ValueError):
            factory.build('regression', 'unknown', {})

    def test_full_row_classification_fit_and_normalized_probabilities(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(90, 3))
        y = np.repeat([0, 1, 2], 30)
        X[:, 0] += y * 2
        config = EnsembleExperimentConfig(forest_trees=3, forest_min_samples_leaf=2,
                                         boosting_iterations=4, boosting_min_samples_leaf=2)
        with tempfile.TemporaryDirectory() as out:
            runner = EnsembleExperimentRunner(None, out, config)
            for family, params in [('random_forest', {'max_depth': 3, 'class_weight': 'balanced'}),
                                   ('hist_gradient_boosting', {'learning_rate': .1, 'max_leaf_nodes': 7, 'class_weight': 'balanced'})]:
                row, model, prediction, prob = runner.fit_and_score('classification', family, params, X, y, X, y)
                self.assertEqual(row['n_train'], len(X))
                self.assertTrue(row['budget_completed'])
                self.assertEqual(prob.shape, (90, 3))
                np.testing.assert_allclose(prob.sum(axis=1), 1.)
                np.testing.assert_array_equal(prediction, model.classes_[prob.argmax(axis=1)])
                for label in [0, 1, 2]:
                    self.assertEqual(row[f'n_train_class_{label}'], 30)

    def test_regression_predictions_and_fixed_boosting_budget(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(80, 3))
        y = 12 + X[:, 0] * 3
        config = EnsembleExperimentConfig(forest_trees=3, forest_min_samples_leaf=2,
                                         boosting_iterations=4, boosting_min_samples_leaf=2)
        with tempfile.TemporaryDirectory() as out:
            runner = EnsembleExperimentRunner(None, out, config)
            for family, params in [('random_forest', {'max_depth': 3}),
                                   ('hist_gradient_boosting', {'learning_rate': .1, 'max_leaf_nodes': 7})]:
                row, model, prediction, prob = runner.fit_and_score('regression', family, params, X, y, X, y)
                self.assertTrue(row['budget_completed'])
                self.assertTrue(np.isfinite(prediction).all())
                self.assertIsNone(prob)
                self.assertLess(row['rmse'], np.std(y))


if __name__ == '__main__':
    unittest.main()
