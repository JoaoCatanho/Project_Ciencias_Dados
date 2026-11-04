"""Check fold-local transforms, full-row fits, selection, and model restrictions."""
import json
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from Python.modelling_setup import ExperimentConfig
from Python.supervised_models import SupervisedModelFactory, SupervisedExperimentRunner


class SupervisedTests(unittest.TestCase):
    def test_factory_contains_two_nonensemble_models_per_task(self):
        factory = SupervisedModelFactory()
        for task, allowed, count in [('regression', (Ridge, DecisionTreeRegressor), 9),
                                     ('classification', (LogisticRegression, DecisionTreeClassifier), 12)]:
            choices = factory.candidates(task)
            self.assertEqual(len(choices), count)
            for family, params in choices:
                self.assertIsInstance(factory.build(task, family, params), allowed)
        with self.assertRaises(ValueError):
            factory.build('regression', 'logistic', {})

    def test_preprocessing_ignores_validation_statistics(self):
        setup = SimpleNamespace(predictor_plan={'regression': {'numeric': ['value'], 'categorical': ['category']}},
                                predictors=lambda task: ['value', 'category'])
        train = pd.DataFrame({'value': [0., 2., np.nan, 4.], 'category': ['a', 'b', 'a', 'b']})
        val = pd.DataFrame({'value': [10000., np.nan], 'category': ['new', 'a']})
        with tempfile.TemporaryDirectory() as out:
            pre, X, Z, _ = SupervisedExperimentRunner(setup, out).prepare('regression', train, val)
        numeric = pre._pipeline.named_steps['pre'].named_transformers_['num']
        self.assertEqual(X.shape[0], 4)
        self.assertEqual(Z.shape[0], 2)
        self.assertEqual(X.shape[1], Z.shape[1])
        self.assertAlmostEqual(numeric.named_steps['imputer'].statistics_[0], 2.)
        self.assertAlmostEqual(numeric.named_steps['scaler'].mean_[0], 2.)
        self.assertTrue(np.isfinite(Z).all())

    def test_selection_excludes_incomplete_or_unconverged_candidates(self):
        records = []
        for alpha, score, converged, folds in [(0.1, 3., True, 2), (10., 1., False, 2), (1000., .5, True, 1)]:
            for _ in range(folds):
                records.append(dict(model='ridge', parameters=json.dumps({'alpha': alpha}),
                                    rmse=score, converged=converged, fit_seconds=.1))
        with tempfile.TemporaryDirectory() as out:
            runner = SupervisedExperimentRunner(None, out, protocol=ExperimentConfig(n_splits=2))
            _, selection = runner.select('regression', pd.DataFrame(records))
        self.assertEqual(selection['ridge'], {'alpha': .1})

    def test_full_training_fit_and_three_class_probabilities(self):
        X = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
        y = np.array([0, 0, 1, 1, 2, 2])
        with tempfile.TemporaryDirectory() as out:
            row, model, prediction, prob = SupervisedExperimentRunner(None, out).fit_and_score(
                'classification', 'logistic', {'C': 1., 'class_weight': 'balanced'}, X, y, X, y)
        self.assertEqual(row['n_train'], len(X))
        self.assertTrue(row['converged'])
        self.assertEqual(prob.shape, (6, 3))
        np.testing.assert_allclose(prob.sum(axis=1), 1.)
        np.testing.assert_array_equal(prediction, model.classes_[prob.argmax(axis=1)])
        self.assertGreater(row['f1_macro'], .5)


if __name__ == '__main__':
    unittest.main()
