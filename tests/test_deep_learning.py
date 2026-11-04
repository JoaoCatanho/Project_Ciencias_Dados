"""Calendar isolation, finite conditioning, real learning, heads and checkpoints."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from Python.deep_learning_experiments import NeuralExperimentRunner
from Python.deep_learning import (NeuralConfig, NeuralPreprocessor, CalendarEarlyStoppingSplit,
                                  NeuralTrainer, TabularMLP)


class DeepLearningTests(unittest.TestCase):
    def test_calendar_stopping_is_inside_training_and_preserves_class_support(self):
        frame = pd.DataFrame({'issue_d': pd.to_datetime(['2014-01-01'] * 3 + ['2014-02-01'] * 3),
                              'target_class': [0, 1, 2, 0, 1, 2]})
        fit, stop = CalendarEarlyStoppingSplit().split(frame, 'classification')
        self.assertLess(fit.issue_d.max(), stop.issue_d.min())
        with self.assertRaises(ValueError):
            CalendarEarlyStoppingSplit().split(frame[frame.index != 1], 'classification')

    def test_numeric_bounds_ignore_extreme_validation_and_unknown_categories(self):
        train = pd.DataFrame({'annual_inc': [10., 20., 30., 40.], 'purpose': ['a', 'b', 'a', 'b']})
        pre = NeuralPreprocessor(['annual_inc'], ['purpose'], 'regression').fit(train)
        bounds = pre.bounds_.copy()
        val = pd.DataFrame({'annual_inc': [1e15, np.nan], 'purpose': ['new', 'a']})
        X = pre.transform(val)
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(X.dtype, np.float32)
        pd.testing.assert_frame_equal(bounds, pre.bounds_)
        self.assertLess(pre.bounds_.iloc[1].annual_inc, 41.)
        with self.assertRaises(ValueError):
            NeuralPreprocessor(['grade'], [], 'regression')

    def test_output_heads_and_balanced_weights_use_fitting_labels(self):
        c = NeuralConfig(hidden=(8, 4), dropout=0., max_epochs=3, batch_size=8)
        X = np.arange(18, dtype=np.float32).reshape(6, 3) / 10
        y = np.array([0, 0, 0, 1, 2, 2])
        trainer = NeuralTrainer('classification', class_weight='balanced', config=c).fit(X, y, fixed_epochs=2)
        np.testing.assert_allclose(trainer.class_weights_, [2/3, 2., 1.])
        prob = trainer.predict(X)
        self.assertEqual(prob.shape, (6, 3))
        np.testing.assert_allclose(prob.sum(axis=1), 1., atol=1e-6)
        self.assertEqual(TabularMLP(3, 'regression')(torch.zeros(2, 3)).shape, (2, 1))

    def test_regression_learns_and_safe_checkpoint_reproduces_predictions(self):
        X = np.linspace(-1, 1, 40, dtype=np.float32).reshape(-1, 1)
        y = 12 + 3 * X[:, 0]
        c = NeuralConfig(hidden=(8, 4), dropout=0., batch_size=10, max_epochs=40)
        trainer = NeuralTrainer('regression', learning_rate=.01, config=c).fit(X, y, fixed_epochs=30)
        pred = trainer.predict(X)
        self.assertLess(np.mean((pred - y) ** 2), np.var(y))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'; trainer.save(path)
            restored = NeuralTrainer.load(path)
            np.testing.assert_allclose(restored.predict(X), pred)

    def test_outer_refit_retains_all_classification_rows_after_inner_stopping(self):
        dates = pd.to_datetime(['2014-01-01'] * 6 + ['2014-02-01'] * 6 + ['2014-03-01'] * 6)
        frame = pd.DataFrame({'issue_d': dates, 'target_class': [0, 1, 2] * 6,
                              'loan_amnt': np.arange(18, dtype=float) + 1, 'purpose': ['a'] * 18})
        evaluation = frame.iloc[:6].copy()
        evaluation.issue_d = pd.Timestamp('2014-04-01')
        cfg = NeuralConfig(hidden=(4,), dropout=0., max_epochs=2, patience=1, batch_size=8)
        with tempfile.TemporaryDirectory() as out:
            runner = NeuralExperimentRunner('.', out, cfg)
            runner.setup.predictor_plan['classification'] = {'numeric': ['loan_amnt'], 'categorical': ['purpose']}
            row, bundle, prob = runner.fit_partition('classification', frame, evaluation,
                {'learning_rate': .001, 'class_weight': 'balanced'}, 42, 'unit')
        self.assertEqual(row['inner_fitted'], 12)
        self.assertEqual(row['n_stopping'], 6)
        self.assertEqual(row['n_train'], 18)
        self.assertEqual(row['n_train_class_1'], 6)
        self.assertEqual(prob.shape, (6, 3))
        self.assertLess(pd.Timestamp(row['inner_end']), pd.Timestamp(row['stopping_start']))

    def test_seed_reproducibility_and_actual_early_stopping(self):
        X = np.arange(24, dtype=np.float32).reshape(8, 3) / 20
        y = 10 + X[:, 0]
        c = NeuralConfig(hidden=(4,), dropout=.1, batch_size=4, max_epochs=10, patience=2, min_delta=1e9)
        a = NeuralTrainer('regression', seed=42, config=c).fit(X, y, stopping=(X, y))
        b = NeuralTrainer('regression', seed=42, config=c).fit(X, y, stopping=(X, y))
        self.assertEqual(a.best_epoch_, 1)
        self.assertEqual(len(a.history_), 3)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))


if __name__ == '__main__':
    unittest.main()
