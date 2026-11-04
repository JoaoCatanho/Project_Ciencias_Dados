"""Resampling provenance, feature availability, train-only bounds, and metrics."""
import unittest
import tempfile
import numpy as np
import pandas as pd
from Python.imbalance_diagnostics import (DiagnosticConfig, TrainingSampler,
                                         DiagnosticPreprocessor, ImbalanceMetrics, ImbalanceDiagnosticRunner)


class ImbalanceDiagnosticTests(unittest.TestCase):
    def test_sampling_preserves_every_minority_original_and_is_reproducible(self):
        y = np.array([0] * 1000 + [1] * 4 + [2] * 100)
        sampler = TrainingSampler()
        for policy in ['under', 'over', 'hybrid']:
            indices = sampler.indices(y, policy, 42)
            np.testing.assert_array_equal(indices, sampler.indices(y, policy, 42))
            self.assertTrue(set(np.flatnonzero(y != 0)).issubset(indices))
            self.assertTrue(np.all((indices >= 0) & (indices < len(y))))
            if policy in ('under', 'hybrid'):
                self.assertEqual(np.sum(y[indices] == 0), 200)
            if policy in ('over', 'hybrid'):
                self.assertLessEqual(np.sum(y[indices] == 1), 80)
        np.testing.assert_array_equal(sampler.indices(y, 'weighted', 42), np.arange(len(y)))

    @staticmethod
    def data():
        return pd.DataFrame({'annual_inc': [10., 20., 30., 40.],
            'annual_inc_joint': [np.nan, 200., np.nan, 400.],
            'application_type': ['Individual', 'Joint App', 'Joint App', 'Individual'],
            'loan_amnt': [10., 20., 30., 40.], 'dti': [1., 2., 3., 4.],
            'installment': [1., 2., 3., 4.], 'loan_to_income': [1.] * 4,
            'dti_x_loan_to_income': [1., 2., 3., 4.], 'installment_burden': [1.2] * 4})

    def test_joint_income_does_not_fill_missing_with_zero_or_use_nonapplicable_values(self):
        data = self.data()
        numeric = [c for c in data if c not in ['annual_inc_joint', 'application_type']]
        pre = DiagnosticPreprocessor(numeric, ['application_type'], [], 'joint')
        frame = pre.deterministic(data)
        self.assertAlmostEqual(frame.loan_to_income.iloc[0], 1.)
        self.assertAlmostEqual(frame.loan_to_income.iloc[1], .1)
        self.assertTrue(np.isnan(frame.loan_to_income.iloc[2]))
        self.assertAlmostEqual(frame.loan_to_income.iloc[3], 1.)
        self.assertNotIn('annual_inc_joint', frame)

    def test_clipping_statistics_fit_on_training_only_and_ablation_drops_engineered(self):
        data = self.data().iloc[[0, 1, 3]].copy()
        numeric = [c for c in data if c not in ['annual_inc_joint', 'application_type']]
        pre = DiagnosticPreprocessor(numeric, ['application_type'], ['loan_to_income'], 'joint_robust').fit(data)
        bounds = pre.bounds_.copy()
        validation = data.iloc[[0]].copy()
        validation.annual_inc = 1e12
        Z = pre.transform(validation)
        pd.testing.assert_frame_equal(pre.bounds_, bounds)
        self.assertTrue(np.isfinite(Z).all())
        self.assertLess(pre.bounds_.loc[.995, 'annual_inc'], 41)
        original = DiagnosticPreprocessor(numeric, ['application_type'], ['loan_to_income'], 'original')
        self.assertNotIn('loan_to_income', original.columns)

    def test_estimator_receives_normalized_resampling_weights_and_keeps_evaluation_size(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(64, 3))
        y = np.array([0] * 50 + [1] * 4 + [2] * 10)
        with tempfile.TemporaryDirectory() as out:
            runner = ImbalanceDiagnosticRunner('.', out, DiagnosticConfig(target_risk_share_nonrisk=.2))
            row, model, probability = runner._fit('logistic', 'over', 42, X, y, X, y)
        self.assertEqual(len(probability), len(y))
        self.assertEqual(row['n_train_original'], len(y))
        self.assertGreater(row['n_train_fitted'], len(y))
        self.assertAlmostEqual(row['uniform_weight_per_fitted_row'] * row['n_train_fitted'], len(y))
        self.assertIsNone(model.class_weight)

    def test_review_budget_and_threshold_rule(self):
        cfg = DiagnosticConfig(review_fraction=.25)
        y = np.array([0, 1, 2, 0])
        p = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .2, .7], [.6, .3, .1]])
        metrics = ImbalanceMetrics(cfg).score(y, p)
        self.assertEqual(metrics['risk_precision_at_budget'], 1.)
        self.assertEqual(metrics['risk_recall_at_budget'], 1.)
        self.assertEqual(metrics['ap_class_1'], 1.)
        np.testing.assert_array_equal(ImbalanceMetrics.threshold_prediction(p, .25), [0, 1, 2, 1])


if __name__ == '__main__':
    unittest.main()
