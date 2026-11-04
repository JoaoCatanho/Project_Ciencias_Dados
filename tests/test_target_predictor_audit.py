"""Leakage aliases, scenario boundaries, and CV-only review constraints."""
import tempfile
import unittest
import numpy as np
import pandas as pd
from Python.target_predictor_audit import AuditConfig, PredictorAuditExperiment, ReviewOperatingPoints
from Python.preprocessing import LendingClubPreprocessor, leakage_columns


class TargetAuditTests(unittest.TestCase):
    def test_postorigination_aliases_and_hardship_auxiliaries_are_excluded(self):
        names = ['last_pymnt_amnt', 'deferral_term', 'payment_plan_start_date',
                 'orig_projected_additional_accrued_interest', 'annual_inc']
        self.assertEqual(set(leakage_columns(pd.DataFrame(columns=names))), set(names[:-1]))
        pre = LendingClubPreprocessor([], [], task='regression')
        self.assertNotIn('grade', pre.drop_leakage(pd.DataFrame(columns=['grade', 'annual_inc'])))

    def test_feature_scenarios_are_explicit_and_do_not_mutate_prior_plan(self):
        with tempfile.TemporaryDirectory() as out:
            experiment = PredictorAuditExperiment('.', out)
            original = experiment.setup.predictor_plan['classification']['categorical'].copy()
            numeric, categorical = experiment.columns('pricing_bureau')
            self.assertTrue({'grade', 'sub_grade'}.issubset(categorical))
            self.assertTrue({'tax_liens', 'collections_12_mths_ex_med', 'acc_now_delinq'}.issubset(numeric))
            self.assertEqual(original, experiment.setup.predictor_plan['classification']['categorical'])
            self.assertFalse({'loan_status', 'last_pymnt_amnt', 'target_class'} & set(numeric + categorical))

    def test_precision_goals_can_be_infeasible_without_relaxing_constraints(self):
        oof = pd.DataFrame({'fold': np.repeat([1, 2], 100),
                            'target_class': ([1] + [0] * 99) * 2,
                            'probability_1': [.1] * 200})
        chosen, _ = ReviewOperatingPoints().select(oof, 1)
        self.assertFalse(chosen.feasible.any())
        self.assertTrue(chosen.threshold.isna().all())

    def test_thresholds_meet_precision_and_support_in_every_training_fold(self):
        y = ([1] * 30 + [0] * 70) * 2
        score = ([.8] * 30 + [.1] * 70) * 2
        oof = pd.DataFrame({'fold': np.repeat([1, 2], 100), 'target_class': y, 'probability_1': score})
        chosen, _ = ReviewOperatingPoints().select(oof, 1)
        self.assertTrue(chosen.feasible.all())
        self.assertTrue(chosen.minimum_fold_alerts.ge(20).all())
        self.assertTrue(chosen.minimum_fold_precision.ge(chosen.precision_floor).all())
        self.assertTrue(chosen.recall.eq(1).all())

    def test_fixed_budget_is_deterministic_and_does_not_relabel_targets(self):
        y = np.array([0, 1, 2, 0])
        original = y.copy()
        score = np.array([.1, .9, .2, .3])
        op = ReviewOperatingPoints()
        row = op.review_budget(y, score, 1, .25)
        self.assertEqual(row['alerts'], 1)
        self.assertEqual(row['precision'], 1)
        self.assertEqual(row['recall'], 1)
        np.testing.assert_array_equal(y, original)


if __name__ == '__main__':
    unittest.main()
