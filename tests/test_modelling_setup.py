"""Focused checks for chronological boundaries and baseline probability metrics."""
import unittest
import numpy as np
import pandas as pd
from Python.model_selection import MonthlyTimeSeriesCV
from Python.modelling_setup import BaselineRunner, ModellingDataSetup


class SetupTests(unittest.TestCase):
    def test_month_boundaries_with_unsorted_rows(self):
        dates = pd.Series(pd.to_datetime(['2020-03-01', '2020-01-01', '2020-02-01',
                                         '2020-03-20', '2020-04-01', '2020-02-20']))
        splits = list(MonthlyTimeSeriesCV(2).split(np.zeros((6, 1)), groups=dates))
        for train, val in splits:
            self.assertLess(dates.iloc[train].max(), dates.iloc[val].min())
            self.assertFalse(set(dates.iloc[train].dt.to_period('M')) & set(dates.iloc[val].dt.to_period('M')))
        self.assertTrue(set(splits[0][0]).issubset(splits[1][0]))

    def test_late_initial_training_window(self):
        dates = pd.date_range('2013-01-01', '2014-12-01', freq='MS')
        splits = list(MonthlyTimeSeriesCV(5, min_train_end='2014-02').split(dates, groups=dates))
        self.assertEqual(len(splits), 5)
        self.assertEqual(dates[splits[0][0]].max(), pd.Timestamp('2014-02-01'))
        self.assertTrue(all(len(val) == 2 for _, val in splits))
        self.assertEqual(dates[splits[-1][1]].max(), pd.Timestamp('2014-12-01'))

    def test_invalid_dates_rejected(self):
        with self.assertRaises(ValueError):
            list(MonthlyTimeSeriesCV(2).split([0, 1, 2], groups=['2020-01-01', None, '2020-03-01']))

    def test_probability_auc_and_missing_class(self):
        runner = BaselineRunner()
        y = np.array([0, 0, 0, 1, 2])
        metrics = runner.metrics('classification', y, np.zeros(5), np.tile([.6, .2, .2], (5, 1)))
        self.assertAlmostEqual(metrics['auc_macro_ovr'], .5)
        self.assertAlmostEqual(metrics['accuracy'], metrics['f1_micro'])
        self.assertLess(metrics['f1_macro'], metrics['accuracy'])
        missing = runner.metrics('classification', y[:3], np.zeros(3), np.tile([.6, .2, .2], (3, 1)))
        self.assertTrue(np.isnan(missing['auc_macro_ovr']))

    def test_leaking_predictor_rejected(self):
        setup = ModellingDataSetup('.')
        setup.predictor_plan['regression']['numeric'].append('installment')
        with self.assertRaises(ValueError):
            setup.predictors('regression')


if __name__ == '__main__':
    unittest.main()
