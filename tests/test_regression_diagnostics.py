import unittest
import pandas as pd
from Python.regression_diagnostics import RegressionDiagnostics


class RegressionDiagnosticTests(unittest.TestCase):
    def test_extreme_error_concentration_and_perfect_predictions(self):
        frame = pd.DataFrame({'int_rate': [1., 2., 3.], 'prediction': [1., 2., 103.]})
        summary = RegressionDiagnostics.extreme_error_summary(frame)
        self.assertAlmostEqual(summary['mse_share_top_1'], 1.)
        self.assertEqual(summary['median_absolute_error'], 0.)
        frame.prediction = frame.int_rate
        self.assertEqual(RegressionDiagnostics.extreme_error_summary(frame)['mse_share_top_1'], 0.)


if __name__ == '__main__':
    unittest.main()
