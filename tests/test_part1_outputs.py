"""Integration checks for artifacts produced by the corrected Part 1 notebooks.

Run after executing notebooks 01 through 05b in order. These checks do not inspect
held-out outcomes: they verify development boundaries and deterministic encoding.
"""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs' / 'tables'
PROCESSED = ROOT / 'data' / 'processed'


@unittest.skipUnless((TABLES / 'hypothesis_development_scope.csv').exists(),
                     'Execute the corrected notebooks to generate integration artifacts')
class Part1OutputTests(unittest.TestCase):
    def test_all_notebook_cells_executed_without_errors(self):
        for path in (ROOT / 'notebooks').glob('*.ipynb'):
            cells = json.loads(path.read_text())['cells']
            for index, cell in enumerate(cells):
                if cell['cell_type'] != 'code':
                    continue
                self.assertIsNotNone(cell['execution_count'], (path.name, index))
                self.assertFalse([o for o in cell['outputs'] if o['output_type'] == 'error'],
                                 (path.name, index))

    def test_development_scope_and_funded_population(self):
        scope = pd.read_csv(TABLES / 'hypothesis_development_scope.csv').set_index('cohort')
        self.assertTrue((pd.to_datetime(scope.max_date).dt.to_period('M') <= pd.Period('2016-12')).all())
        splits = json.loads((ROOT / 'outputs' / 'split_ids.json').read_text())
        for cohort, task in [('funded', 'regression'), ('classification', 'classification')]:
            self.assertEqual(int(scope.loc[cohort, 'n']), splits[task]['n_train'] + splits[task]['n_val'])
        self.assertGreater(int(scope.loc['funded', 'n']), int(scope.loc['classification', 'n']))
        feature_scope = pd.read_csv(TABLES / 'feature_development_scope.csv')
        self.assertTrue((pd.to_datetime(feature_scope.max_date).dt.to_period('M') <= pd.Period('2016-12')).all())
        self.assertEqual(int(feature_scope.n.iloc[0]), int(scope.loc['classification', 'n']))
        proportions = pd.read_csv(TABLES / 'hypothesis_state_approval_ratio.csv')
        np.testing.assert_allclose(proportions.funded_proportion,
                                   proportions.n_accepted / (proportions.n_accepted + proportions.n_rejected))

    def test_feature_specification_and_actual_missingness(self):
        spec = pd.read_csv(TABLES / 'feature_engineering_spec.csv')
        self.assertGreaterEqual(int(spec.eligible_regression.sum()), 10)
        reg_columns = pq.ParquetFile(PROCESSED / 'accepted_regression_fe.parquet').schema_arrow.names
        self.assertFalse({'credit_hist_months', 'credit_age_ratio', 'inq_per_year',
                          'installment_burden', 'rate_burden'} & set(reg_columns))
        self.assertTrue(set(spec.loc[spec.eligible_regression, 'name']).issubset(reg_columns))
        plan = json.loads((ROOT / 'outputs' / 'predictor_plan.json').read_text())
        reg_predictors = set(plan['regression']['numeric'] + plan['regression']['categorical'])
        self.assertFalse({'issue_d', 'credit_hist_months', 'verification_status',
                          'int_rate', 'installment', 'grade', 'sub_grade'} & reg_predictors)

        cols = ['emp_length', 'emp_length_years', 'pub_rec_bankruptcies', 'has_bankruptcy',
                'inquiries_per_open_account', 'closed_credit_share']
        df = pd.read_parquet(PROCESSED / 'accepted_classification_fe.parquet', columns=cols)
        self.assertTrue(df.loc[df.emp_length.eq('< 1 year'), 'emp_length_years'].eq(0.5).all())
        self.assertTrue(df.loc[df.emp_length.eq('1 year'), 'emp_length_years'].eq(1.).all())
        self.assertTrue(df.loc[df.emp_length.eq('10+ years'), 'emp_length_years'].eq(10.).all())
        self.assertTrue(df.has_bankruptcy.isna().equals(df.pub_rec_bankruptcies.isna()))
        self.assertFalse(np.isinf(df[['inquiries_per_open_account', 'closed_credit_share']]).any().any())
        self.assertTrue(df.closed_credit_share.dropna().between(0, 1).all())

    def test_projection_samples_belong_to_expected_periods(self):
        for name, source, lower, upper in [
            ('umap_classification_sample', 'accepted_classification', None, '2014-12'),
            ('umap_validation_sample', 'accepted_classification', '2015-01', '2016-12'),
            ('pca_validation_sample', 'accepted_regression', '2015-01', '2016-12'),
        ]:
            embedding = pd.read_parquet(PROCESSED / (name + '.parquet'))
            dates = pd.read_parquet(PROCESSED / (source + '.parquet'), columns=['issue_d'])
            months = dates.loc[embedding.index, 'issue_d'].dt.to_period('M')
            self.assertTrue(months.le(pd.Period(upper)).all(), name)
            if lower:
                self.assertTrue(months.ge(pd.Period(lower)).all(), name)
            self.assertGreater(len(embedding), 0)
            self.assertTrue(np.isfinite(embedding.select_dtypes('number')).all().all())

    def test_hypothesis_outputs_and_cleaned_dti(self):
        results = pd.read_csv(TABLES / 'hypothesis_results.csv')
        self.assertEqual(len(results), 6)
        self.assertTrue(results.p_value.between(0, 1).all())
        cliffs = results.loc[results.effect_size_name.eq("Cliff's delta")]
        self.assertTrue(cliffs.effect_size.between(-1, 1).all())
        self.assertTrue(cliffs.ci_low.le(cliffs.ci_high).all())
        frame = pd.read_parquet(PROCESSED / 'accepted_regression.parquet', columns=['dti'])
        self.assertFalse(frame.dti.lt(0).any())
        self.assertFalse(frame.dti.eq(999).any())


if __name__ == '__main__':
    unittest.main()
