"""Regression checks for the object-oriented Part 1 analysis components."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats

from src import ChronologicalSplitter, FeatureValidator, HypothesisTester, LendingClubEDA


class AnalysisComponentTests(unittest.TestCase):
    def test_split_boundaries_reuse_and_input_preservation(self):
        frame = pd.DataFrame({
            "issue_d": ["2014-12-01", "2014-12-31", "2015-01-01",
                        "2016-12-31", "2017-01-01", None],
            "amount": range(6),
        })
        original = frame.copy(deep=True)
        splitter = ChronologicalSplitter("issue_d", "2014-12", "2016-12")
        result = splitter.split(frame)
        self.assertEqual(result.train.index.tolist(), [0, 1])
        self.assertEqual(result.val.index.tolist(), [2, 3])
        self.assertEqual(result.test.index.tolist(), [4])
        self.assertEqual(result.cutoffs["test_start"], "2017-01-01T00:00:00")
        assert_frame_equal(frame, original)
        assert_frame_equal(splitter.split(frame).train, result.train)
        result.train.loc[0, "amount"] = -1
        assert_frame_equal(frame, original)

    def test_sampling_uses_instance_seed_without_advancing_random_state(self):
        frame = pd.DataFrame({"x": range(100)})
        eda = LendingClubEDA(seed=7)
        expected = np.sort(np.random.default_rng(7).choice(100, 12, replace=False))
        assert_frame_equal(eda.reproducible_sample(frame, 12), frame.iloc[expected])
        assert_frame_equal(eda.reproducible_sample(frame, 12), frame.iloc[expected])
        overridden = eda.reproducible_sample(frame, 12, seed=42)
        assert_frame_equal(overridden, LendingClubEDA().reproducible_sample(frame, 12))
        self.assertEqual(eda.seed, 7)

    def test_eda_statistics_and_output_files(self):
        frame = pd.DataFrame({"a": [1., 2., 3., 4., 100.], "b": [2., 4., 6., 8., 10.]})
        eda = LendingClubEDA()
        self.assertEqual(eda.iqr_outliers(frame.a), 1)
        summary = eda.describe_numeric(frame, ["a", "b", "absent"])
        self.assertEqual(summary.index.tolist(), ["a", "b"])
        self.assertEqual(summary.loc["a", "mean"], 22.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eda.plot_histograms(frame, ["a", "b"], root / "hist.png")
            eda.plot_scatter(frame, "a", "b", root / "scatter.png")
            corr = eda.plot_correlation_heatmap(frame, ["a", "b"], root / "corr.png")
            assert_frame_equal(corr, frame.corr(method="spearman"))
            info = eda.run_pca(frame, ["a", "b"], root, n_components=2)
            np.testing.assert_allclose(info["explained_variance_ratio"].sum(), 1.)
            for name in ["hist.png", "scatter.png", "corr.png", "pca_scree.png",
                         "pca_2d.png", "pca_loadings.csv"]:
                self.assertGreater((root / name).stat().st_size, 0)
            self.assertEqual(pd.read_csv(root / "pca_loadings.csv", index_col=0).index.tolist(), ["a", "b"])

    def test_hypothesis_statistics_and_result_reporting(self):
        tester = HypothesisTester()
        a, b = np.array([1., 2., 3., 4.]), np.array([3., 4., 5., 8.])
        for method, reference in [(tester.mannwhitney_test, stats.mannwhitneyu),
                                  (tester.welch_ttest, lambda x, y: stats.ttest_ind(x, y, equal_var=False))]:
            result = method(a, b, name="comparison", h0="same", h1="different", bootstrap=False)
            expected = reference(a, b)
            self.assertAlmostEqual(result.statistic, expected.statistic)
            self.assertAlmostEqual(result.p_value, expected.pvalue)
            self.assertIn("comparison", result.summary())
            table = tester.results_to_dataframe([result])
            self.assertEqual(table.iloc[0].to_dict(), result.to_dict())
        table = pd.DataFrame([[20, 10], [5, 25]])
        result = tester.chi_square_test(table, name="association", h0="independent", h1="associated")
        self.assertAlmostEqual(result.statistic, stats.chi2_contingency(table, correction=False).statistic)
        self.assertIsNone(tester.mannwhitney_test(a[:1], b, name="short", h0="same", h1="different"))

    def test_bootstrap_configuration_is_used_by_tests(self):
        a, b = np.arange(12.), np.arange(12.) + 2
        tester = HypothesisTester(seed=7, max_n_per_sample=5, n_boot=30, alpha=0.1)
        expected = HypothesisTester().bootstrap_diff_ci(
            a, b, seed=7, max_n_per_sample=5, n_boot=30, alpha=0.1,
        )
        self.assertEqual(tester.bootstrap_diff_ci(a, b), expected)
        result = tester.welch_ttest(a, b, name="comparison", h0="same", h1="different")
        self.assertEqual((result.ci_low, result.ci_high), expected)

    def test_feature_diagnostics_keep_schema_and_verdicts(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
        frame["engineered"] = frame.a * 2
        validator = FeatureValidator()
        for task, target in [("classification", pd.Series(np.tile([0, 1, 2], 20))),
                             ("regression", frame.b)]:
            result = validator.build_diagnostics(frame, ["engineered"], ["a", "b", "c"], target, task)
            self.assertEqual(result.columns.tolist(), [
                "name", "kind", "target_signal", "max_redundancy", "max_redundancy_with",
                "max_self_collinearity", "max_self_collinearity_with",
            ])
            feature = result.set_index("name").loc["engineered"]
            self.assertAlmostEqual(feature.max_redundancy, 1.)
            self.assertEqual(feature.max_redundancy_with, "a")
            self.assertEqual(validator.verdict(feature), "near-redundant — drop or keep only one")
        row = pd.Series({"max_redundancy": 0.5, "target_signal": 0.01})
        self.assertTrue(validator.verdict(row).startswith("keep"))
        self.assertTrue(validator.verdict_reg(row).startswith("weak signal"))
        self.assertTrue(FeatureValidator(threshold=0.4).verdict(row).startswith("near-redundant"))

    def test_notebook_cells_compile_and_project_imports_resolve(self):
        import ast
        import importlib
        root = Path(__file__).resolve().parents[1]
        for notebook in (root / "notebooks").glob("*.ipynb"):
            for index, cell in enumerate(json.loads(notebook.read_text())["cells"], 1):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                compile(source, f"{notebook.name}:cell{index}", "exec")
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                        module = importlib.import_module(node.module)
                        for name in node.names:
                            self.assertTrue(hasattr(module, name.name), (notebook.name, name.name))



class TemporalAndFeatureCorrectionTests(unittest.TestCase):
    def test_development_scope_uses_all_outcomes_and_excludes_future_dates(self):
        frame = pd.DataFrame({
            'issue_d': pd.to_datetime(['2014-01-01', '2016-12-31 12:00', '2017-01-01', None], format='mixed'),
            'loan_status': ['Current', 'Fully Paid', 'Charged Off', 'Current'],
        })
        policy = ChronologicalSplitter('issue_d', '2014-12', '2016-12')
        development = policy.development(frame)
        self.assertEqual(development.index.tolist(), [0, 1])
        rejected = frame.rename(columns={'issue_d': 'Application Date'})
        self.assertEqual(policy.development(rejected, 'Application Date').index.tolist(), [0, 1])

    def test_encoding_and_regression_eligibility(self):
        from src.feature_engineering import LendingClubFeatureEngineer
        frame = pd.DataFrame({
            'emp_length': ['< 1 year', '1 year', '10+ years', None],
            'pub_rec_bankruptcies': [np.nan, 0., 1., 2.],
            'issue_d': pd.to_datetime(['2018-01-01'] * 4),
            'earliest_cr_line': pd.to_datetime(['2000-01-01'] * 4),
            'inq_last_6mths': [2., 1., 0., np.nan], 'open_acc': [4., 0., 3., 2.],
            'total_acc': [8., 0., 2., 4.],
        })
        engineer = LendingClubFeatureEngineer()
        result = engineer.transform(frame, 'classification')
        np.testing.assert_allclose(result.emp_length_years, [0.5, 1., 10., np.nan], equal_nan=True)
        np.testing.assert_allclose(result.has_bankruptcy, [np.nan, 0., 1., 1.], equal_nan=True)
        self.assertEqual(result.inquiries_per_open_account.iloc[0], 0.5)
        self.assertTrue(np.isnan(result.inquiries_per_open_account.iloc[1]))
        self.assertEqual(result.closed_credit_share.iloc[0], 0.5)
        self.assertTrue(np.isnan(result.closed_credit_share.iloc[2]))
        regression = engineer.transform(frame, 'regression')
        self.assertFalse({'credit_hist_months', 'credit_age_ratio', 'inq_per_year',
                          'installment_burden', 'rate_burden'} & set(regression.columns))
        spec = engineer.specification_frame()
        self.assertGreaterEqual(int(spec.eligible_regression.sum()), 10)
        self.assertTrue(set(spec.loc[spec.eligible_regression, 'name']).issubset(regression.columns))
        self.assertNotIn('emp_length_years', frame.columns)
        nullable = frame.copy()
        nullable['emp_length'] = nullable.emp_length.astype('string')
        nullable_result = engineer.transform(nullable, 'classification')
        self.assertTrue(np.isnan(nullable_result.emp_length_years.iloc[-1]))


    def test_dti_sentinels_are_missing_but_legitimate_extremes_remain(self):
        from src.data_loader import LendingClubLoader
        loader = LendingClubLoader('unused', 'unused')
        frame = loader._parse_accepted(pd.DataFrame({'dti': [-1., 999., 45., 120.]}))
        np.testing.assert_allclose(frame.dti, [np.nan, np.nan, 45., 120.], equal_nan=True)
        rejected = loader._parse_rejected(pd.DataFrame({'Debt-To-Income Ratio': ['-1%', '10%', '120%']}))
        np.testing.assert_allclose(rejected['Debt-To-Income Ratio'], [np.nan, 10., 120.], equal_nan=True)

    def test_validation_pca_transform_does_not_refit(self):
        from unittest.mock import patch
        train = pd.DataFrame({'a': [0., 1., 2., 3.], 'b': [1., 3., 2., 5.]})
        validation = pd.DataFrame({'a': [1000., np.nan], 'b': [2000., 3.]}, index=[20, 21])
        eda = LendingClubEDA()
        with tempfile.TemporaryDirectory() as directory:
            eda.run_pca(train, ['a', 'b'], Path(directory), n_components=2)
            means = eda._pca_scaler.mean_.copy()
            with patch.object(eda._pca_scaler, 'fit', side_effect=AssertionError('refit')):
                with patch.object(eda._pca, 'fit', side_effect=AssertionError('refit')):
                    projected = eda.transform_pca(validation)
            self.assertEqual(projected.index.tolist(), [20])
            np.testing.assert_array_equal(means, eda._pca_scaler.mean_)
            np.testing.assert_allclose(means, train.mean().values)
        with self.assertRaises(RuntimeError):
            LendingClubEDA().transform_pca(validation)

    def test_histogram_log1p_accepts_zero(self):
        from unittest.mock import patch
        import matplotlib.axes
        seen = []
        original = matplotlib.axes.Axes.hist
        def capture(ax, values, *args, **kwargs):
            seen.append(np.asarray(values))
            return original(ax, values, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(matplotlib.axes.Axes, 'hist', capture):
                LendingClubEDA().plot_histograms(pd.DataFrame({'income': [0., 10., 100.]}),
                                                 ['income'], Path(directory) / 'hist.png',
                                                 log_scale_cols=['income'])
        np.testing.assert_allclose(seen[0], np.log1p([0., 10., 100.]))


if __name__ == "__main__":
    unittest.main()
