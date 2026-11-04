"""Development-only target/predictor audit and class-specific review experiments.

Audit status/follow-up fields are never estimator inputs. Grades are allowed only
for the after-pricing classifier. Three additional bureau counts are tested as
an explicitly conditional original-bureau-snapshot scenario, not certified new
production inputs. Historical preprocessing plans and results remain unchanged.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, precision_recall_curve
from threadpoolctl import threadpool_limits

from .ensemble_models import EnsembleModelFactory
from .imbalance_diagnostics import ImbalanceMetrics
from .model_selection import MonthlyTimeSeriesCV
from .modelling_setup import ModellingDataSetup, ExperimentConfig
from .preprocessing import LendingClubPreprocessor, derive_classification_target, leakage_columns
from .supervised_models import SupervisedModelFactory

ISSUER_SOURCE = 'https://www.sec.gov/Archives/edgar/data/1409970/000119312512325943/d353028d424b3.htm'
BUREAU_COLUMNS = ('tax_liens', 'collections_12_mths_ex_med', 'acc_now_delinq')
PRICING_COLUMNS = ('grade', 'sub_grade')
AUDIT_DATES = ('last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d')


@dataclass
class AuditConfig:
    variants: tuple = ('current', 'pricing_grade', 'pricing_bureau')
    families: tuple = ('logistic', 'boosting')
    seed: int = 42
    threads: int = 2
    precision_floors: tuple = (0.05, 0.10, 0.20)
    minimum_alerts_per_fold: int = 20
    review_fractions: tuple = (0.01, 0.05)
    cutoff_scenarios: tuple = ('2018-12-31', '2019-12-31')

    def to_dict(self):
        return asdict(self)


class TargetPredictorAudit:
    def __init__(self, root, output_dir, config=None):
        self.root, self.output = Path(root), Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config or AuditConfig()
        self.setup = ModellingDataSetup(root)

    def run(self):
        """Audit the development source including Current/unknown outcomes.

        accepted_regression_fe retains statuses dropped from classification but
        itself requires non-null int_rate; matching known-outcome counts below
        verifies its coverage for this classification audit.
        """
        path = self.root / 'data/processed/accepted_regression_fe.parquet'
        schema = pq.read_schema(path)
        cols = ['issue_d', 'term', 'loan_status', 'application_type'] + list(AUDIT_DATES)
        data = pd.read_parquet(path, columns=cols, filters=[('issue_d', '<', pd.Timestamp('2017-01-01'))])
        data = data.sort_values('issue_d', kind='stable').reset_index(drop=True)
        data['target_class'] = derive_classification_target(data.loan_status)
        data['partition'] = np.where(data.issue_d < pd.Timestamp('2015-01-01'), 'train', 'validation')
        data['month'] = data.issue_d.dt.to_period('M').astype(str)
        data['year'] = data.issue_d.dt.year
        expected = json.loads((self.root / 'outputs/split_ids.json').read_text())['classification']
        assert data.loc[data.partition.eq('train'), 'target_class'].notna().sum() == expected['n_train']
        assert data.loc[data.partition.eq('validation'), 'target_class'].notna().sum() == expected['n_val']
        # Keep all source statuses, including excluded outcomes, in denominators.
        for keys, name in [(['partition', 'month', 'term', 'loan_status'], 'status_month_term'),
                           (['partition', 'year', 'term', 'loan_status'], 'status_year_term')]:
            table = data.groupby(keys, dropna=False).size().rename('n').reset_index()
            denominator = table.groupby(keys[:-1], dropna=False).n.transform('sum')
            table['share_of_source_cohort'] = table.n / denominator
            table.to_csv(self.output / f'{name}.csv', index=False)
        selected = data[data.target_class.notna()]
        counts = selected.groupby(['partition', 'term', 'target_class']).size().unstack(fill_value=0)
        counts.columns = ['Good', 'At risk', 'Default']
        counts.reset_index().to_csv(self.output / 'class_term_counts.csv', index=False)
        eligibility = data.groupby(['partition', 'year', 'term'], dropna=False).agg(
            source_n=('loan_status', 'size'), eligible_n=('target_class', 'count'))
        eligibility['excluded_share'] = 1 - eligibility.eligible_n / eligibility.source_n
        eligibility.reset_index().to_csv(self.output / 'outcome_exclusion_by_vintage.csv', index=False)
        date_rows = []
        for col in AUDIT_DATES:
            values = pd.to_datetime(data[col], errors='coerce')
            date_rows.append(dict(field=col, n_observed=int(values.notna().sum()),
                                  earliest=str(values.min()), latest=str(values.max()), role='audit only; not observation cutoff'))
        pd.DataFrame(date_rows).to_csv(self.output / 'event_date_audit.csv', index=False)
        scenarios = []
        # Calendar-month maturity, not actual repayment/outcome observation time.
        scheduled = data.issue_d.dt.year * 12 + data.issue_d.dt.month + data.term
        for scenario in self.config.cutoff_scenarios:
            end = pd.Timestamp(scenario)
            matured = scheduled <= end.year * 12 + end.month
            frame = data[['partition', 'term', 'loan_status']].copy()
            frame['scheduled_term_elapsed'] = matured.astype(float).where(data.term.notna())
            table = frame.groupby(['partition', 'term', 'loan_status'], dropna=False).scheduled_term_elapsed.agg(['size', 'mean']).reset_index()
            table['hypothetical_cutoff'] = scenario
            table = table.rename(columns={'size': 'n', 'mean': 'share_scheduled_term_elapsed'})
            scenarios.append(table)
        pd.concat(scenarios, ignore_index=True).to_csv(self.output / 'maturity_scenarios.csv', index=False)
        required = ['snapshot_date', 'observation_date', 'loan_status_date', 'outcome_observed_date']
        conclusion = dict(confirmed_common_observation_cutoff=None,
            missing_observation_metadata=[col for col in required if col not in schema.names],
            scheduled_maturity_is_not_actual_outcome_time=True,
            note='Scenarios are hypothetical, not inferred extraction dates; event maxima cannot establish all label observation dates.',
            historical_backtest_supported=False, target_and_cohorts_changed=False,
            audit_source='accepted_regression_fe development rows with non-null interest rate; known-outcome coverage verified')
        (self.output / 'followup_conclusion.json').write_text(json.dumps(conclusion, indent=2))
        return counts.reset_index(), eligibility.reset_index(), conclusion

    def inventory(self):
        path = self.root / 'data/processed/accepted_classification_fe.parquet'
        names = pq.read_schema(path).names
        existing = set(self.setup.predictors('classification'))
        generic_banned = set(leakage_columns(pd.DataFrame(columns=names)))
        administrative = {'id', 'member_id', 'url', 'desc', 'policy_code', 'pymnt_plan'}
        rows = []
        for name in names:
            if name in existing:
                decision, reason = 'existing_input', 'Prior classification allowlist; same origination availability assumptions'
            elif name in PRICING_COLUMNS:
                decision, reason = 'new_pricing_scenario', 'Project explicitly permits original assigned grade/subgrade for after-pricing classifier only'
            elif name in BUREAU_COLUMNS:
                decision, reason = 'conditional_bureau_scenario', 'Issuer underwriting includes tax liens, nonmedical collections, current delinquencies; assume these are original bureau aggregates, not later refreshes'
            elif name in generic_banned:
                decision, reason = 'excluded_post_origination', 'Outcome/payment/hardship/settlement information'
            elif name in {'target_class', 'loan_status'}:
                decision, reason = 'target_only', 'Never a predictor'
            elif name == 'issue_d':
                decision, reason = 'split_audit_only', 'Funding month is not a verified application date'
            elif name in administrative:
                decision, reason = 'excluded_administrative', 'Identifier/administrative/unstructured field not selected'
            else:
                decision, reason = 'deferred_not_verified', 'No new use without a documented definition and defensible original-snapshot availability; names alone do not establish leakage or eligibility'
            rows.append(dict(column=name, decision=decision, reason=reason,
                             source='Project.pdf; issuer prospectus' if name in PRICING_COLUMNS + BUREAU_COLUMNS else 'Existing project plan/schema',
                             evidence_url=ISSUER_SOURCE if name in PRICING_COLUMNS + BUREAU_COLUMNS else ''))
        inventory = pd.DataFrame(rows)
        inventory.to_csv(self.output / 'column_inventory.csv', index=False)
        # Census missingness by vintage/partition for explicitly reviewed inputs.
        review = sorted(existing | set(PRICING_COLUMNS) | set(BUREAU_COLUMNS) |
                        {'annual_inc_joint', 'dti_joint', 'sec_app_fico_range_low'})
        frame = pd.read_parquet(path, columns=['issue_d', 'target_class', 'application_type'] +
                               [c for c in review if c != 'application_type'],
                               filters=[('issue_d', '<', pd.Timestamp('2017-01-01'))])
        frame['partition'] = np.where(frame.issue_d < pd.Timestamp('2015-01-01'), 'train', 'validation')
        missing = []
        for (partition, year), group in frame.groupby(['partition', frame.issue_d.dt.year]):
            for col in review:
                missing.append(dict(partition=partition, year=year, column=col, n=len(group),
                                    missing_share=float(group[col].isna().mean())))
        pd.DataFrame(missing).to_csv(self.output / 'predictor_missingness_vintage.csv', index=False)
        joint_rows = []
        for (partition, app), group in frame.groupby(['partition', 'application_type'], dropna=False):
            for col in ['annual_inc_joint', 'dti_joint', 'sec_app_fico_range_low']:
                joint_rows.append(dict(partition=partition, application_type=app, field=col, n=len(group),
                                       missing_share=float(group[col].isna().mean())))
        pd.DataFrame(joint_rows).to_csv(self.output / 'joint_nonapplicability.csv', index=False)
        return inventory


class ReviewOperatingPoints:
    """Independent per-class review flags, not replacement multiclass targets.

    Precision floors are analytic examples, not approved lending policies.
    Require the floor AND at least min-alerts in EVERY chronological CV fold.
    If infeasible, persist an infeasible result rather than relax it using val.
    """
    def __init__(self, config=None):
        self.config = config or AuditConfig()

    @staticmethod
    def binary_metrics(y, probability, label, threshold):
        truth, alert = np.asarray(y) == label, np.asarray(probability) >= threshold
        n, hits = int(alert.sum()), int(np.sum(alert & truth))
        return dict(alerts=n, true_positives=hits, precision=hits / n if n else np.nan,
                    recall=hits / truth.sum() if truth.any() else np.nan, alert_share=n / len(truth))

    def select(self, oof, label):
        score = oof[f'probability_{label}'].to_numpy()
        quantiles = np.quantile(score, [0, .5, .75, .9, .95, .97, .98, .99, .995, .999, .9995, .9999, 1])
        thresholds = np.unique(np.r_[quantiles, [.0001, .0005, .001, .002, .005, .01, .02, .05, .1, .2, .3, .5, .7, .9, .99]])
        rows = []
        for threshold in thresholds:
            folds = [self.binary_metrics(g.target_class, g[f'probability_{label}'], label, threshold)
                     for _, g in oof.groupby('fold')]
            pooled = self.binary_metrics(oof.target_class, score, label, threshold)
            rows.append(dict(threshold=float(threshold), **pooled,
                             minimum_fold_precision=min(r['precision'] if r['alerts'] else -1 for r in folds),
                             minimum_fold_alerts=min(r['alerts'] for r in folds),
                             mean_fold_recall=np.mean([r['recall'] for r in folds])))
        candidates = pd.DataFrame(rows)
        chosen = []
        for floor in self.config.precision_floors:
            valid = candidates[(candidates.minimum_fold_precision >= floor) &
                               (candidates.minimum_fold_alerts >= self.config.minimum_alerts_per_fold)]
            if valid.empty:
                chosen.append(dict(label=label, precision_floor=floor, feasible=False, threshold=np.nan))
            else:
                best = valid.sort_values(['mean_fold_recall', 'precision', 'threshold'], ascending=[False, False, True]).iloc[0]
                chosen.append(dict(label=label, precision_floor=floor, feasible=True, **best.to_dict()))
        return pd.DataFrame(chosen), candidates

    def review_budget(self, y, probability, label, fraction):
        truth = np.asarray(y) == label
        n = max(1, int(np.ceil(len(truth) * fraction)))
        ties = np.random.default_rng(self.config.seed).random(len(truth))
        idx = np.lexsort((ties, -np.asarray(probability)))[:n]
        hits = int(truth[idx].sum())
        prevalence = float(truth.mean())
        return dict(label=label, fraction=fraction, alerts=n, true_positives=hits,
                    precision=hits / n, recall=hits / max(1, truth.sum()),
                    prevalence=prevalence, precision_lift=(hits / n) / prevalence if prevalence else np.nan)


class PredictorAuditExperiment:
    """Fixed models, 3 declared predictor scenarios, 5 month CV folds.

    Both models use balanced weights; no sampling or simultaneous robust-feature
    changes. New fields are selected from documented scenarios before fitting.
    Goal-specific model/scenario selection maximizes TRAINING-CV average precision,
    separately for At risk and Default; macro F1 is still reported, not optimized.
    """
    def __init__(self, root, output_dir, config=None):
        self.root, self.output = Path(root), Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config or AuditConfig()
        self.setup = ModellingDataSetup(root)
        self.metrics = ImbalanceMetrics()
        self.operating = ReviewOperatingPoints(self.config)

    def columns(self, variant):
        if variant not in self.config.variants:
            raise ValueError('Unknown predictor scenario')
        plan = self.setup.predictor_plan['classification']
        numeric, categorical = list(plan['numeric']), list(plan['categorical'])
        if variant in ('pricing_grade', 'pricing_bureau'):
            categorical += list(PRICING_COLUMNS)
        if variant == 'pricing_bureau':
            numeric += list(BUREAU_COLUMNS)
        columns = numeric + categorical
        banned = leakage_columns(pd.DataFrame(columns=columns))
        if banned or {'loan_status', 'target_class', 'issue_d', 'id'} & set(columns):
            raise ValueError('Ineligible model input')
        return numeric, categorical

    def load(self):
        train, val = self.setup.load_development('classification')
        extras = list(PRICING_COLUMNS + BUREAU_COLUMNS)
        frame = pd.read_parquet(self.root / 'data/processed/accepted_classification_fe.parquet',
            columns=['issue_d', 'target_class'] + extras, filters=[('issue_d', '<', pd.Timestamp('2017-01-01'))])
        frame = frame.sort_values('issue_d', kind='stable').reset_index(drop=True)
        for data, aux in [(train, frame[frame.issue_d < '2015-01-01']), (val, frame[frame.issue_d >= '2015-01-01'])]:
            aux = aux.reset_index(drop=True)
            np.testing.assert_array_equal(data.target_class, aux.target_class)
            np.testing.assert_array_equal(data.issue_d, aux.issue_d)
            for col in extras:
                data[col] = aux[col].to_numpy()
        return train, val

    @staticmethod
    def model(family):
        if family == 'logistic':
            return SupervisedModelFactory().build('classification', 'logistic', {'C': 1., 'class_weight': 'balanced'})
        if family == 'boosting':
            return EnsembleModelFactory().build('classification', 'hist_gradient_boosting',
                {'learning_rate': .1, 'max_leaf_nodes': 31, 'class_weight': 'balanced'})
        raise ValueError('Unknown family')

    def fit_score(self, family, X, y, Z, y_score):
        model = self.model(family)
        start = perf_counter()
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=self.config.threads):
            warnings.simplefilter('always')
            model.fit(X, y)
            prob = model.predict_proba(Z)
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            raise RuntimeError('Nonconverged fit; do not use it for selection')
        assert np.array_equal(model.classes_, [0, 1, 2])
        return model, prob, dict(family=family, n_train=len(y), n_evaluation=len(y_score),
            n_features=X.shape[1], fit_predict_seconds=perf_counter() - start,
            fit_warnings=' | '.join(str(w.message) for w in captured), **self.metrics.score(y_score, prob))

    def run(self):
        start = perf_counter()
        train, val = self.load()
        cv = list(MonthlyTimeSeriesCV(5, '2014-02').split(train, groups=train.issue_d))
        rows, threshold_rows, curve_rows, summaries = [], [], [], []
        oof_dir = self.output / 'oof'
        oof_dir.mkdir(exist_ok=True)
        for variant in self.config.variants:
            numeric, categorical = self.columns(variant)
            predictions = {family: [] for family in self.config.families}
            for fold, (a, b) in enumerate(cv, 1):
                pre = LendingClubPreprocessor(numeric, categorical, task='classification')
                X = pre.fit_transform(train.iloc[a])
                Z = pre.transform(train.iloc[b])
                for family in self.config.families:
                    _, prob, row = self.fit_score(family, X, train.iloc[a].target_class, Z, train.iloc[b].target_class)
                    row.update(variant=variant, fold=fold)
                    rows.append(row)
                    frame = train.iloc[b][['issue_d', 'target_class']].copy()
                    frame['training_row'], frame['fold'] = b, fold
                    for label in [0, 1, 2]:
                        frame[f'probability_{label}'] = prob[:, label]
                    predictions[family].append(frame)
                    print(f'{variant}/{family}: CV fold {fold}, risk AP={row["ap_class_1"]:.4f}', flush=True)
                pd.DataFrame(rows).to_csv(self.output / 'cv_scores.csv', index=False)
            for family in self.config.families:
                oof = pd.concat(predictions[family], ignore_index=True)
                oof.to_parquet(oof_dir / f'{family}_{variant}.parquet', index=False)
                for label in [1, 2]:
                    chosen, candidates = self.operating.select(oof, label)
                    chosen['family'], chosen['variant'] = family, variant
                    candidates['family'], candidates['variant'], candidates['label'] = family, variant, label
                    threshold_rows.append(chosen)
                    curve_rows.append(candidates)
        scores = pd.DataFrame(rows)
        summary = scores.groupby(['family', 'variant']).agg(
            risk_ap=('ap_class_1', 'mean'), risk_ap_fold_std=('ap_class_1', 'std'),
            default_ap=('ap_class_2', 'mean'), default_ap_fold_std=('ap_class_2', 'std'),
            macro_f1=('f1_macro', 'mean')).reset_index()
        summary.to_csv(self.output / 'cv_summary.csv', index=False)
        choices = {}
        for goal, column in [('At risk', 'risk_ap'), ('Default', 'default_ap')]:
            ranked = summary.assign(complexity=summary.variant.map({v:i for i,v in enumerate(self.config.variants)}))
            best = ranked.sort_values([column, 'complexity', 'family'], ascending=[False, True, True]).iloc[0]
            choices[goal] = dict(family=best.family, variant=best.variant, cv_average_precision=float(best[column]))
        thresholds = pd.concat(threshold_rows, ignore_index=True)
        thresholds.to_csv(self.output / 'cv_operating_points.csv', index=False)
        pd.concat(curve_rows, ignore_index=True).to_csv(self.output / 'cv_threshold_candidates.csv', index=False)
        (self.output / 'goal_selections.json').write_text(json.dumps(
            {'source': 'Training CV average precision, separately for each review goal', 'choices': choices}, indent=2))
        # All variants/goals/thresholds fixed before external validation scoring.
        validation_rows, operating_rows, budgets = [], [], []
        for variant in self.config.variants:
            numeric, categorical = self.columns(variant)
            pre = LendingClubPreprocessor(numeric, categorical, task='classification')
            X = pre.fit_transform(train)
            Z = pre.transform(val)
            for family in self.config.families:
                model, prob, row = self.fit_score(family, X, train.target_class, Z, val.target_class)
                row['variant'] = variant
                validation_rows.append(row)
                frame = val[['issue_d', 'target_class']].copy()
                frame['prediction'] = prob.argmax(axis=1)
                for label in [0, 1, 2]:
                    frame[f'probability_{label}'] = prob[:, label]
                frame.to_parquet(self.output / f'{family}_{variant}_validation_predictions.parquet', index=False)
                for _, point in thresholds[(thresholds.family == family) & (thresholds.variant == variant)].iterrows():
                    detail = dict(family=family, variant=variant, label=int(point.label),
                        precision_floor=point.precision_floor, cv_feasible=bool(point.feasible),
                        threshold=point.threshold)
                    if point.feasible:
                        detail.update(self.operating.binary_metrics(val.target_class, prob[:, int(point.label)],
                                                                    int(point.label), point.threshold))
                        detail['validation_floor_met'] = detail['precision'] >= point.precision_floor
                    operating_rows.append(detail)
                for label in [1, 2]:
                    for fraction in self.config.review_fractions:
                        budgets.append(dict(family=family, variant=variant,
                            **self.operating.review_budget(val.target_class, prob[:, label], label, fraction)))
                bundle = dict(model=model, preprocessing=pre, numeric=numeric, categorical=categorical,
                    scenario=variant, task='origination classification only', configuration=self.config.to_dict(),
                    availability='Original assigned grades and original baseline bureau snapshot assumed; conditional bureau scenario',
                    split=self.setup.split_config, stage='development audit; not final test-evaluated')
                artifact = self.output / f'{family}_{variant}_model.joblib'
                joblib.dump(bundle, artifact, compress=3)
                restored = joblib.load(artifact)
                with threadpool_limits(limits=self.config.threads):
                    check = restored['model'].predict_proba(restored['preprocessing'].transform(val.iloc[:20]))
                np.testing.assert_allclose(check, prob[:20], rtol=1e-10)
                print(f'{variant}/{family}: full validation and artifact checks complete', flush=True)
        pd.DataFrame(validation_rows).to_csv(self.output / 'validation_scores.csv', index=False)
        pd.DataFrame(operating_rows).to_csv(self.output / 'validation_operating_points.csv', index=False)
        pd.DataFrame(budgets).to_csv(self.output / 'validation_review_budgets.csv', index=False)
        (self.output / 'execution_summary.json').write_text(json.dumps({'seconds': perf_counter() - start}, indent=2))
        return scores, pd.DataFrame(validation_rows)
