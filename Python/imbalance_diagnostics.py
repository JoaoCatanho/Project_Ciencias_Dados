"""Controlled sampling/feature/decision experiments for Notebook 09b.

No synthetic data, no test reads, no balancing of evaluation rows. Transform
statistics use the complete current training fold BEFORE resampling so that
sampler comparisons change only the estimator's fitting examples/objective.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from .model_selection import MonthlyTimeSeriesCV
from .modelling_setup import ModellingDataSetup, BaselineRunner, ExperimentConfig
from .preprocessing import LendingClubPreprocessor


@dataclass
class DiagnosticConfig:
    seeds: tuple = (42, 43, 44)
    policies: tuple = ('none', 'weighted', 'under', 'over', 'hybrid')
    treatments: tuple = ('current', 'original', 'joint', 'joint_robust')
    good_to_default_cap: float = 2.0
    target_risk_share_nonrisk: float = 0.02
    maximum_risk_replication: int = 20
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    review_fraction: float = 0.01
    risk_thresholds: tuple = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    logistic_c: float = 1.0
    logistic_max_iter: int = 200
    logistic_tol: float = 0.001
    threads: int = 2
    boosting_iterations: int = 100
    boosting_leaf_nodes: int = 31
    boosting_learning_rate: float = 0.1

    def to_dict(self):
        return asdict(self)


class DiagnosticDataSetup(ModellingDataSetup):
    """Attach joint income as an explicitly declared auxiliary origination input.

    It is used only to recompute three affordability features, never selected as
    an independent predictor. Missing/non-applicable joint income stays missing.
    """
    def load_classification(self):
        train, val = self.load_development('classification')
        end = pd.Timestamp(self.split_config['val_end']) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        aux = pd.read_parquet(self.root / 'data/processed/accepted_classification_fe.parquet',
            columns=['issue_d', 'target_class', 'annual_inc_joint'], filters=[('issue_d', '<', end)])
        aux = aux.sort_values('issue_d', kind='stable').reset_index(drop=True)
        cutoff = pd.Timestamp(self.split_config['train_end']) + pd.offsets.MonthEnd(0)
        for frame, addition in [(train, aux.loc[aux.issue_d <= cutoff]), (val, aux.loc[aux.issue_d > cutoff])]:
            addition = addition.reset_index(drop=True)
            np.testing.assert_array_equal(frame.issue_d, addition.issue_d)
            np.testing.assert_array_equal(frame.target_class, addition.target_class)
            frame['annual_inc_joint'] = addition.annual_inc_joint.to_numpy()
        return train, val


class TrainingSampler:
    """Return source-row indices; duplicates denote random oversampling.

    Keep every Default and At risk original. Undersample only Good to at most
    twice the Default count. Over/hybrid raise At risk toward 2% of non-risk
    rows, capped at twenty times its original count. Never combine sampling
    policies with balanced class weights automatically.
    """
    def __init__(self, config=None):
        self.config = config or DiagnosticConfig()

    def indices(self, y, policy, seed):
        if policy not in self.config.policies:
            raise ValueError('Unknown sampling policy')
        y = np.asarray(y)
        if set(np.unique(y)) != {0, 1, 2}:
            raise ValueError('Expected all three classes in fitting data')
        rng = np.random.default_rng(seed)
        groups = {label: np.flatnonzero(y == label) for label in [0, 1, 2]}
        if policy in ('under', 'hybrid'):
            n = min(len(groups[0]), int(self.config.good_to_default_cap * len(groups[2])))
            groups[0] = rng.choice(groups[0], size=n, replace=False)
        if policy in ('over', 'hybrid'):
            original = groups[1]
            desired = int(np.ceil(self.config.target_risk_share_nonrisk * (len(groups[0]) + len(groups[2]))))
            target = max(len(original), min(desired, self.config.maximum_risk_replication * len(original)))
            extra = rng.choice(original, size=target - len(original), replace=True)
            groups[1] = np.concatenate([original, extra])
        result = np.concatenate(list(groups.values()))
        if policy in ('none', 'weighted'):
            return np.arange(len(y))
        return result[rng.permutation(len(result))]


class DiagnosticPreprocessor:
    """Feature ablations and train-fitted numeric treatment, with saved state.

    current: unchanged allowlist. original: remove engineered columns from that
    allowlist (not a reconstruction of every raw source column). joint: change
    loan_to_income, dti_x_loan_to_income and installment_burden to use joint
    income for Joint App, with missing/invalid joint income left NaN.
    joint_robust: joint policy + training 0.5–99.5% numeric clipping + log1p on
    declared nonnegative skewed variables. Individual primary income remains an
    input. Primary DTI remains unchanged; this is not a complete joint-risk model.
    """
    LOG_COLUMNS = ('annual_inc', 'loan_amnt', 'revol_bal', 'tot_cur_bal', 'tot_hi_cred_lim',
                   'bc_open_to_buy', 'loan_to_income', 'dti_x_loan_to_income', 'bc_pressure',
                   'installment_burden', 'loan_per_term', 'rate_burden')

    def __init__(self, numeric, categorical, engineered, treatment='current', config=None):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.engineered = list(engineered)
        self.treatment = treatment
        self.config = config or DiagnosticConfig()
        if treatment not in self.config.treatments:
            raise ValueError('Unknown feature treatment')
        if treatment == 'original':
            self.numeric = [c for c in self.numeric if c not in self.engineered]
            self.categorical = [c for c in self.categorical if c not in self.engineered]
        self.columns = self.numeric + self.categorical

    def deterministic(self, data):
        result = data[self.columns].copy()
        if self.treatment in ('joint', 'joint_robust'):
            if 'annual_inc_joint' not in data:
                raise ValueError('Joint treatment requires annual_inc_joint auxiliary input')
            joint = data.application_type.eq('Joint App')
            income = data.annual_inc.where(~joint, data.annual_inc_joint)
            income = income.where(income > 0)
            result['loan_to_income'] = data.loan_amnt / income
            result['dti_x_loan_to_income'] = data.dti * result.loan_to_income
            result['installment_burden'] = data.installment * 12 / income
        result[self.numeric] = result[self.numeric].replace([np.inf, -np.inf], np.nan)
        if self.treatment == 'joint_robust':
            for col in set(self.LOG_COLUMNS) & set(self.numeric):
                result[col] = result[col].where(result[col] >= 0)
        return result

    def fit(self, data):
        frame = self.deterministic(data)
        self.bounds_ = None
        if self.treatment == 'joint_robust':
            self.bounds_ = frame[self.numeric].quantile([self.config.lower_quantile, self.config.upper_quantile])
            frame = self._robust(frame)
        self.preprocessor_ = LendingClubPreprocessor(self.numeric, self.categorical, task='classification')
        self.preprocessor_.fit(frame)
        return self

    def _robust(self, frame):
        frame = frame.copy()
        frame[self.numeric] = frame[self.numeric].clip(lower=self.bounds_.iloc[0], upper=self.bounds_.iloc[1], axis=1)
        for col in set(self.LOG_COLUMNS) & set(self.numeric):
            frame[col] = np.log1p(frame[col])
        return frame

    def transform(self, data):
        frame = self.deterministic(data)
        if self.bounds_ is not None:
            frame = self._robust(frame)
        return self.preprocessor_.transform(frame)


class ImbalanceMetrics:
    """Per-class AP and fixed review-budget diagnostics alongside prior metrics."""
    def __init__(self, config=None):
        self.config = config or DiagnosticConfig()
        self.base = BaselineRunner()

    def score(self, y, probability, prediction=None):
        y = np.asarray(y)
        prediction = probability.argmax(axis=1) if prediction is None else np.asarray(prediction)
        scores = self.base.metrics('classification', y, prediction, probability)
        for label in [0, 1, 2]:
            truth = y == label
            scores[f'precision_class_{label}'] = float(np.sum(truth & (prediction == label)) / max(1, np.sum(prediction == label)))
            scores[f'recall_class_{label}'] = float(np.sum(truth & (prediction == label)) / max(1, truth.sum()))
            scores[f'ap_class_{label}'] = average_precision_score(truth, probability[:, label]) if truth.any() else np.nan
        budget = max(1, int(np.ceil(len(y) * self.config.review_fraction)))
        # Outcome-independent seeded tie breaking for equal probability scores.
        ties = np.random.default_rng(42).random(len(y))
        selected = np.lexsort((ties, -probability[:, 1]))[:budget]
        hits = int(np.sum(y[selected] == 1))
        scores.update(risk_prevalence=float(np.mean(y == 1)), review_n=budget,
                      risk_precision_at_budget=hits / budget, risk_recall_at_budget=hits / max(1, np.sum(y == 1)))
        return scores

    @staticmethod
    def threshold_prediction(probability, threshold):
        nonrisk = np.where(probability[:, 0] >= probability[:, 2], 0, 2)
        return np.where(probability[:, 1] >= threshold, 1, nonrisk)


class ImbalanceDiagnosticRunner:
    """Staged selection using training CV only, then full external validation.

    Sampling is varied at fixed current features and C=1. Then feature treatments
    are compared using the selected sampler. Boosting confirms that combined
    choice at fixed Notebook 09 parameters. This is a staged ablation, not a full
    interaction search or nested/unbiased CV estimate of the selected procedure.
    """
    def __init__(self, project_root, output_dir, config=None):
        self.root = Path(project_root)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'cv_predictions').mkdir(exist_ok=True)
        self.config = config or DiagnosticConfig()
        self.setup = DiagnosticDataSetup(self.root)
        self.protocol = ExperimentConfig()
        self.sampler = TrainingSampler(self.config)
        self.metrics = ImbalanceMetrics(self.config)
        self.plan = self.setup.predictor_plan['classification']
        spec = pd.read_csv(self.root / 'outputs/tables/feature_engineering_spec_classification.csv')
        self.engineered = spec['name'].tolist()

    def seeds(self, policy):
        return self.config.seeds if policy in ('under', 'over', 'hybrid') else (self.config.seeds[0],)

    def preprocessor(self, treatment):
        return DiagnosticPreprocessor(self.plan['numeric'], self.plan['categorical'], self.engineered,
                                      treatment, self.config)

    def model(self, family, policy, seed):
        weight = 'balanced' if policy == 'weighted' else None
        if family == 'logistic':
            return LogisticRegression(C=self.config.logistic_c, solver='newton-cg',
                max_iter=self.config.logistic_max_iter, tol=self.config.logistic_tol,
                class_weight=weight, random_state=seed)
        if family == 'boosting':
            return HistGradientBoostingClassifier(max_iter=self.config.boosting_iterations,
                learning_rate=self.config.boosting_learning_rate, max_leaf_nodes=self.config.boosting_leaf_nodes,
                min_samples_leaf=100, l2_regularization=1., early_stopping=False,
                class_weight=weight, random_state=seed)
        raise ValueError('Unknown family')

    def _fit(self, family, policy, seed, X, y, Z, y_score):
        indices = self.sampler.indices(y, policy, seed)
        fitting_y = np.asarray(y)[indices]
        model = self.model(family, policy, seed)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=self.config.threads):
            warnings.simplefilter('always')
            model.fit(X[indices], fitting_y, sample_weight=np.full(len(indices), len(y) / len(indices)))
            prob = np.zeros((len(Z), 3))
            prob[:, model.classes_.astype(int)] = model.predict_proba(Z)
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            raise RuntimeError(f'Nonconverged {family}/{policy}/{seed}; inspect optimizer budget.')
        if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
            raise ValueError('Invalid class probabilities')
        row = dict(family=family, policy=policy, seed=seed, n_train_original=len(y),
                   n_train_fitted=len(indices), n_distinct_fitted=len(np.unique(indices)),
                   uniform_weight_per_fitted_row=len(y) / len(indices),
                   n_evaluation=len(y_score), n_features=X.shape[1], fit_predict_seconds=perf_counter() - started,
                   fit_warnings=' | '.join(str(w.message) for w in captured),
                   **self.metrics.score(y_score, prob))
        for label in [0, 1, 2]:
            row[f'original_class_{label}'] = int(np.sum(np.asarray(y) == label))
            row[f'fitted_class_{label}'] = int(np.sum(fitting_y == label))
        return row, model, prob

    def _cv_path(self, family, treatment, policy, seed, fold):
        return self.output / 'cv_predictions' / f'{family}_{treatment}_{policy}_{seed}_{fold}.parquet'

    def _save_cv_prediction(self, prob, y, positions, family, treatment, policy, seed, fold):
        pd.DataFrame({'training_row': positions, 'target_class': np.asarray(y, dtype=np.int8),
                      'p_risk': prob[:, 1], 'default_prediction': prob.argmax(axis=1).astype(np.int8),
                      'nonrisk_prediction': np.where(prob[:, 0] >= prob[:, 2], 0, 2).astype(np.int8)}).to_parquet(
            self._cv_path(family, treatment, policy, seed, fold), index=False)

    def _cv(self, train, treatment, policies, family='logistic'):
        rows = []
        cv = MonthlyTimeSeriesCV(self.protocol.n_splits, self.protocol.classification_cv_min_train_end)
        for fold, (a, b) in enumerate(cv.split(train, groups=train.issue_d), 1):
            pre = self.preprocessor(treatment).fit(train.iloc[a])
            X, Z = pre.transform(train.iloc[a]), pre.transform(train.iloc[b])
            for policy in policies:
                trial_seeds = (self.config.seeds[0],) if family == 'boosting' else self.seeds(policy)
                for seed in trial_seeds:
                    row, _, prob = self._fit(family, policy, seed, X, train.iloc[a].target_class,
                                             Z, train.iloc[b].target_class)
                    row.update(treatment=treatment, fold=fold)
                    rows.append(row)
                    self._save_cv_prediction(prob, train.iloc[b].target_class, b, family, treatment, policy, seed, fold)
                    print(f'{family}/{treatment}: fold {fold}, {policy}, seed {seed}: macro F1={row["f1_macro"]:.3f}', flush=True)
            pd.DataFrame(rows).to_csv(self.output / f'{family}_{treatment}_cv_checkpoint.csv', index=False)
        return pd.DataFrame(rows)

    @staticmethod
    def summary(scores, keys):
        rows = []
        for key, group in scores.groupby(keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            fold_means = group.groupby('fold').f1_macro.mean()
            seed_means = group.groupby('seed').f1_macro.mean()
            rows.append(dict(zip(keys, key), macro_f1_mean=group.f1_macro.mean(),
                             macro_f1_fold_std=fold_means.std(ddof=1),
                             macro_f1_seed_std=seed_means.std(ddof=1), n_folds=group.fold.nunique(),
                             n_seeds=group.seed.nunique(), risk_ap_mean=group.ap_class_1.mean(),
                             risk_recall_mean=group.recall_class_1.mean(), risk_precision_mean=group.precision_class_1.mean()))
        return pd.DataFrame(rows)

    def _threshold_selection(self, treatment, policy):
        rows = []
        for fold in range(1, self.protocol.n_splits + 1):
            for seed in self.seeds(policy):
                data = pd.read_parquet(self._cv_path('logistic', treatment, policy, seed, fold))
                for threshold in (None,) + self.config.risk_thresholds:
                    prediction = data.default_prediction if threshold is None else np.where(
                        data.p_risk >= threshold, 1, data.nonrisk_prediction)
                    rows.append(dict(fold=fold, seed=seed, decision='argmax' if threshold is None else str(threshold),
                        f1_macro=f1_score(data.target_class, prediction, average='macro', labels=[0, 1, 2], zero_division=0)))
        scores = pd.DataFrame(rows)
        summary = scores.groupby('decision').f1_macro.agg(['mean', 'std']).reset_index()
        summary['prefer_default'] = summary.decision.eq('argmax')
        summary = summary.sort_values(['mean', 'prefer_default', 'decision'], ascending=[False, False, True])
        summary.to_csv(self.output / 'decision_cv_summary.csv', index=False)
        scores.to_csv(self.output / 'decision_cv_scores.csv', index=False)
        best = summary.iloc[0].decision
        return None if best == 'argmax' else float(best)

    def run(self):
        started = perf_counter()
        train, validation = self.setup.load_classification()
        # Ensure policy comparisons have exactly the existing Part 1 cohorts.
        counts = json.loads((self.root / 'outputs/split_ids.json').read_text())['classification']
        assert len(train) == counts['n_train'] and len(validation) == counts['n_val']
        sampling_cv = self._cv(train, 'current', self.config.policies)
        sampling_cv.to_csv(self.output / 'sampling_cv_scores.csv', index=False)
        sampling_summary = self.summary(sampling_cv, ['policy'])
        sampling_summary['complexity'] = sampling_summary.policy.map({name: i for i, name in enumerate(self.config.policies)})
        sampling_summary = sampling_summary.sort_values(['macro_f1_mean', 'complexity'], ascending=[False, True])
        sampling_summary.to_csv(self.output / 'sampling_cv_summary.csv', index=False)
        policy = sampling_summary.iloc[0].policy
        print(f'Sampling choice from training CV: {policy}', flush=True)
        feature_rows = [sampling_cv[sampling_cv.policy == policy]]
        for treatment in self.config.treatments:
            if treatment != 'current':
                feature_rows.append(self._cv(train, treatment, [policy]))
        feature_cv = pd.concat(feature_rows, ignore_index=True)
        feature_cv.to_csv(self.output / 'feature_cv_scores.csv', index=False)
        feature_summary = self.summary(feature_cv, ['treatment'])
        feature_summary['complexity'] = feature_summary.treatment.map({name: i for i, name in enumerate(self.config.treatments)})
        # Round numerical dust only: no joint applications in training/CV means
        # joint and current can be mathematically identical; prefer simpler ties.
        feature_summary['selection_score'] = feature_summary.macro_f1_mean.round(10)
        feature_summary = feature_summary.sort_values(['selection_score', 'complexity'], ascending=[False, True])
        feature_summary.to_csv(self.output / 'feature_cv_summary.csv', index=False)
        treatment = feature_summary.iloc[0].treatment
        threshold = self._threshold_selection(treatment, policy)
        selection = dict(policy=policy, treatment=treatment, risk_threshold=threshold,
                         source='Training CV only; staged selection, no validation scoring yet',
                         criterion='Mean macro F1; simpler policies resolve ties')
        (self.output / 'selection.json').write_text(json.dumps(selection, indent=2))
        print(f'Feature/decision choice: {selection}', flush=True)
        # Confirm the fixed combined treatment with boosting; Notebook 09's
        # matching balanced/current CV results provide a paired reference.
        if policy == 'weighted' and treatment == 'current':
            old = pd.read_csv(self.root / 'outputs/ensembles/classification_cv_scores.csv')
            parameters = json.dumps({'class_weight': 'balanced', 'learning_rate': 0.1, 'max_leaf_nodes': 31}, sort_keys=True)
            boosting_cv = old[(old.model == 'hist_gradient_boosting') & (old.parameters == parameters)].copy()
            boosting_cv['reference_reused'] = True
        else:
            boosting_cv = self._cv(train, treatment, [policy], family='boosting')
            boosting_cv['reference_reused'] = False
        boosting_cv.to_csv(self.output / 'boosting_confirmation_cv.csv', index=False)
        # Freeze every choice before examining this notebook's validation scores.
        rows = []
        for current_treatment in self.config.treatments:
            policies = self.config.policies if current_treatment == 'current' else (policy,)
            pre = self.preprocessor(current_treatment).fit(train)
            X, Z = pre.transform(train), pre.transform(validation)
            for candidate_policy in policies:
                for seed in self.seeds(candidate_policy):
                    row, model, probability = self._fit('logistic', candidate_policy, seed, X,
                        train.target_class, Z, validation.target_class)
                    row.update(treatment=current_treatment, decision='argmax')
                    rows.append(row)
                    label = f'logistic_{current_treatment}_{candidate_policy}_{seed}'
                    matrix = confusion_matrix(validation.target_class, probability.argmax(axis=1), labels=[0, 1, 2])
                    pd.DataFrame(matrix, index=['Good', 'At risk', 'Default'], columns=['Good', 'At risk', 'Default']).to_csv(self.output / f'{label}_confusion.csv')
                    if candidate_policy == policy and current_treatment == treatment:
                        self._save_bundle(model, pre, probability, validation, label, selection)
                        if threshold is not None:
                            adjusted = self.metrics.threshold_prediction(probability, threshold)
                            adjusted_row = row.copy()
                            adjusted_row.update(decision=f'risk_threshold_{threshold}', **self.metrics.score(validation.target_class, probability, adjusted))
                            rows.append(adjusted_row)
                            pd.DataFrame(confusion_matrix(validation.target_class, adjusted, labels=[0, 1, 2]),
                                index=['Good', 'At risk', 'Default'], columns=['Good', 'At risk', 'Default']).to_csv(self.output / f'{label}_threshold_confusion.csv')
                    print(f'Validation: {label}, macro F1={row["f1_macro"]:.3f}', flush=True)
            pd.DataFrame(rows).to_csv(self.output / 'validation_scores.csv', index=False)
        if policy == 'weighted' and treatment == 'current':
            # Exact same estimator/configuration/data as Notebook 09; do not
            # manufacture an independent confirmation by repeating the fit.
            probability_frame = pd.read_parquet(self.root / 'outputs/ensembles/classification_hist_gradient_boosting_validation_predictions.parquet')
            np.testing.assert_array_equal(probability_frame.target_class, validation.target_class)
            probability = probability_frame[[f'probability_{c}' for c in [0, 1, 2]]].to_numpy()
            row = dict(family='boosting', policy=policy, seed=42, treatment=treatment,
                       decision='argmax', reference_reused=True, n_evaluation=len(validation),
                       **self.metrics.score(validation.target_class, probability))
        else:
            pre = self.preprocessor(treatment).fit(train)
            X, Z = pre.transform(train), pre.transform(validation)
            # Primary seed confirmation; stochastic sampling stability is assessed
            # separately in logistic CV and external development comparisons.
            row, model, probability = self._fit('boosting', policy, self.config.seeds[0], X,
                                                 train.target_class, Z, validation.target_class)
            row.update(treatment=treatment, decision='argmax', reference_reused=False)
            self._save_bundle(model, pre, probability, validation, f'boosting_{treatment}_{policy}_42', selection)
        pd.DataFrame([row]).to_csv(self.output / 'boosting_confirmation_validation.csv', index=False)
        (self.output / 'execution_summary.json').write_text(json.dumps({'seconds': perf_counter() - started}, indent=2))
        return sampling_cv, feature_cv, pd.DataFrame(rows)

    def _save_bundle(self, model, pre, probability, validation, label, selection):
        frame = validation[['issue_d', 'target_class']].copy()
        frame['prediction'] = probability.argmax(axis=1)
        for c in [0, 1, 2]:
            frame[f'probability_{c}'] = probability[:, c]
        frame.to_parquet(self.output / f'{label}_validation_predictions.parquet', index=False)
        bundle = dict(model=model, preprocessing=pre, selection=selection, configuration=self.config.to_dict(),
                      task='classification', split=self.setup.split_config,
                      stage='validation-informed diagnostic development; not final test model')
        joblib.dump(bundle, self.output / f'{label}_model.joblib', compress=3)
        restored = joblib.load(self.output / f'{label}_model.joblib')
        with threadpool_limits(limits=self.config.threads):
            reloaded = restored['model'].predict_proba(restored['preprocessing'].transform(validation.iloc[:20]))
        np.testing.assert_allclose(reloaded, probability[:20], rtol=1e-10)
