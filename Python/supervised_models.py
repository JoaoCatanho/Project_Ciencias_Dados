"""Non-ensemble models and fold-aware experiments for Notebook 08.

Every learned transform is fitted on the current training partition. Candidates
share only that partition's fitted transformation, equivalent to fitting identical
preprocessing in each candidate pipeline. All training and evaluation rows are
retained. No held-out test data are loaded by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .model_selection import MonthlyTimeSeriesCV
from .modelling_setup import BaselineRunner, ExperimentConfig
from .preprocessing import LendingClubPreprocessor


@dataclass
class SupervisedExperimentConfig:
    seed: int = 42
    blas_threads: int = 2
    ridge_alphas: tuple = (0.1, 10.0, 1000.0)
    logistic_cs: tuple = (0.01, 0.1, 1.0)
    tree_depths: tuple = (5, 10, 15)
    regression_leaf_sizes: tuple = (100, 500)
    classification_leaf_size: int = 500
    class_weights: tuple = (None, 'balanced')
    logistic_max_iter: int = 200
    logistic_tol: float = 0.001

    def to_dict(self):
        return asdict(self)


class SupervisedModelFactory:
    """Construct only Ridge, logistic regression, and individual decision trees."""

    def __init__(self, config=None):
        self.config = config or SupervisedExperimentConfig()

    def candidates(self, task):
        c = self.config
        if task == 'regression':
            return ([('ridge', {'alpha': a}) for a in c.ridge_alphas]
                    + [('decision_tree', {'max_depth': d, 'min_samples_leaf': leaf})
                       for d in c.tree_depths for leaf in c.regression_leaf_sizes])
        if task == 'classification':
            return ([('logistic', {'C': value, 'class_weight': weight})
                     for value in c.logistic_cs for weight in c.class_weights]
                    + [('decision_tree', {'max_depth': d,
                                          'min_samples_leaf': c.classification_leaf_size,
                                          'class_weight': weight})
                       for d in c.tree_depths for weight in c.class_weights])
        raise ValueError('task must be regression or classification')

    def build(self, task, family, parameters):
        c = self.config
        if task == 'regression' and family == 'ridge':
            return Ridge(solver='cholesky', **parameters)
        if task == 'classification' and family == 'logistic':
            return LogisticRegression(solver='newton-cg', max_iter=c.logistic_max_iter,
                                      tol=c.logistic_tol, random_state=c.seed, **parameters)
        if family == 'decision_tree' and task in ('regression', 'classification'):
            model = DecisionTreeRegressor if task == 'regression' else DecisionTreeClassifier
            return model(random_state=c.seed, **parameters)
        raise ValueError('Unsupported non-ensemble task/model combination')


class SupervisedExperimentRunner:
    """Select each family's configuration by CV, then score external validation.

    Convergence warnings are captured and persisted. A parameter candidate is
    eligible for selection only when all required folds finish without a
    ConvergenceWarning. Numeric/optimization warnings remain visible in tables.
    Class weights are calculated by sklearn from the current fitting labels only.
    """

    def __init__(self, setup, output_dir, config=None, protocol=None):
        self.setup = setup
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config or SupervisedExperimentConfig()
        self.protocol = protocol or ExperimentConfig()
        self.factory = SupervisedModelFactory(self.config)
        self.evaluator = BaselineRunner(self.protocol)

    def prepare(self, task, training, evaluation):
        columns = self.setup.predictors(task)
        plan = self.setup.predictor_plan[task]
        pre = LendingClubPreprocessor(plan['numeric'], plan['categorical'], task=task)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            X = pre.fit_transform(training[columns])
            Z = pre.transform(evaluation[columns])
        if not np.isfinite(X).all() or not np.isfinite(Z).all():
            raise ValueError('Preprocessed predictors must be finite.')
        notes = ' | '.join(dict.fromkeys(str(w.message) for w in captured))
        return pre, X, Z, notes

    def fit_and_score(self, task, family, parameters, X, y, Z, y_evaluation):
        model = self.factory.build(task, family, parameters)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with threadpool_limits(limits=self.config.blas_threads, user_api='blas'):
                model.fit(X, y)
        fit_seconds = perf_counter() - started
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
        started = perf_counter()
        with threadpool_limits(limits=self.config.blas_threads, user_api='blas'):
            train_prediction = model.predict(X)
            prediction = model.predict(Z)
            probabilities = None
            if task == 'classification':
                probabilities = np.zeros((len(Z), 3))
                probabilities[:, model.classes_.astype(int)] = model.predict_proba(Z)
        predict_seconds = perf_counter() - started
        train_primary = (float(np.sqrt(np.mean((np.asarray(y) - train_prediction) ** 2)))
                         if task == 'regression' else
                         f1_score(y, train_prediction, average='macro', labels=[0, 1, 2], zero_division=0))
        record = dict(model=family, parameters=json.dumps(parameters, sort_keys=True),
                      n_train=len(X), n_evaluation=len(Z), n_features=X.shape[1],
                      fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                      train_primary=train_primary, converged=converged,
                      fit_warnings=' | '.join(dict.fromkeys(str(w.message) for w in captured)),
                      n_iterations=int(np.max(model.n_iter_)) if hasattr(model, 'n_iter_') and model.n_iter_ is not None else 0,
                      **self.evaluator.metrics(task, y_evaluation, prediction, probabilities))
        if task == 'classification':
            for label in [0, 1, 2]:
                p = np.asarray(y_evaluation) == label
                record[f'recall_class_{label}'] = float(np.mean(prediction[p] == label)) if p.any() else np.nan
                record[f'n_train_class_{label}'] = int(np.sum(np.asarray(y) == label))
        return record, model, prediction, probabilities

    def select(self, task, scores):
        metric = 'rmse' if task == 'regression' else 'f1_macro'
        summary = scores.groupby(['model', 'parameters']).agg(
            cv_mean=(metric, 'mean'), cv_std=(metric, 'std'), n_folds=(metric, 'count'),
            converged=('converged', 'all'), mean_fit_seconds=('fit_seconds', 'mean'),
        ).reset_index()
        summary['eligible'] = summary.converged & summary.n_folds.eq(self.protocol.n_splits)
        selections = {}
        for family, group in summary.groupby('model'):
            valid = group[group.eligible].sort_values(['cv_mean', 'parameters'], ascending=[task == 'regression', True])
            if valid.empty:
                raise RuntimeError(f'No fully converged CV configuration for {family}; inspect diagnostics.')
            selections[family] = json.loads(valid.iloc[0].parameters)
        return summary, selections

    def _save_interpretation(self, task, family, model, pre):
        names = pre._pipeline.named_steps['pre'].get_feature_names_out()
        if hasattr(model, 'feature_importances_'):
            table = pd.DataFrame({'feature': names, 'importance': model.feature_importances_})
            table = table.sort_values('importance', ascending=False)
        elif task == 'regression':
            table = pd.DataFrame({'feature': names, 'coefficient': model.coef_})
            table['absolute_coefficient'] = table.coefficient.abs()
            table = table.sort_values('absolute_coefficient', ascending=False)
        else:
            table = pd.concat([pd.DataFrame({'class': label, 'feature': names, 'coefficient': coef})
                               for label, coef in zip(model.classes_, model.coef_)], ignore_index=True)
        table.to_csv(self.output / f'{task}_{family}_feature_effects.csv', index=False)

    def run(self, task):
        train, validation = self.setup.load_development(task)
        target = 'int_rate' if task == 'regression' else 'target_class'
        cv = MonthlyTimeSeriesCV(self.protocol.n_splits,
            min_train_end=self.protocol.classification_cv_min_train_end if task == 'classification' else None)
        records = []
        candidates = self.factory.candidates(task)
        for fold, (a, b) in enumerate(cv.split(train, groups=train.issue_d), 1):
            started = perf_counter()
            pre, X, Z, notes = self.prepare(task, train.iloc[a], train.iloc[b])
            preprocessing_seconds = perf_counter() - started
            for family, parameters in candidates:
                row, model, prediction, probabilities = self.fit_and_score(
                    task, family, parameters, X, train.iloc[a][target], Z, train.iloc[b][target])
                row.update(task=task, partition=f'cv_{fold}', preprocessing_seconds=preprocessing_seconds,
                           preprocessing_notes=notes)
                records.append(row)
                print(f'{task} fold {fold}: {family} {parameters}; converged={row["converged"]}', flush=True)
                del model, prediction, probabilities
            pd.DataFrame(records).to_csv(self.output / f'{task}_cv_scores.csv', index=False)
            del pre, X, Z
        cv_scores = pd.DataFrame(records)
        tuning, selections = self.select(task, cv_scores)
        tuning.to_csv(self.output / f'{task}_tuning.csv', index=False)
        (self.output / f'{task}_selection.json').write_text(json.dumps(
            {'metric': 'rmse' if task == 'regression' else 'f1_macro',
             'source': 'training CV only', 'models': selections}, indent=2))
        # No external validation scoring occurs before selections are saved.
        started = perf_counter()
        pre, X, Z, notes = self.prepare(task, train, validation)
        preprocessing_seconds = perf_counter() - started
        validation_records = []
        for family, parameters in selections.items():
            row, model, prediction, probabilities = self.fit_and_score(
                task, family, parameters, X, train[target], Z, validation[target])
            row.update(task=task, partition='validation', preprocessing_seconds=preprocessing_seconds,
                       preprocessing_notes=notes)
            if not row['converged']:
                raise RuntimeError(f'{task} {family}: final training fit did not converge.')
            frame = validation[['issue_d', target]].copy()
            frame['partition_row'] = np.arange(len(validation))
            frame['prediction'] = prediction
            if probabilities is not None:
                for label in [0, 1, 2]:
                    frame[f'probability_{label}'] = probabilities[:, label]
            frame.to_parquet(self.output / f'{task}_{family}_validation_predictions.parquet', index=False)
            self._save_interpretation(task, family, model, pre)
            bundle = dict(model=model, preprocessing=pre, predictors=self.setup.predictors(task),
                          task=task, target=target, parameters=parameters,
                          split=self.setup.split_config, experiment=self.config.to_dict(),
                          stage='development: full training cohort only; not final deployment model')
            path = self.output / f'{task}_{family}_development_model.joblib'
            joblib.dump(bundle, path, compress=3)
            restored = joblib.load(path)
            check = restored['preprocessing'].transform(validation.iloc[:20][restored['predictors']])
            np.testing.assert_allclose(restored['model'].predict(check), prediction[:20], rtol=1e-10)
            if probabilities is not None:
                np.testing.assert_allclose(restored['model'].predict_proba(check), probabilities[:20], rtol=1e-10)
            validation_records.append(row)
            print(f'{task} {family}: full validation and saved-model checks complete', flush=True)
        validation_scores = pd.DataFrame(validation_records)
        validation_scores.to_csv(self.output / f'{task}_validation_scores.csv', index=False)
        return cv_scores, tuning, validation_scores
