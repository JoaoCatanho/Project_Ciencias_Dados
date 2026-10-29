"""Fold-aware experiments for the NumPy kNN implementation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from threadpoolctl import threadpool_limits

from .knn import KNNClassifierScratch, KNNRegressorScratch
from .model_selection import MonthlyTimeSeriesCV
from .modelling_setup import BaselineRunner, ExperimentConfig
from .preprocessing import LendingClubPreprocessor


@dataclass
class KNNExperimentConfig:
    seed: int = 42
    training_cap: int = 10000
    neighbors: tuple = (5, 15, 31)
    weights: tuple = ('uniform', 'distance')
    batch_size: int = 256
    blas_threads: int = 2

    def to_dict(self):
        return asdict(self)


class KNNExperimentRunner:
    """Tune only on training CV, then evaluate the selected model on validation.

    Imputation, scaling and encoding are fitted afresh on the sampled training
    rows in each fold. Search candidates share the same maximum-k neighbour
    computation, an exact reuse because smaller k uses the sorted prefix.
    Validation rows are never sampled. Classification sampling is proportional,
    with at least one training row per available class; it is not class balancing.
    """

    def __init__(self, setup, output_dir, config=None, protocol=None):
        self.setup = setup
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config or KNNExperimentConfig()
        self.protocol = protocol or ExperimentConfig()
        self.evaluator = BaselineRunner(self.protocol)
        self.candidates = [(k, w) for k in self.config.neighbors for w in self.config.weights]

    def sample_indices(self, data, task, seed):
        rng = np.random.default_rng(seed)
        size = min(self.config.training_cap, len(data))
        if task == 'regression' or size == len(data):
            return np.sort(rng.choice(len(data), size=size, replace=False))
        labels, counts = np.unique(data.target_class.to_numpy(), return_counts=True)
        if size < len(labels):
            raise ValueError('Training cap must accommodate every class.')
        # Allocate one per class, then distribute remaining slots proportionally.
        allocation = np.ones(len(labels), dtype=int)
        exact = (size - len(labels)) * (counts - 1) / (len(data) - len(labels))
        allocation += np.floor(exact).astype(int)
        for i in np.argsort(-(exact - np.floor(exact)), kind='stable')[:size - allocation.sum()]:
            allocation[i] += 1
        chosen = [rng.choice(np.flatnonzero(data.target_class.to_numpy() == label),
                             size=int(n), replace=False) for label, n in zip(labels, allocation)]
        return np.sort(np.concatenate(chosen))

    def _predict_candidates(self, model, matrix, candidates):
        outputs = {}
        for k, weight in candidates:
            outputs[(k, weight)] = np.empty((len(matrix), len(model.classes_))) if hasattr(model, 'classes_') else np.empty(len(matrix))
        with threadpool_limits(limits=self.config.blas_threads, user_api='blas'):
            for query_slice, distances, indices in model.iter_neighbors(matrix):
                for k, weight in candidates:
                    if hasattr(model, 'classes_'):
                        prediction = model.predict_proba_from_neighbors(distances, indices, k, weight)
                    else:
                        prediction = model.predict_from_neighbors(distances, indices, k, weight)
                    outputs[(k, weight)][query_slice] = prediction
        return outputs

    def _fit_score(self, task, training, evaluation, partition, candidates):
        started = perf_counter()
        target = 'int_rate' if task == 'regression' else 'target_class'
        fold_seed = self.config.seed + (int(partition.split('_')[-1]) if partition.startswith('cv_') else 100)
        chosen = self.sample_indices(training, task, fold_seed)
        sample = training.iloc[chosen]
        pd.DataFrame({'development_row': sample.index}).to_csv(
            self.output / f'{task}_{partition}_training_rows.csv', index=False)
        predictors = self.setup.predictors(task)
        plan = self.setup.predictor_plan[task]
        pre = LendingClubPreprocessor(plan['numeric'], plan['categorical'], task=task)
        X = pre.fit_transform(sample[predictors])
        Z = pre.transform(evaluation[predictors])
        if not np.isfinite(X).all() or not np.isfinite(Z).all():
            raise ValueError('Preprocessed matrices must be finite.')
        model_type = KNNRegressorScratch if task == 'regression' else KNNClassifierScratch
        model = model_type(n_neighbors=max(k for k, _ in candidates), batch_size=self.config.batch_size)
        model.fit(X, sample[target].to_numpy())
        fit_seconds = perf_counter() - started
        predicted = self._predict_candidates(model, Z, candidates)
        search_seconds = perf_counter() - started - fit_seconds
        records = []
        for (k, weight), values in predicted.items():
            prob = values if task == 'classification' else None
            pred = model.classes_[values.argmax(axis=1)] if prob is not None else values
            row = dict(task=task, partition=partition, model='knn_scratch', k=k, weights=weight,
                       n_train_available=len(training), n_train=len(sample), n_evaluation=len(evaluation),
                       n_features=X.shape[1], fit_transform_seconds=fit_seconds,
                       shared_neighbor_search_seconds=search_seconds,
                       **self.evaluator.metrics(task, evaluation[target], pred, prob))
            if task == 'classification':
                for label in [0, 1, 2]:
                    row[f'n_train_class_{label}'] = int((sample[target] == label).sum())
            records.append(row)
        # Matched training-sample dummy references isolate the sample-budget effect.
        strategies = ['mean', 'median'] if task == 'regression' else ['prior']
        for strategy in strategies:
            dummy = DummyRegressor(strategy=strategy) if task == 'regression' else DummyClassifier(strategy=strategy)
            dummy.fit(np.zeros((len(sample), 1)), sample[target])
            placeholder = np.zeros((len(evaluation), 1))
            prob = dummy.predict_proba(placeholder) if task == 'classification' else None
            records.append(dict(task=task, partition=partition, model=f'dummy_sample_{strategy}',
                                n_train=len(sample), n_evaluation=len(evaluation),
                                **self.evaluator.metrics(task, evaluation[target], dummy.predict(placeholder), prob)))
        return records, model, pre, predicted

    def run(self, task):
        train, validation = self.setup.load_development(task)
        cv = MonthlyTimeSeriesCV(self.protocol.n_splits,
            min_train_end=self.protocol.classification_cv_min_train_end if task == 'classification' else None)
        rows = []
        for fold, (a, b) in enumerate(cv.split(train, groups=train.issue_d), 1):
            records, _, _, _ = self._fit_score(task, train.iloc[a], train.iloc[b], f'cv_{fold}', self.candidates)
            rows.extend(records)
            print(f'{task}: CV fold {fold}/{self.protocol.n_splits} complete', flush=True)
        cv_scores = pd.DataFrame(rows)
        cv_scores.to_csv(self.output / f'{task}_cv_scores.csv', index=False)
        metric = 'rmse' if task == 'regression' else 'f1_macro'
        tuning = (cv_scores[cv_scores.model == 'knn_scratch'].groupby(['k', 'weights'])[metric]
                  .agg(['mean', 'std', 'count']).reset_index())
        tuning = tuning.sort_values(['mean', 'k', 'weights'], ascending=[task == 'regression', True, True])
        tuning.to_csv(self.output / f'{task}_tuning.csv', index=False)
        best = (int(tuning.iloc[0].k), str(tuning.iloc[0].weights))
        # Selection is persisted before the external validation outcomes are scored.
        (self.output / f'{task}_selection.json').write_text(json.dumps(
            dict(k=best[0], weights=best[1], selection_metric=metric, source='training CV only'), indent=2))
        records, model, pre, predicted = self._fit_score(task, train, validation, 'validation', [best])
        model.n_neighbors, model.weights = best
        target = 'int_rate' if task == 'regression' else 'target_class'
        values = predicted[best]
        frame = validation[['issue_d', target]].copy()
        frame['development_row'] = validation.index
        frame['prediction'] = model.classes_[values.argmax(axis=1)] if task == 'classification' else values
        if task == 'classification':
            for i, label in enumerate(model.classes_):
                frame[f'probability_{label}'] = values[:, i]
        frame.to_parquet(self.output / f'{task}_validation_predictions.parquet', index=False)
        bundle = dict(model=model, preprocessing=pre, predictors=self.setup.predictors(task),
                      task=task, target=target, experiment=self.config.to_dict(),
                      split=self.setup.split_config, stage='development: train only; not final deployment model')
        joblib.dump(bundle, self.output / f'{task}_development_model.joblib', compress=3)
        validation_scores = pd.DataFrame(records)
        validation_scores.to_csv(self.output / f'{task}_validation_scores.csv', index=False)
        # Check that a saved bundle reproduces the exported predictions.
        restored = joblib.load(self.output / f'{task}_development_model.joblib')
        check_X = restored['preprocessing'].transform(validation.iloc[:20][restored['predictors']])
        np.testing.assert_allclose(restored['model'].predict(check_X), frame.prediction.iloc[:20], rtol=1e-10)
        print(f'{task}: full validation and artifact reload check complete', flush=True)
        return cv_scores, tuning, validation_scores
