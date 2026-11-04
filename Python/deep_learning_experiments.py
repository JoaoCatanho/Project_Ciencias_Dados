"""Chronological nested stopping, refitting and evaluation for Notebook 10."""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .deep_learning import (NeuralConfig, NeuralPreprocessor, CalendarEarlyStoppingSplit,
                            NeuralTrainer, NeuralModelBundle)
from .model_selection import MonthlyTimeSeriesCV
from .modelling_setup import ModellingDataSetup, ExperimentConfig, BaselineRunner
from .imbalance_diagnostics import ImbalanceMetrics


class NeuralExperimentRunner:
    """Tune on five original training folds, then assess three final seeds.

    Epochs are chosen on the latest month INSIDE the fitting partition. Then fit
    fresh preprocessing/target statistics and a fresh network on that complete
    partition for exactly that many epochs. No outer evaluation labels select
    epochs. Regression is capped at 50k seeded fitting rows; classification keeps
    every fitting row. All stopping and evaluation rows are retained.
    """
    def __init__(self, root, output_dir, config=None):
        self.root, self.output = Path(root), Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'histories').mkdir(exist_ok=True)
        (self.output / 'samples').mkdir(exist_ok=True)
        self.config = config or NeuralConfig()
        self.setup, self.protocol = ModellingDataSetup(root), ExperimentConfig()
        self.classification_metrics, self.regression_metrics = ImbalanceMetrics(), BaselineRunner()

    def candidates(self, task):
        if task == 'regression':
            return [dict(learning_rate=float(rate), class_weight='none') for rate in self.config.regression_rates]
        return [dict(learning_rate=0.001, class_weight=weight) for weight in self.config.classification_weights]

    def _sample(self, frame, task, seed, label):
        if task == 'regression' and len(frame) > self.config.regression_training_cap:
            indices = np.sort(np.random.default_rng(seed).choice(len(frame), self.config.regression_training_cap, replace=False))
            frame = frame.iloc[indices]
        if task == 'regression':
            pd.DataFrame({'training_row': frame.index}).to_csv(self.output / 'samples' / f'{label}.csv', index=False)
        return frame

    def _preprocessor(self, task):
        plan = self.setup.predictor_plan[task]
        return NeuralPreprocessor(plan['numeric'], plan['categorical'], task, self.config)

    def _score(self, task, y, prediction):
        return (self.regression_metrics.metrics(task, y, prediction) if task == 'regression'
                else self.classification_metrics.score(y, prediction))

    def fit_partition(self, task, training, evaluation, parameters, seed, partition):
        start = perf_counter()
        target = 'int_rate' if task == 'regression' else 'target_class'
        inner_fit, stopping = CalendarEarlyStoppingSplit().split(training, task)
        key = f'{task}_{partition}_seed{seed}_lr{parameters["learning_rate"]}_{parameters["class_weight"]}'
        fitting = self._sample(inner_fit, task, seed, key + '_inner')
        inner_pre = self._preprocessor(task).fit(fitting)
        X, stop_X = inner_pre.transform(fitting), inner_pre.transform(stopping)
        trainer = NeuralTrainer(task, seed=seed, config=self.config, **parameters)
        trainer.fit(X, fitting[target], stopping=(stop_X, stopping[target]))
        epochs = trainer.best_epoch_
        pd.DataFrame(trainer.history_).to_csv(self.output / 'histories' / f'{key}_stopping.csv', index=False)
        inner_record = dict(inner_available=len(inner_fit), inner_fitted=len(fitting), n_stopping=len(stopping),
            inner_end=str(inner_fit.issue_d.max()), stopping_start=str(stopping.issue_d.min()),
            stopping_end=str(stopping.issue_d.max()), selected_epochs=epochs, stopping_epochs_run=len(trainer.history_))
        if task == 'classification':
            for label in [0, 1, 2]:
                inner_record[f'inner_class_{label}'] = int(fitting[target].eq(label).sum())
                inner_record[f'stopping_class_{label}'] = int(stopping[target].eq(label).sum())
        del X, stop_X, inner_pre, trainer
        # Fresh refit uses every available classification row, including stopping
        # rows, only after epochs have been fixed on the inner calendar holdout.
        final_fit = self._sample(training, task, seed, key + '_refit')
        pre = self._preprocessor(task).fit(final_fit)
        X = pre.transform(final_fit)
        final_trainer = NeuralTrainer(task, seed=seed, config=self.config, **parameters)
        final_trainer.fit(X, final_fit[target], fixed_epochs=epochs)
        pd.DataFrame(final_trainer.history_).to_csv(self.output / 'histories' / f'{key}_refit.csv', index=False)
        bundle = NeuralModelBundle(pre, final_trainer)
        prediction = bundle.predict_frame(evaluation)
        row = dict(task=task, partition=partition, seed=seed, parameters=json.dumps(parameters, sort_keys=True),
            n_train_available=len(training), n_train=len(final_fit), n_evaluation=len(evaluation),
            n_features=X.shape[1], fit_predict_seconds=perf_counter() - start, **inner_record,
            **self._score(task, evaluation[target], prediction))
        if task == 'classification':
            for label in [0, 1, 2]:
                row[f'n_train_class_{label}'] = int(final_fit[target].eq(label).sum())
        print(f'{key}: selected {epochs} epochs; '
              f'{"RMSE=" + format(row["rmse"], ".3f") if task == "regression" else "macro F1=" + format(row["f1_macro"], ".3f")}', flush=True)
        return row, bundle, prediction

    def run(self, task):
        with threadpool_limits(limits=self.config.threads):
            return self._run(task)

    def _run(self, task):
        train, validation = self.setup.load_development(task)
        cv = MonthlyTimeSeriesCV(self.protocol.n_splits,
            self.protocol.classification_cv_min_train_end if task == 'classification' else None)
        rows = []
        for fold, (a, b) in enumerate(cv.split(train, groups=train.issue_d), 1):
            for parameters in self.candidates(task):
                row, _, _ = self.fit_partition(task, train.iloc[a], train.iloc[b], parameters,
                                                self.config.cv_seed, f'cv_{fold}')
                rows.append(row)
                pd.DataFrame(rows).to_csv(self.output / f'{task}_cv_scores.csv', index=False)
        cv_scores = pd.DataFrame(rows)
        metric = 'rmse' if task == 'regression' else 'f1_macro'
        tuning = cv_scores.groupby('parameters')[metric].agg(['mean', 'std', 'count']).reset_index()
        tuning = tuning.sort_values(['mean', 'parameters'], ascending=[task == 'regression', True])
        tuning.to_csv(self.output / f'{task}_tuning.csv', index=False)
        selected = json.loads(tuning.iloc[0].parameters)
        (self.output / f'{task}_selection.json').write_text(json.dumps(
            {'parameters': selected, 'source': 'Training CV only', 'metric': metric}, indent=2))
        scores = []
        target = 'int_rate' if task == 'regression' else 'target_class'
        for seed in self.config.final_seeds:
            row, bundle, prediction = self.fit_partition(task, train, validation, selected, seed, 'validation')
            directory = self.output / 'models' / f'{task}_seed{seed}'
            metadata = dict(task=task, parameters=selected, seed=seed, configuration=self.config.to_dict(),
                split=self.setup.split_config, predictors=self.setup.predictor_plan[task],
                n_train=row['n_train'], n_train_available=row['n_train_available'], selected_epochs=row['selected_epochs'],
                stage='Training-only development artifact; final test unread',
                feature_policy='Original allowlist; train-fitted 0.5–99.5% clipping and declared log1p',
                epoch_policy='Inner last-month early stopping, fresh full-partition fixed-epoch refit')
            bundle.save(directory, metadata)
            restored = NeuralModelBundle.load(directory)
            np.testing.assert_allclose(restored.predict_frame(validation.iloc[:20]), prediction[:20], rtol=1e-6, atol=1e-6)
            frame = validation[['issue_d', target]].copy()
            frame['prediction'] = prediction if task == 'regression' else prediction.argmax(axis=1)
            if task == 'classification':
                for label in [0, 1, 2]:
                    frame[f'probability_{label}'] = prediction[:, label]
            frame.to_parquet(self.output / f'{task}_seed{seed}_validation_predictions.parquet', index=False)
            scores.append(row)
            pd.DataFrame(scores).to_csv(self.output / f'{task}_validation_scores.csv', index=False)
        return cv_scores, tuning, pd.DataFrame(scores)
