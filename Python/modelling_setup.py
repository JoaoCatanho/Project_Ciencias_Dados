"""Part 2 data contracts, reproducible experiment settings, and dummy baselines."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support,
                             r2_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

from .model_selection import MonthlyTimeSeriesCV
from .preprocessing import LendingClubPreprocessor


@dataclass
class ExperimentConfig:
    """Initial budget; later changes must be recorded before examining test results."""
    seed: int = 42
    n_splits: int = 5
    classification_cv_min_train_end: str = '2014-02'
    search_candidates: int = 12
    expensive_training_cap: int = 50000
    preprocessing_check_train: int = 5000
    preprocessing_check_validation: int = 1000
    regression_selection_metric: str = 'rmse'
    classification_selection_metric: str = 'f1_macro'
    neural_network_seeds: tuple = (42, 43, 44)
    neural_network_max_epochs: int = 100
    neural_network_patience: int = 10

    def to_dict(self):
        return asdict(self)


class ModellingDataSetup:
    """Load allowlisted development columns only; keep held-out outcomes unread."""

    def __init__(self, project_root):
        self.root = Path(project_root)
        self.split_config = json.loads((self.root / 'outputs/split_definition.json').read_text())
        self.predictor_plan = json.loads((self.root / 'outputs/predictor_plan.json').read_text())

    def predictors(self, task):
        plan = self.predictor_plan[task]
        columns = plan['numeric'] + plan['categorical']
        if len(columns) != len(set(columns)):
            raise ValueError('Predictor plan contains duplicate columns.')
        banned = {'id', 'issue_d', 'target_class', 'loan_status'}
        if banned.intersection(columns):
            raise ValueError('Identifier, date, or target included as a predictor.')
        pre = LendingClubPreprocessor(plan['numeric'], plan['categorical'], task=task)
        retained = pre.drop_leakage(pd.DataFrame(columns=columns)).columns
        if set(columns) != set(retained):
            raise ValueError('Predictor plan includes excluded fields.')
        return columns

    def load_development(self, task):
        target = 'int_rate' if task == 'regression' else 'target_class'
        columns = self.predictors(task) + [target, 'issue_d']
        end = (pd.Timestamp(self.split_config['val_end']) + pd.offsets.MonthEnd(0)
               + pd.Timedelta(days=1))
        data = pd.read_parquet(
            self.root / f'data/processed/accepted_{task}_fe.parquet',
            columns=columns, filters=[('issue_d', '<', end)],
        ).sort_values('issue_d', kind='stable').reset_index(drop=True)
        if data['issue_d'].isna().any() or data[target].isna().any():
            raise ValueError('Development dates and targets must be complete.')
        if task == 'classification':
            if set(data[target].unique()) != {0, 1, 2}:
                raise ValueError('Expected exactly the three defined classes.')
            data[target] = data[target].astype(int)
        elif not np.isfinite(data[target].to_numpy(dtype=float)).all():
            raise ValueError('Regression target must be finite.')
        cutoff = pd.Timestamp(self.split_config['train_end']) + pd.offsets.MonthEnd(0)
        train = data.loc[data.issue_d <= cutoff].reset_index(drop=True)
        val = data.loc[data.issue_d > cutoff].reset_index(drop=True)
        if train.empty or val.empty:
            raise ValueError('Empty development partition.')
        if not train.issue_d.max() < val.issue_d.min():
            raise ValueError('Training and validation periods overlap.')
        return train, val


class BaselineRunner:
    """Evaluate uninformative predictors on identical rows without preprocessing.

    Dummy models ignore X, so a constant placeholder avoids expensive encoding.
    AUC uses probabilities with fixed class order [0, 1, 2]. Undefined fold AUC
    is recorded as NaN rather than silently dropping or relabelling classes.
    """

    def __init__(self, config=None):
        self.config = config or ExperimentConfig()

    def metrics(self, task, y, prediction, probabilities=None):
        if task == 'regression':
            return dict(mae=mean_absolute_error(y, prediction),
                        rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                        r2=r2_score(y, prediction))
        result = {'accuracy': accuracy_score(y, prediction)}
        for average in ['macro', 'micro']:
            p, r, f, _ = precision_recall_fscore_support(
                y, prediction, labels=[0, 1, 2], average=average, zero_division=0)
            result.update({f'precision_{average}': p, f'recall_{average}': r,
                           f'f1_{average}': f})
        result['auc_macro_ovr'] = result['auc_micro_ovr'] = np.nan
        if probabilities is not None and len(np.unique(y)) == 3:
            binary = label_binarize(y, classes=[0, 1, 2])
            result['auc_macro_ovr'] = roc_auc_score(binary, probabilities, average='macro')
            result['auc_micro_ovr'] = roc_auc_score(binary, probabilities, average='micro')
        return result

    def run(self, task, train, validation):
        target = 'int_rate' if task == 'regression' else 'target_class'
        strategies = ['mean', 'median'] if task == 'regression' else ['most_frequent', 'prior']
        cv = MonthlyTimeSeriesCV(self.config.n_splits,
            min_train_end=self.config.classification_cv_min_train_end if task == 'classification' else None)
        folds = list(cv.split(train, groups=train.issue_d))
        records, matrices = [], {}
        partitions = [(f'cv_{i + 1}', train.iloc[a], train.iloc[b])
                      for i, (a, b) in enumerate(folds)] + [('validation', train, validation)]
        for strategy in strategies:
            for partition, fit_data, score_data in partitions:
                model = (DummyRegressor(strategy=strategy) if task == 'regression'
                         else DummyClassifier(strategy=strategy, random_state=self.config.seed))
                model.fit(np.zeros((len(fit_data), 1)), fit_data[target])
                X = np.zeros((len(score_data), 1))
                pred = model.predict(X)
                prob = None
                if task == 'classification':
                    prob = np.zeros((len(score_data), 3))
                    prob[:, model.classes_.astype(int)] = model.predict_proba(X)
                    if partition == 'validation':
                        matrices[strategy] = confusion_matrix(score_data[target], pred, labels=[0, 1, 2])
                records.append(dict(task=task, model=f'dummy_{strategy}', partition=partition,
                                    n_train=len(fit_data), n_evaluation=len(score_data),
                                    **self.metrics(task, score_data[target], pred, prob)))
        return pd.DataFrame(records), matrices
