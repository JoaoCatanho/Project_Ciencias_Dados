"""Bagging and histogram boosting with the shared fold-aware experiment runner.

The feature plan and preprocessing remain identical to Notebook 08. Random
forests bootstrap observations and randomize split features; histogram gradient
boosting builds trees sequentially. No test rows or evaluation-driven repairs
are introduced here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from threadpoolctl import threadpool_limits

from .supervised_models import SupervisedExperimentRunner


@dataclass
class EnsembleExperimentConfig:
    seed: int = 42
    blas_threads: int = 2
    forest_jobs: int = 2
    forest_trees: int = 64
    forest_depths: tuple = (12, 20)
    forest_min_samples_leaf: int = 100
    forest_max_features: float = 0.5
    forest_max_samples: float = 0.5
    boosting_iterations: int = 100
    boosting_learning_rates: tuple = (0.05, 0.1)
    boosting_leaf_nodes: tuple = (15, 31)
    boosting_classification_learning_rate: float = 0.1
    boosting_min_samples_leaf: int = 100
    boosting_l2: float = 1.0
    class_weights: tuple = (None, 'balanced')

    def to_dict(self):
        return asdict(self)


class EnsembleModelFactory:
    """Two bagging candidates + four boosting candidates for regression.

    Classification compares four forests and four boosting configurations,
    including both unweighted and training-derived balanced class weights.
    """

    def __init__(self, config=None):
        self.config = config or EnsembleExperimentConfig()

    def candidates(self, task):
        c = self.config
        if task == 'regression':
            return ([('random_forest', {'max_depth': d}) for d in c.forest_depths]
                    + [('hist_gradient_boosting', {'learning_rate': rate, 'max_leaf_nodes': leaves})
                       for rate in c.boosting_learning_rates for leaves in c.boosting_leaf_nodes])
        if task == 'classification':
            return ([('random_forest', {'max_depth': d, 'class_weight': weight})
                     for d in c.forest_depths for weight in c.class_weights]
                    + [('hist_gradient_boosting', {'learning_rate': c.boosting_classification_learning_rate,
                                                   'max_leaf_nodes': leaves, 'class_weight': weight})
                       for leaves in c.boosting_leaf_nodes for weight in c.class_weights])
        raise ValueError('task must be regression or classification')

    def build(self, task, family, parameters):
        c = self.config
        if task not in ('regression', 'classification'):
            raise ValueError('Unsupported ensemble task')
        if family == 'random_forest':
            cls = RandomForestRegressor if task == 'regression' else RandomForestClassifier
            return cls(n_estimators=c.forest_trees, bootstrap=True,
                       max_samples=c.forest_max_samples, max_features=c.forest_max_features,
                       min_samples_leaf=c.forest_min_samples_leaf,
                       n_jobs=c.forest_jobs, random_state=c.seed, **parameters)
        if family == 'hist_gradient_boosting':
            cls = HistGradientBoostingRegressor if task == 'regression' else HistGradientBoostingClassifier
            # Disable random internal early-stopping splits: the outer calendar
            # folds alone determine comparisons. Iteration budget is fixed.
            return cls(max_iter=c.boosting_iterations, early_stopping=False,
                       min_samples_leaf=c.boosting_min_samples_leaf,
                       l2_regularization=c.boosting_l2, random_state=c.seed, **parameters)
        raise ValueError('Unsupported ensemble family')


class EnsembleExperimentRunner(SupervisedExperimentRunner):
    """Reuse full-cohort CV, selection, persistence, and save/reload checks.

    Restrict OpenMP as well as BLAS so histogram boosting does not oversubscribe
    the machine. In inherited tables, ``converged`` means no ConvergenceWarning,
    not proof that a fixed-budget ensemble reached an optimal loss.
    """

    def __init__(self, setup, output_dir, config=None, protocol=None):
        config = config or EnsembleExperimentConfig()
        super().__init__(setup, output_dir, config=config, protocol=protocol)
        self.factory = EnsembleModelFactory(config)

    def run(self, task):
        with threadpool_limits(limits=self.config.blas_threads):
            return super().run(task)

    def fit_and_score(self, task, family, parameters, X, y, Z, y_evaluation):
        with threadpool_limits(limits=self.config.blas_threads):
            row, model, prediction, probabilities = super().fit_and_score(
                task, family, parameters, X, y, Z, y_evaluation)
        row['trees_fitted'] = (len(model.estimators_) if family == 'random_forest'
                               else int(model.n_iter_ * model.n_trees_per_iteration_))
        row['budget_completed'] = (len(model.estimators_) == self.config.forest_trees
                                   if family == 'random_forest'
                                   else model.n_iter_ == self.config.boosting_iterations)
        return row, model, prediction, probabilities

    def _save_interpretation(self, task, family, model, pre):
        if family == 'random_forest':
            super()._save_interpretation(task, family, model, pre)
        else:
            # HistGradientBoosting does not expose native impurity importances.
            # Report fitted structure rather than inventing an importance score.
            pd.DataFrame([
                {'property': 'iterations', 'value': model.n_iter_},
                {'property': 'trees_per_iteration', 'value': model.n_trees_per_iteration_},
                {'property': 'encoded_features', 'value': model.n_features_in_},
                {'property': 'early_stopping', 'value': model.early_stopping},
            ]).to_csv(self.output / f'{task}_{family}_structure.csv', index=False)
