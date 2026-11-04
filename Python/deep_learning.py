"""PyTorch tabular MLPs, training-only conditioning, and calendar early stopping.

Preserves the original task allowlists, chronological split and three classes.
No grade/joint-income extension or resampling is introduced. Numerical treatment
is an explicitly declared, training-fitted revision motivated by prior audits.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from .preprocessing import LendingClubPreprocessor


@dataclass
class NeuralConfig:
    hidden: tuple = (128, 64)
    dropout: float = 0.2
    weight_decay: float = 1e-4
    batch_size: int = 2048
    prediction_batch_size: int = 32768
    max_epochs: int = 60
    patience: int = 8
    min_delta: float = 1e-4
    gradient_clip: float = 5.0
    threads: int = 2
    cv_seed: int = 42
    final_seeds: tuple = (42, 43, 44)
    regression_rates: tuple = (0.001, 0.0003)
    classification_weights: tuple = ('none', 'balanced')
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    regression_training_cap: int = 50000

    def to_dict(self):
        return asdict(self)


class NeuralPreprocessor:
    """Fit winsorization, log1p, imputation and encoding on fitting rows only.

    Nonnegative log inputs with negative values become NaN. Missing fields are
    not replaced with zero. The allowlists are supplied by the frozen task plan.
    All-missing numeric fields retain the existing imputer's fitted drop policy.
    """
    LOG_COLUMNS = ('annual_inc', 'loan_amnt', 'revol_bal', 'tot_cur_bal', 'tot_hi_cred_lim',
                   'bc_open_to_buy', 'loan_to_income', 'dti_x_loan_to_income', 'bc_pressure',
                   'installment_burden', 'loan_per_term', 'rate_burden')

    def __init__(self, numeric, categorical, task, config=None):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.task, self.config = task, config or NeuralConfig()
        probe = pd.DataFrame(columns=self.numeric + self.categorical)
        guard = LendingClubPreprocessor(self.numeric, self.categorical, task)
        if set(guard.drop_leakage(probe)) != set(probe) or {'target_class', 'loan_status', 'issue_d', 'id'} & set(probe):
            raise ValueError('Ineligible neural model predictor')

    def _prepare(self, data):
        frame = data[self.numeric + self.categorical].copy()
        frame[self.numeric] = frame[self.numeric].replace([np.inf, -np.inf], np.nan)
        for col in set(self.numeric) & set(self.LOG_COLUMNS):
            frame[col] = frame[col].where(frame[col] >= 0)
        return frame

    def _condition(self, frame):
        frame = frame.copy()
        frame[self.numeric] = frame[self.numeric].clip(lower=self.bounds_.iloc[0], upper=self.bounds_.iloc[1], axis=1)
        for col in set(self.numeric) & set(self.LOG_COLUMNS):
            frame[col] = np.log1p(frame[col])
        return frame

    def fit(self, data):
        frame = self._prepare(data)
        self.bounds_ = frame[self.numeric].quantile([self.config.lower_quantile, self.config.upper_quantile])
        self.preprocessor_ = LendingClubPreprocessor(self.numeric, self.categorical, self.task)
        self.preprocessor_.fit(self._condition(frame))
        return self

    def transform(self, data):
        matrix = self.preprocessor_.transform(self._condition(self._prepare(data)))
        matrix = np.asarray(matrix, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite neural input')
        return matrix


class CalendarEarlyStoppingSplit:
    """Hold out the final unique month inside a fitting partition.

    All classification classes must occur in both sides; fail rather than move
    a future outer-fold/validation observation into early stopping. The first
    classification fold has 13 At risk fitting and 20 stopping examples.
    """
    def split(self, frame, task):
        months = pd.to_datetime(frame.issue_d).dt.to_period('M')
        if months.nunique() < 2 or months.isna().any():
            raise ValueError('Need at least two complete training months')
        last = months.max()
        fit, stop = frame.loc[months < last], frame.loc[months == last]
        if task == 'classification':
            for portion in [fit, stop]:
                if set(portion.target_class.unique()) != {0, 1, 2}:
                    raise ValueError('Early-stopping fitting and scoring must each contain all three classes')
        return fit, stop


class TabularMLP(nn.Module):
    """Two ReLU/dropout hidden layers and a task-specific linear head.

    Classification returns logits, never softmax inside the training network.
    Regression returns one standardized target, inverted by the trainer.
    """
    def __init__(self, input_dim, task, hidden=(128, 64), dropout=0.2):
        super().__init__()
        if task not in ('regression', 'classification'):
            raise ValueError('Unknown task')
        self.task = task
        layers, previous = [], input_dim
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, 1 if task == 'regression' else 3))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


class NeuralTrainer:
    """Deterministic CPU AdamW training with optional calendar stopping data.

    Balanced CE weights use fitting labels only. Weighted unreduced CE is averaged
    over row count, preserving the global weighted objective across minibatches
    instead of normalizing by a different random class-weight sum per batch.
    Target standardization for regression uses fitting labels only.
    """
    def __init__(self, task, learning_rate=0.001, class_weight='none', seed=42, config=None):
        self.task, self.learning_rate = task, learning_rate
        self.class_weight, self.seed = class_weight, seed
        self.config = config or NeuralConfig()

    def _initialize(self, X, y):
        c = self.config
        torch.set_num_threads(c.threads)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.input_dim_ = X.shape[1]
        self.model_ = TabularMLP(self.input_dim_, self.task, c.hidden, c.dropout)
        self.target_mean_, self.target_scale_ = 0., 1.
        self.class_weights_ = None
        if self.task == 'regression':
            self.target_mean_ = float(np.mean(y))
            self.target_scale_ = max(float(np.std(y)), 1e-6)
        else:
            counts = np.bincount(np.asarray(y, dtype=int), minlength=3)
            if np.any(counts == 0):
                raise ValueError('Every fitting class must be represented')
            if self.class_weight == 'balanced':
                self.class_weights_ = torch.tensor(len(y) / (3 * counts), dtype=torch.float32)
            elif self.class_weight != 'none':
                raise ValueError('Unsupported class weighting')

    def _target(self, y):
        if self.task == 'regression':
            return torch.tensor((np.asarray(y, dtype=np.float32) - self.target_mean_) / self.target_scale_).reshape(-1, 1)
        return torch.tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)

    def _loss(self, output, target):
        if self.task == 'regression':
            return nn.functional.mse_loss(output, target)
        return nn.functional.cross_entropy(output, target, weight=self.class_weights_, reduction='none').mean()

    def _evaluation_loss(self, X, y):
        self.model_.eval()
        total = 0.
        with torch.no_grad():
            for start in range(0, len(X), self.config.prediction_batch_size):
                out = self.model_(X[start:start + self.config.prediction_batch_size])
                loss = self._loss(out, y[start:start + self.config.prediction_batch_size])
                total += float(loss) * len(out)
        return total / len(X)

    def fit(self, X, y, stopping=None, fixed_epochs=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Neural fitting arrays must be finite')
        self._initialize(X, y)
        tensor, target = torch.from_numpy(np.ascontiguousarray(X)), self._target(y)
        if stopping is not None:
            stop_X = torch.from_numpy(np.ascontiguousarray(stopping[0], dtype=np.float32))
            stop_y = self._target(stopping[1])
        elif fixed_epochs is None:
            raise ValueError('Provide training-only stopping data or a fixed epoch count')
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)
        generator = torch.Generator().manual_seed(self.seed)
        best_loss, stale, best_state = np.inf, 0, None
        self.best_epoch_, self.history_ = 0, []
        epochs = self.config.max_epochs if fixed_epochs is None else int(fixed_epochs)
        if epochs < 1:
            raise ValueError('At least one epoch required')
        for epoch in range(1, epochs + 1):
            self.model_.train()
            order = torch.randperm(len(tensor), generator=generator)
            total = 0.
            for start in range(0, len(tensor), self.config.batch_size):
                idx = order[start:start + self.config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self.model_(tensor[idx]), target[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite neural training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), self.config.gradient_clip)
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            value = self._evaluation_loss(stop_X, stop_y) if stopping is not None else np.nan
            self.history_.append(dict(epoch=epoch, training_loss=total / len(tensor), stopping_loss=value))
            if epoch % 10 == 0:
                phase = 'inner' if stopping is not None else 'refit'
                stop_text = f', stopping={value:.4f}' if stopping is not None else ''
                print(f'MLP {self.task}/{phase}/seed{self.seed}: epoch {epoch}, loss={total / len(tensor):.4f}{stop_text}', flush=True)
            if stopping is not None:
                if value < best_loss - self.config.min_delta:
                    best_loss, self.best_epoch_, stale = value, epoch, 0
                    best_state = copy.deepcopy(self.model_.state_dict())
                else:
                    stale += 1
                if stale >= self.config.patience:
                    break
            else:
                self.best_epoch_ = epoch
        if stopping is not None:
            if best_state is None:
                raise RuntimeError('No valid early-stopping checkpoint')
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        outputs = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(X), self.config.prediction_batch_size):
                out = self.model_(torch.from_numpy(np.ascontiguousarray(X[start:start + self.config.prediction_batch_size])))
                if self.task == 'classification':
                    out = out.softmax(dim=1)
                else:
                    out = out * self.target_scale_ + self.target_mean_
                outputs.append(out.numpy())
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise RuntimeError('Nonfinite neural predictions')
        return result[:, 0] if self.task == 'regression' else result

    def save(self, path):
        torch.save(dict(state_dict=self.model_.state_dict(), input_dim=self.input_dim_, task=self.task,
            learning_rate=self.learning_rate, class_weight=self.class_weight, seed=self.seed,
            configuration=self.config.to_dict(), target_mean=self.target_mean_, target_scale=self.target_scale_,
            class_weights=None if self.class_weights_ is None else self.class_weights_.tolist(), best_epoch=self.best_epoch_), path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        trainer = cls(saved['task'], saved['learning_rate'], saved['class_weight'], saved['seed'], NeuralConfig(**saved['configuration']))
        torch.set_num_threads(trainer.config.threads)
        trainer.input_dim_ = saved['input_dim']
        trainer.model_ = TabularMLP(trainer.input_dim_, trainer.task, trainer.config.hidden, trainer.config.dropout)
        trainer.model_.load_state_dict(saved['state_dict']); trainer.model_.eval()
        trainer.target_mean_, trainer.target_scale_ = saved['target_mean'], saved['target_scale']
        trainer.class_weights_ = None if saved['class_weights'] is None else torch.tensor(saved['class_weights'])
        trainer.best_epoch_, trainer.history_ = saved['best_epoch'], []
        return trainer


class NeuralModelBundle:
    def __init__(self, preprocessing, trainer):
        self.preprocessing, self.trainer = preprocessing, trainer

    def predict_frame(self, frame):
        outputs = []
        for start in range(0, len(frame), self.trainer.config.prediction_batch_size):
            X = self.preprocessing.transform(frame.iloc[start:start + self.trainer.config.prediction_batch_size])
            outputs.append(self.trainer.predict(X))
        return np.concatenate(outputs)

    def save(self, directory, metadata):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.trainer.save(directory / 'model.pt')
        joblib.dump(self.preprocessing, directory / 'preprocessing.joblib', compress=3)
        (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        self.preprocessing.bounds_.T.to_csv(directory / 'numeric_training_bounds.csv')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        return cls(joblib.load(directory / 'preprocessing.joblib'), NeuralTrainer.load(directory / 'model.pt'))
