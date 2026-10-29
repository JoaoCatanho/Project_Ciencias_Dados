"""Calendar-month expanding validation; indices refer to the original row order."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class MonthlyTimeSeriesCV(BaseCrossValidator):
    """Split unique months into an initial block and expanding validation blocks.

    Pass origination dates as ``groups``. No month is divided between partitions.
    This preserves chronology but does not establish historical label availability.
    """

    def __init__(self, n_splits: int = 5, min_train_end: str | None = None):
        self.n_splits = n_splits
        self.min_train_end = min_train_end

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError('Pass origination dates as groups.')
        dates = pd.Series(pd.to_datetime(groups, errors='coerce')).reset_index(drop=True)
        if len(dates) != len(X) or dates.isna().any():
            raise ValueError('Dates must match X and contain no missing values.')
        months = dates.dt.to_period('M')
        unique = np.sort(months.unique())
        if self.n_splits < 2 or len(unique) < self.n_splits + 1:
            raise ValueError('Need at least n_splits + 1 distinct months and n_splits >= 2.')
        if self.min_train_end is None:
            blocks = np.array_split(unique, self.n_splits + 1)
        else:
            cutoff = pd.Period(self.min_train_end, freq='M')
            initial = unique[unique <= cutoff]
            remaining = unique[unique > cutoff]
            if len(initial) == 0 or len(remaining) < self.n_splits:
                raise ValueError('Need initial training months and at least one month per fold.')
            blocks = [initial] + list(np.array_split(remaining, self.n_splits))
        for i in range(1, len(blocks)):
            train = np.flatnonzero(months.isin(np.concatenate(blocks[:i])))
            val = np.flatnonzero(months.isin(blocks[i]))
            yield train, val
