"""
src/splits.py
-------------
Chronological split utilities.

The project spec requires:
* Splits defined BEFORE any data-dependent preprocessing.
* Explicit calendar cutoffs (not file order).
* Keep same-month records together.
* Final test set preserved for final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class ChronologicalSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cutoffs: dict


def chronological_split(
    df: pd.DataFrame,
    date_col: str,
    train_end: str,
    val_end: str,
) -> ChronologicalSplits:
    """
    Split a dataframe by calendar cutoffs.

    Parameters
    ----------
    df : DataFrame
        Must contain `date_col` as datetime64.
    date_col : str
        Column with the reference date (e.g. 'issue_d').
    train_end : str
        Inclusive last month for training, e.g. '2014-12'.
    val_end : str
        Inclusive last month for validation, e.g. '2016-12'.
        Everything after val_end becomes test.

    Returns
    -------
    ChronologicalSplits with .train, .val, .test and .cutoffs.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    train_end_ts = pd.Timestamp(train_end) + pd.offsets.MonthEnd(0)
    val_end_ts = pd.Timestamp(val_end) + pd.offsets.MonthEnd(0)

    train = df[df[date_col] <= train_end_ts].copy()
    val = df[(df[date_col] > train_end_ts) & (df[date_col] <= val_end_ts)].copy()
    test = df[df[date_col] > val_end_ts].copy()

    return ChronologicalSplits(
        train=train,
        val=val,
        test=test,
        cutoffs={
            "train_end": train_end_ts.isoformat(),
            "val_end": val_end_ts.isoformat(),
            "test_start": (val_end_ts + pd.offsets.MonthBegin(1)).isoformat(),
        },
    )