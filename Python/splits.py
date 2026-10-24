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

import pandas as pd


@dataclass
class ChronologicalSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cutoffs: dict


class ChronologicalSplitter:
    """Apply the same calendar split policy to multiple dataframes.

    Cutoffs include the entire specified month. Missing dates are excluded,
    matching the original split behavior. Inputs are copied, never mutated.
    """

    def __init__(self, date_col: str, train_end: str, val_end: str) -> None:
        self.date_col = date_col
        self.train_end = train_end
        self.val_end = val_end

    def split(self, df: pd.DataFrame) -> ChronologicalSplits:
        """Return train, validation, and test data with the applied cutoffs."""
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")

        train_end_ts = pd.Timestamp(self.train_end) + pd.offsets.MonthEnd(0)
        val_end_ts = pd.Timestamp(self.val_end) + pd.offsets.MonthEnd(0)

        train = df[df[self.date_col] <= train_end_ts].copy()
        val = df[(df[self.date_col] > train_end_ts) & (df[self.date_col] <= val_end_ts)].copy()
        test = df[df[self.date_col] > val_end_ts].copy()

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

    def development(self, df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
        """Select records through the validation month; exclude missing dates.

        An alternate date column aligns rejected applications with the same
        development cutoff. No outcome filtering is performed here.
        """
        dates = pd.to_datetime(df[date_col or self.date_col], errors="coerce")
        cutoff = pd.Timestamp(self.val_end).to_period("M")
        return df.loc[dates.dt.to_period("M") <= cutoff].copy()
