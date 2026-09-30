"""
src/data_loader.py
------------------
Object-oriented loaders for the Lending Club accepted/rejected CSV files.

Design goals
------------
* OOP (per project spec: Python code must be object oriented).
* Reproducible loading, with a documented schema report.
* Deterministic parsing only (percent signs, dates with fixed formats).
  No data-dependent transformations (no imputation, no scaling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema report container
# ---------------------------------------------------------------------------
@dataclass
class SchemaReport:
    """Container for a per-column schema summary."""
    name: str
    n_rows: int
    n_cols: int
    dtypes: pd.Series
    missing: pd.Series
    missing_pct: pd.Series
    n_unique: pd.Series
    duplicate_rows: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "dtype": self.dtypes.astype(str),
            "missing": self.missing,
            "missing_pct": self.missing_pct.round(4),
            "n_unique": self.n_unique,
        }).sort_values("missing_pct", ascending=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class LendingClubLoader:
    """
    Load the accepted and rejected Lending Club CSVs.

    Parameters
    ----------
    accepted_path : str | Path
        Path to accepted_2007_to_2018Q4.csv
    rejected_path : str | Path
        Path to rejected_2007_to_2018Q4.csv
    usecols : Iterable[str] | None
        Optional column subset for the accepted file (large file).
    nrows : int | None
        Optional row cap for quick dev loops.
    """

    # Columns whose values are strings like "13.5%" that must be numeric.
    PERCENT_COLS_ACCEPTED = ("int_rate", "revol_util", "bc_util")

    # Rejected-file columns (names as published on Kaggle).
    REJECTED_COLS = (
        "Amount Requested",
        "Application Date",
        "Loan Title",
        "Risk_Score",
        "Debt-To-Income Ratio",
        "Zip Code",
        "State",
        "Employment Length",
        "Policy Code",
    )

    def __init__(
        self,
        accepted_path: str | Path,
        rejected_path: str | Path,
        usecols: Optional[Iterable[str]] = None,
        nrows: Optional[int] = None,
    ) -> None:
        self.accepted_path = Path(accepted_path)
        self.rejected_path = Path(rejected_path)
        self.usecols = list(usecols) if usecols is not None else None
        self.nrows = nrows

    # ----------------------------- public API -----------------------------
    def load_accepted(self) -> pd.DataFrame:
        """Load and deterministically parse the accepted-loans CSV."""
        logger.info("Loading accepted file: %s", self.accepted_path)
        df = pd.read_csv(
            self.accepted_path,
            usecols=self.usecols,
            nrows=self.nrows,
            low_memory=False,
        )
        df = self._parse_accepted(df)
        logger.info("Accepted shape: %s", df.shape)
        return df

    def load_rejected(self) -> pd.DataFrame:
        """Load and deterministically parse the rejected-applications CSV."""
        logger.info("Loading rejected file: %s", self.rejected_path)
        df = pd.read_csv(
            self.rejected_path,
            nrows=self.nrows,
            low_memory=False,
        )
        df = self._parse_rejected(df)
        logger.info("Rejected shape: %s", df.shape)
        return df

    def schema_report(self, df: pd.DataFrame, name: str) -> SchemaReport:
        """Produce a per-column schema summary for a dataframe."""
        return SchemaReport(
            name=name,
            n_rows=len(df),
            n_cols=df.shape[1],
            dtypes=df.dtypes,
            missing=df.isna().sum(),
            missing_pct=(df.isna().mean() * 100).round(4),
            n_unique=df.nunique(dropna=True),
            duplicate_rows=int(df.duplicated().sum()),
        )

    # ----------------------------- internals ------------------------------
    def _parse_accepted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic parsing of the accepted file."""
        df = df.copy()

        # term: " 36 months" -> 36
        if "term" in df.columns:
            df["term"] = (
                df["term"].astype(str).str.extract(r"(\d+)").astype(float)
            )

        # percent-string columns -> numeric
        for col in self.PERCENT_COLS_ACCEPTED:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str).str.replace("%", "", regex=False)
                    .replace({"nan": np.nan}).astype(float)
                )

        # dates: fixed formats (deterministic parsing, not learned)
        if "issue_d" in df.columns:
            df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        if "earliest_cr_line" in df.columns:
            df["earliest_cr_line"] = pd.to_datetime(
                df["earliest_cr_line"], format="%b-%Y", errors="coerce"
            )
        for col in ("last_pymnt_d", "next_pymnt_d", "last_credit_pull_d"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")

        return df

    def _parse_rejected(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic parsing of the rejected file."""
        df = df.copy()

        # Debt-To-Income Ratio: "10%" -> 10.0
        if "Debt-To-Income Ratio" in df.columns:
            df["Debt-To-Income Ratio"] = (
                df["Debt-To-Income Ratio"].astype(str)
                .str.replace("%", "", regex=False)
                .replace({"nan": np.nan})
                .astype(float)
            )

        # Risk_Score: 0.0 means missing (per the project spec)
        if "Risk_Score" in df.columns:
            df["Risk_Score"] = df["Risk_Score"].replace({0.0: np.nan, 0: np.nan})

        # Application Date -> datetime
        if "Application Date" in df.columns:
            df["Application Date"] = pd.to_datetime(
                df["Application Date"], errors="coerce"
            )

        # Zip Code -> string (leading zeros matter for geography)
        if "Zip Code" in df.columns:
            df["Zip Code"] = df["Zip Code"].astype(str).str.zfill(5)

        return df