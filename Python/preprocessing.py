"""
src/preprocessing.py
--------------------
Cleaning and leak-safe transformation helpers.

Important rules enforced here (from the project specification):
* Leakage columns (post-origination) are hard-coded and MUST be dropped
  before any supervised modelling.
* Regression on int_rate must also drop grade, sub_grade, and installment.
* Classification must drop Current loans (unknown outcome).
* Transformations that "learn" from data (imputation, scaling, encoding)
  are exposed as sklearn Pipelines/ColumnTransformers so they can be fit
  on training folds only.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leakage / exclusion sets (from the spec's "Note on data leakage")
# ---------------------------------------------------------------------------
LEAKAGE_EXACT = [
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amt", "last_pymnt_amnt", "next_pymnt_d",
    "deferral_term", "payment_plan_start_date", "orig_projected_additional_accrued_interest",
    "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
    "debt_settlement_flag", "debt_settlement_flag_date",
]

LEAKAGE_PREFIXES = ("hardship_", "settlement_")

# Extra exclusions for the pre-pricing regression task only.
REGRESSION_EXTRA_BANNED = ["grade", "sub_grade", "installment", "int_rate", "loan_status",
    "issue_d", "credit_hist_months", "credit_age_ratio", "verification_status",
    "verification_status_joint", "funded_amnt", "funded_amnt_inv"]

# Extra exclusions for the classification task.
CLASSIFICATION_EXTRA_BANNED = ["loan_status"]

# Administrative fields deemed irrelevant.
ADMIN_IRRELEVANT = ["member_id", "url", "desc", "policy_code", "pymnt_plan"]


def _expand_prefixes(df: pd.DataFrame, prefixes: Sequence[str]) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


def leakage_columns(df: pd.DataFrame) -> list[str]:
    """Return all leakage columns present in df (exact + prefix matches)."""
    cols = [c for c in LEAKAGE_EXACT if c in df.columns]
    cols += _expand_prefixes(df, LEAKAGE_PREFIXES)
    return sorted(set(cols))


# Module-level constants for reuse (the spec names them explicitly).
REGRESSION_BANNED = sorted(set(LEAKAGE_EXACT + REGRESSION_EXTRA_BANNED))
CLASSIFICATION_BANNED = sorted(set(LEAKAGE_EXACT + CLASSIFICATION_EXTRA_BANNED))


# ---------------------------------------------------------------------------
# Target derivation
# ---------------------------------------------------------------------------
GOOD_STATUSES = {"Fully Paid"}
AT_RISK_STATUSES = {"Late (16-30 days)", "Late (31-120 days)", "In Grace Period"}
DEFAULT_STATUSES = {"Charged Off", "Default"}
UNKNOWN_STATUSES = {"Current"}  # must be dropped for classification


def derive_classification_target(loan_status: pd.Series) -> pd.Series:
    """
    Map loan_status -> 3-class target.

    Returns a Series of {0: Good, 1: At risk, 2: Default} with NaN
    for statuses whose outcome is unknown (e.g. 'Current').
    """
    mapping = {}
    for s in GOOD_STATUSES:
        mapping[s] = 0
    for s in AT_RISK_STATUSES:
        mapping[s] = 1
    for s in DEFAULT_STATUSES:
        mapping[s] = 2
    # everything else (Current, unknown) -> NaN
    return loan_status.map(mapping).astype("float").astype("Int64")


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class LendingClubPreprocessor:
    """
    Cleaning + leak-safe transformation orchestration.

    Parameters
    ----------
    numeric_cols : Sequence[str]
        Columns to be scaled.
    categorical_cols : Sequence[str]
        Columns to be one-hot encoded.
    task : {"regression", "classification", "clustering"}
        Determines which exclusion list to apply.

    Notes
    -----
    * All learned transforms (imputer, scaler, encoder) are wrapped in a
      sklearn ColumnTransformer so they can be fit on a training split only.
    * Deterministic parsing (percent removal, date parsing) is done in the
      loader, not here, to keep this class purely about learned transforms.
    """

    def __init__(
        self,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        task: str = "classification",
    ) -> None:
        assert task in {"regression", "classification", "clustering"}
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.task = task
        self._pipeline: Pipeline | None = None

    # --------------------------- cleaning -------------------------------
    def drop_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop leakage columns (and admin fields) appropriate to the task."""
        banned = set(leakage_columns(df)) | set(ADMIN_IRRELEVANT)
        if self.task == "regression":
            banned |= set(REGRESSION_EXTRA_BANNED)
        elif self.task == "classification":
            banned |= set(CLASSIFICATION_EXTRA_BANNED)
        # clustering: only the generic leakage list applies
        drop = [c for c in df.columns if c in banned]
        logger.info("Dropping %d banned columns for task=%s", len(drop), self.task)
        return df.drop(columns=drop, errors="ignore")

    def drop_unknown_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """For classification, drop rows with Current / unknown loan_status."""
        if self.task != "classification":
            return df
        before = len(df)
        df = df[~df["loan_status"].isin(UNKNOWN_STATUSES)].copy()
        logger.info("Dropped %d rows with unknown outcome", before - len(df))
        return df

    # --------------------------- pipeline -------------------------------
    def build_pipeline(self) -> Pipeline:
        """
        Build a leak-safe ColumnTransformer wrapped in a Pipeline.

        Call .fit(X_train) / .transform(X_val) from notebooks.
        """
        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_cols),
                ("cat", categorical_pipe, self.categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self._pipeline = Pipeline(steps=[("pre", pre)])
        return self._pipeline

    def fit(self, X_train: pd.DataFrame) -> "LendingClubPreprocessor":
        if self._pipeline is None:
            self.build_pipeline()
        assert self._pipeline is not None
        self._pipeline.fit(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self._pipeline is not None, "Call fit() first."
        return self._pipeline.transform(X)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)