"""
src/feature_engineering.py
--------------------------
Object-oriented feature engineering for the Lending Club project.

Design rules enforced here (from the specification):
* Every engineered feature is derived ONLY from columns eligible at the
  prediction time of the relevant task.
* No feature is derived from post-origination fields, `grade`, `sub_grade`,
  or `installment` (for the regression task).
* No feature is derived from leakage-adjacent signals such as
  "has_recovery", "hardship_count", "installment_to_income".
* Every learned transformation (imputation, scaling, encoding, clipping)
  must be fit on the training split only. This module only creates
  *deterministic* derived columns. Learned transforms are added later
  inside sklearn Pipelines in Phase 4.
* Each feature carries a specification record so the notebook can produce
  the required deliverable table (source, formula, rationale, missing-value
  handling, eligibility per task).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature specification record
# ---------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    name: str
    source_columns: list
    formula: str
    rationale: str
    missing_handling: str
    eligible_regression: bool
    eligible_classification: bool
    eligible_clustering: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_columns"] = ", ".join(d["source_columns"])
        return d


# ---------------------------------------------------------------------------
# Engineer
# ---------------------------------------------------------------------------
class LendingClubFeatureEngineer:
    """
    Create deterministic engineered features for the Lending Club dataset.

    Parameters
    ----------
    reference_date : str | pd.Timestamp, optional
        The reference date used to compute credit-history length.
        Per the specification:
        - For the origination classifier, use `issue_d` (per row).
        - For the pre-pricing regression, use a defensible pre-pricing
          reference date. Funding-date history is excluded from regression.
        If None, per-row `issue_d` is used only for classification/clustering.

    Notes
    -----
    The class does not fit anything. It only adds deterministic columns.
    Learned transforms (imputation, scaling, clipping) live inside sklearn
    Pipelines in Phase 4 and are fit on the training split only.
    """

    def __init__(self, reference_date: Optional[str | pd.Timestamp] = None) -> None:
        self.reference_date = (
            pd.Timestamp(reference_date) if reference_date is not None else None
        )
        self.specs: list[FeatureSpec] = []

    # ------------------------- public API -------------------------------
    def transform(self, df: pd.DataFrame, task: str) -> pd.DataFrame:
        """
        Add engineered features to df for the given task.

        task ∈ {"regression", "classification", "clustering"}
        """
        if task not in {"regression", "classification", "clustering"}:
            raise ValueError(f"Unknown task: {task}")

        df = df.copy()
        self.specs = []

        # ---- Core engineered features (all tasks) ---------------------
        self._add_loan_to_income(df, task)
        self._add_dti_x_loan_to_income(df, task)
        self._add_credit_hist_months(df, task)
        self._add_revol_util_bucket(df, task)
        self._add_total_credit_lines(df, task)
        self._add_inquiries_per_open_account(df, task)
        self._add_util_x_fico(df, task)
        self._add_bc_pressure(df, task)
        self._add_delinq_rate(df, task)
        self._add_loan_per_term(df, task)
        self._add_emp_length_years(df, task)
        self._add_pub_rec_bankruptcies_flag(df, task)
        self._add_closed_credit_share(df, task)
        self._add_installment_burden(df, task)      # not for regression
        self._add_rate_burden(df, task)             # not for regression

        # Never expose ineligible derived columns as candidate predictors.
        ineligible = [spec.name for spec in self.specs
                      if not getattr(spec, f"eligible_{task}")]
        return df.drop(columns=ineligible, errors="ignore")

    def specification_frame(self) -> pd.DataFrame:
        """Return the ≥10-feature specification as a DataFrame."""
        return pd.DataFrame([s.to_dict() for s in self.specs])

    # ------------------------- feature builders -------------------------
    def _add_loan_to_income(self, df: pd.DataFrame, task: str) -> None:
        name = "loan_to_income"
        if {"loan_amnt", "annual_inc"}.issubset(df.columns):
            denom = df["annual_inc"].replace(0, np.nan)
            df[name] = df["loan_amnt"] / denom
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["loan_amnt", "annual_inc"],
            formula="loan_amnt / annual_inc",
            rationale="Captures affordability: how large the requested loan is "
                      "relative to the borrower's income.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
            notes="annual_inc == 0 mapped to NaN to avoid division by zero.",
        ))

    def _add_dti_x_loan_to_income(self, df: pd.DataFrame, task: str) -> None:
        name = "dti_x_loan_to_income"
        if {"dti", "loan_to_income"}.issubset(df.columns):
            df[name] = df["dti"] * df["loan_to_income"]
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["dti", "loan_to_income"],
            formula="dti * loan_to_income",
            rationale="Interaction between existing debt load and the new "
                      "loan's relative size. Captures compounding leverage.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
        ))

    def _add_credit_hist_months(self, df: pd.DataFrame, task: str) -> None:
        name = "credit_hist_months"
        if "earliest_cr_line" in df.columns:
            ref = self.reference_date if self.reference_date is not None else df.get("issue_d")
            if ref is None:
                df[name] = np.nan
            else:
                earliest = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
                ref_series = (
                    pd.Series(ref, index=df.index)
                    if not isinstance(ref, pd.Series) else ref
                )
                ref_series = pd.to_datetime(ref_series, errors="coerce")
                df[name] = (ref_series - earliest).dt.days / 30.44
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["earliest_cr_line", "issue_d"],
            formula="(reference_date - earliest_cr_line) in months",
            rationale="Longer credit history is a proxy for financial stability.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=False,
            eligible_classification=True,
            notes=("reference_date defaults to per-row issue_d (funding month). "
                   "Excluded from pre-pricing regression: no verified application date. "
                   "Use baseline mo_sin_old_rev_tl_op for that task instead."),
        ))

    def _add_revol_util_bucket(self, df: pd.DataFrame, task: str) -> None:
        name = "revol_util_bucket"
        if "revol_util" in df.columns:
            # Deterministic bins (fixed thresholds, not learned)
            bins = [-0.001, 30, 60, 80, 100, np.inf]
            labels = [0, 1, 2, 3, 4]
            df[name] = pd.cut(df["revol_util"], bins=bins, labels=labels).astype("float")
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["revol_util"],
            formula="0: [0,30], 1: (30,60], 2: (60,80], 3: (80,100], 4: >100",
            rationale="Utilisation risk is known to be non-linear; binning exposes "
                      "threshold effects without a learned transformation.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
            notes="Deterministic bins; no fitting required.",
        ))

    def _add_total_credit_lines(self, df: pd.DataFrame, task: str) -> None:
        name = "total_credit_lines"
        if {"open_acc", "total_acc"}.issubset(df.columns):
            df[name] = df["open_acc"] + df["total_acc"]
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["open_acc", "total_acc"],
            formula="open_acc + total_acc",
            rationale="Depth of the credit file; a coarse proxy for how much "
                      "history the bureau has on the borrower.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
        ))

    def _add_inquiries_per_open_account(self, df: pd.DataFrame, task: str) -> None:
        name = "inquiries_per_open_account"
        if {"inq_last_6mths", "open_acc"}.issubset(df.columns):
            df[name] = df["inq_last_6mths"] / df["open_acc"].replace(0, np.nan)
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name, source_columns=["inq_last_6mths", "open_acc"],
            formula="inq_last_6mths / open_acc; zero denominator -> NaN",
            rationale="Recent credit-seeking intensity relative to the active credit file.",
            missing_handling="Missing inputs and zero denominator -> NaN; fold-wise imputation.",
            eligible_regression=True, eligible_classification=True,
        ))

    def _add_util_x_fico(self, df: pd.DataFrame, task: str) -> None:
        name = "util_x_fico"
        if {"revol_util", "fico_range_low"}.issubset(df.columns):
            df[name] = df["revol_util"] * df["fico_range_low"]
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["revol_util", "fico_range_low"],
            formula="revol_util * fico_range_low",
            rationale="Joint signal: high utilisation is more dangerous at "
                      "low FICO than at high FICO.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
        ))

    def _add_bc_pressure(self, df: pd.DataFrame, task: str) -> None:
        name = "bc_pressure"
        if {"bc_util", "bc_open_to_buy"}.issubset(df.columns):
            denom = df["bc_open_to_buy"].replace(0, np.nan)
            df[name] = df["bc_util"] / denom
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["bc_util", "bc_open_to_buy"],
            formula="bc_util / bc_open_to_buy; zero denominator -> NaN",
            rationale="Bank-card pressure: high utilisation with little "
                      "available headroom is a strong stress indicator.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
            notes="bc_open_to_buy == 0 mapped to NaN before division.",
        ))

    def _add_delinq_rate(self, df: pd.DataFrame, task: str) -> None:
        name = "delinq_rate"
        if {"delinq_2yrs", "total_acc"}.issubset(df.columns):
            denom = df["total_acc"].replace(0, np.nan)
            df[name] = df["delinq_2yrs"] / denom
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["delinq_2yrs", "total_acc"],
            formula="delinq_2yrs / total_acc; zero denominator -> NaN",
            rationale="Normalised delinquency: raw counts are unfair to "
                      "borrowers with deep credit files.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
        ))

    def _add_loan_per_term(self, df: pd.DataFrame, task: str) -> None:
        name = "loan_per_term"
        if {"loan_amnt", "term"}.issubset(df.columns):
            df[name] = df["loan_amnt"] / df["term"]
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["loan_amnt", "term"],
            formula="loan_amnt / term",
            rationale="Monthly principal burden at origination; captures the "
                      "pace at which the borrower must repay.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
        ))

    def _add_emp_length_years(self, df: pd.DataFrame, task: str) -> None:
        name = "emp_length_years"
        if "emp_length" in df.columns:
            s = df["emp_length"].astype(str).str.extract(r"(\d+)").astype(float)
            df[name] = s.iloc[:, 0].mask(df["emp_length"].eq("< 1 year").fillna(False), 0.5)
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["emp_length"],
            formula="<1 year -> 0.5; 1..9 years -> integer; 10+ years -> 10",
            rationale="Employment stability; a well-known underwriting input.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
            notes="0.5 is a fixed midpoint convention for <1 year; 10 is the "
                  "lower bound for 10+ years. Missing strings remain NaN.",
        ))

    def _add_pub_rec_bankruptcies_flag(self, df: pd.DataFrame, task: str) -> None:
        name = "has_bankruptcy"
        if "pub_rec_bankruptcies" in df.columns:
            df[name] = (df["pub_rec_bankruptcies"] > 0).astype("float").where(
                df["pub_rec_bankruptcies"].notna()
            )
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["pub_rec_bankruptcies"],
            formula="(pub_rec_bankruptcies > 0) as 0/1",
            rationale="Binary signal of past bankruptcy; more stable than the "
                      "raw count which is highly skewed.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=True,
            eligible_classification=True,
        ))

    def _add_closed_credit_share(self, df: pd.DataFrame, task: str) -> None:
        name = "closed_credit_share"
        if {"total_acc", "open_acc"}.issubset(df.columns):
            valid = (df["total_acc"] > 0) & (df["open_acc"] >= 0) & (df["open_acc"] <= df["total_acc"])
            df[name] = ((df["total_acc"] - df["open_acc"]) / df["total_acc"].replace(0, np.nan)).where(valid)
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name, source_columns=["total_acc", "open_acc"],
            formula="(total_acc - open_acc) / total_acc; invalid counts or total_acc <= 0 -> NaN",
            rationale="Share of the historical credit file no longer open; captures account turnover.",
            missing_handling="Missing or inconsistent counts -> NaN; fold-wise imputation.",
            eligible_regression=True, eligible_classification=True,
        ))

    def _add_installment_burden(self, df: pd.DataFrame, task: str) -> None:
        name = "installment_burden"
        eligible = task == "classification"
        if eligible and {"installment", "annual_inc"}.issubset(df.columns):
            denom = df["annual_inc"].replace(0, np.nan)
            df[name] = (df["installment"] * 12) / denom
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["installment", "annual_inc"],
            formula="(installment * 12) / annual_inc",
            rationale="Annualised repayment burden as a share of income.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=False,
            eligible_classification=True,
            eligible_clustering=False,
            notes=("Excluded from regression because installment is derived from "
                   "loan_amnt, term, and int_rate and is banned for that task."),
        ))

    def _add_rate_burden(self, df: pd.DataFrame, task: str) -> None:
        name = "rate_burden"
        eligible = task == "classification"
        if eligible and {"int_rate", "loan_amnt"}.issubset(df.columns):
            df[name] = (df["int_rate"] / 100.0) * df["loan_amnt"]
        else:
            df[name] = np.nan
        self.specs.append(FeatureSpec(
            name=name,
            source_columns=["int_rate", "loan_amnt"],
            formula="(int_rate/100) * loan_amnt",
            rationale="Dollar cost of the loan's first-year interest; a "
                      "direct measure of how burdensome the priced loan is.",
            missing_handling="NaN preserved; imputed inside the training-fold pipeline.",
            eligible_regression=False,
            eligible_classification=True,
            eligible_clustering=False,
            notes=("Excluded from regression because int_rate is the target for "
                   "that task; it is allowed for the origination classifier "
                   "because the priced rate is fixed at origination."),
        ))