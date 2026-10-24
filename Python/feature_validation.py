"""
src/feature_validation.py
-------------------------
Descriptive diagnostics to check whether engineered features are meaningful.

Purpose
-------
Before committing to engineered features in Phase 4, we want evidence that:

1. They are not near-perfect duplicates of an original feature
   (low redundancy).
2. They carry non-trivial association with the target
   (mutual information / correlation with target).
3. They are not collinear with each other in a way that will blow up
   linear models.

We compute three quantities per feature:

* Pearson and Spearman correlation with each original feature (redundancy).
* Mutual information with the target (classification) or Pearson correlation
  with the target (regression).
* Maximum absolute correlation with any other feature (collinearity).

Nothing here trains a model. Everything is fit on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# ---------------------------------------------------------------------------
# Diagnostics container
# ---------------------------------------------------------------------------
@dataclass
class FeatureDiagnostics:
    name: str
    kind: str                    # 'original' or 'engineered'
    target_signal: float         # MI (classification) or |Pearson r| (regression)
    max_redundancy: float        # max |corr| with any feature of the other kind
    max_redundancy_with: str     # which feature is the max-redundancy partner
    max_self_collinearity: float # max |corr| with any other feature
    max_self_collinearity_with: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class FeatureValidator:
    """Training-set diagnostics with shared sampling and redundancy settings.

    Classification and regression use the same diagnostic definitions as before.
    """

    def __init__(self, seed: int = 42, threshold: float = 0.95) -> None:
        self.seed = seed
        self.threshold = threshold

    def target_signal_classification(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_cols: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ) -> pd.Series:
        """
        Mutual information between each column of X and the discrete target y.
        Categorical columns are label-encoded for MI only (no leakage: fit on X).
        """
        seed = self.seed if seed is None else seed
        X_enc = X.copy()
        cat_cols = list(categorical_cols or [])
        for c in cat_cols:
            if c in X_enc.columns:
                X_enc[c] = X_enc[c].astype("category").cat.codes
        X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
        # MI does not accept NaN: median-impute on the fly for the diagnostic only.
        X_enc = X_enc.fillna(X_enc.median(numeric_only=True)).fillna(0)
        mi = mutual_info_classif(
            X_enc.values, y.values, discrete_features=False, random_state=seed,
        )
        return pd.Series(mi, index=X.columns, name="target_signal")

    def target_signal_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_cols: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ) -> pd.Series:
        """
        Absolute Pearson correlation between each column of X and a continuous y.
        Categorical columns are label-encoded for correlation only.
        """
        seed = self.seed if seed is None else seed
        X_enc = X.copy()
        cat_cols = list(categorical_cols or [])
        for c in cat_cols:
            if c in X_enc.columns:
                X_enc[c] = X_enc[c].astype("category").cat.codes
        X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
        corrs = X_enc.corrwith(y, method="pearson", numeric_only=True).abs()
        return corrs.rename("target_signal")

    def redundancy_table(
        self,
        df: pd.DataFrame,
        engineered: Iterable[str],
        originals: Iterable[str],
        method: str = "spearman",
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        For each engineered feature, find the strongest correlation with any
        original feature. Flag if above threshold (near-redundant).
        """
        threshold = self.threshold if threshold is None else threshold
        engineered = [c for c in engineered if c in df.columns]
        originals = [c for c in originals if c in df.columns]
        corr = df[engineered + originals].corr(method=method, numeric_only=True)
        block = corr.loc[engineered, originals]
        rows = []
        for feat in engineered:
            r = block.loc[feat].abs()
            partner = r.idxmax()
            rows.append({
                "engineered": feat,
                "max_abs_corr_with_original": float(r.max()),
                "partner_original": partner,
                "near_redundant": bool(r.max() >= threshold),
            })
        return pd.DataFrame(rows).sort_values(
            "max_abs_corr_with_original", ascending=False
        ).reset_index(drop=True)

    def self_collinearity(
        self,
        df: pd.DataFrame,
        features: Iterable[str],
        method: str = "spearman",
    ) -> pd.DataFrame:
        """
        Max |corr| between each feature and any other feature in the set
        (excluding itself). Useful to spot multicollinearity.
        """
        features = [c for c in features if c in df.columns]
        corr = df[features].corr(method=method, numeric_only=True).abs()
        arr = corr.to_numpy(copy=True)
        np.fill_diagonal(arr, 0.0)
        corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
        rows = []
        for feat in features:
            partner = corr.loc[feat].idxmax()
            rows.append({
                "feature": feat,
                "max_self_collinearity": float(corr.loc[feat].max()),
                "partner": partner,
            })
        return pd.DataFrame(rows).sort_values(
            "max_self_collinearity", ascending=False
        ).reset_index(drop=True)

    def build_diagnostics(
        self,
        df: pd.DataFrame,
        engineered: Iterable[str],
        originals: Iterable[str],
        y: pd.Series,
        task: str,
        categorical_cols: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Combine target signal, redundancy, and self-collinearity for all features.
        """
        seed = self.seed if seed is None else seed
        engineered = [c for c in engineered if c in df.columns]
        originals = [c for c in originals if c in df.columns]
        all_feats = engineered + originals

        # 1. Target signal
        if task == "classification":
            ts = self.target_signal_classification(df[all_feats], y, categorical_cols, seed)
        elif task == "regression":
            ts = self.target_signal_regression(df[all_feats], y, categorical_cols, seed)
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        # 2. Self collinearity (within the full set)
        self_col = self.self_collinearity(df, all_feats)

        # 3. Redundancy engineered -> originals
        red = self.redundancy_table(df, engineered, originals)

        rows = []
        for feat in all_feats:
            kind = "engineered" if feat in engineered else "original"
            if kind == "engineered":
                rr = red[red["engineered"] == feat]
                max_red = float(rr["max_abs_corr_with_original"].iloc[0]) if len(rr) else np.nan
                max_red_partner = rr["partner_original"].iloc[0] if len(rr) else ""
            else:
                # For originals, compute their own redundancy against engineered features
                if engineered:
                    corrs = df[[feat] + engineered].corr(method="spearman", numeric_only=True).loc[feat, engineered].abs()
                    max_red = float(corrs.max())
                    max_red_partner = corrs.idxmax()
                else:
                    max_red = np.nan
                    max_red_partner = ""
            self_row = self_col[self_col["feature"] == feat]
            rows.append(asdict(FeatureDiagnostics(
                name=feat,
                kind=kind,
                target_signal=float(ts.get(feat, np.nan)),
                max_redundancy=max_red,
                max_redundancy_with=max_red_partner,
                max_self_collinearity=float(self_row['max_self_collinearity'].iloc[0]),
                max_self_collinearity_with=self_row['partner'].iloc[0],
            )))

        return pd.DataFrame(rows).sort_values(
            ["kind", "target_signal"], ascending=[True, False]
        ).reset_index(drop=True)

    def verdict(self, row: pd.Series) -> str:
        """Describe a classification feature using the diagnostic thresholds."""
        if row["max_redundancy"] >= self.threshold:
            return "near-redundant — drop or keep only one"
        if row["target_signal"] < 0.005:
            return "weak signal — keep only if it helps a non-linear model"
        return "keep — non-redundant with reasonable signal"

    def verdict_reg(self, row: pd.Series) -> str:
        """Describe a regression feature using the original signal threshold."""
        if row["max_redundancy"] >= self.threshold:
            return "near-redundant — drop or keep only one"
        if row["target_signal"] < 0.02:
            return "weak signal — keep only if it helps a non-linear model"
        return "keep — non-redundant with reasonable signal"
