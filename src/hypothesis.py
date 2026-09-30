"""
src/hypothesis.py
-----------------
Hypothesis-testing helpers with effect sizes and confidence intervals.

Rules enforced here:
* Never claim causation from an observational association.
* Always report effect size alongside p-value.
* Non-parametric tests where normality is not plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    h0: str
    h1: str
    test: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: Optional[int] = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta (non-parametric effect size in [-1, 1])."""
    a, b = np.asarray(a), np.asarray(b)
    # rank-based, O(n log n) using searchsorted
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    # count pairs where a > b, a < b
    greater = np.searchsorted(b_sorted, a, side="left").sum()
    less = (len(b) - np.searchsorted(b_sorted, a, side="right")).sum()
    n = len(a) * len(b)
    return float((greater - less) / n) if n else 0.0


def cramers_v(confusion: np.ndarray) -> float:
    """Cramér's V for a contingency table."""
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum()
    r, k = confusion.shape
    if n == 0 or min(r - 1, k - 1) == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation from a Mann-Whitney U statistic."""
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(1 - 2 * u / (n1 * n2))


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------
def bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray,
    n_boot: int = 2000, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (rng.choice(a, size=len(a), replace=True).mean()
                    - rng.choice(b, size=len(b), replace=True).mean())
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def mannwhitney_test(
    a: np.ndarray, b: np.ndarray,
    name: str, h0: str, h1: str, note: str = "",
    bootstrap: bool = True, seed: int = 42,
) -> TestResult:
    a, b = np.asarray(a), np.asarray(b)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    eff = cliffs_delta(a, b)
    ci = bootstrap_diff_ci(a, b, seed=seed) if bootstrap else (None, None)
    return TestResult(
        name=name, h0=h0, h1=h1, test="Mann-Whitney U",
        statistic=float(stat), p_value=float(p),
        effect_size=eff, effect_size_name="Cliff's delta",
        ci_low=ci[0], ci_high=ci[1], n=len(a) + len(b), note=note,
    )


def chi_square_test(
    table: pd.DataFrame, name: str, h0: str, h1: str, note: str = "",
) -> TestResult:
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    v = cramers_v(table.values)
    return TestResult(
        name=name, h0=h0, h1=h1, test="Chi-square",
        statistic=float(chi2), p_value=float(p),
        effect_size=v, effect_size_name="Cramér's V",
        n=int(table.values.sum()), note=note,
    )


def welch_ttest(
    a: np.ndarray, b: np.ndarray,
    name: str, h0: str, h1: str, note: str = "",
    bootstrap: bool = True, seed: int = 42,
) -> TestResult:
    """Welch's t-test; use only when normality is plausible — otherwise prefer Mann-Whitney."""
    a, b = np.asarray(a), np.asarray(b)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    eff = cohens_d(a, b)
    ci = bootstrap_diff_ci(a, b, seed=seed) if bootstrap else (None, None)
    return TestResult(
        name=name, h0=h0, h1=h1, test="Welch t-test",
        statistic=float(stat), p_value=float(p),
        effect_size=eff, effect_size_name="Cohen's d",
        ci_low=ci[0], ci_high=ci[1], n=len(a) + len(b), note=note,
    )


# ---------------------------------------------------------------------------
# Normality check
# ---------------------------------------------------------------------------
def normality_report(x: np.ndarray, sample_n: int = 5000, seed: int = 42) -> dict:
    """Shapiro-Wilk on a reproducible subsample (Shapiro is unstable for huge n)."""
    x = np.asarray(x); x = x[~np.isnan(x)]
    if len(x) > sample_n:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=sample_n, replace=False)
    stat, p = stats.shapiro(x)
    return {
        "shapiro_stat": float(stat),
        "shapiro_p": float(p),
        "skew": float(stats.skew(x)),
        "n_used": int(len(x)),
    }


def results_to_dataframe(results: list[TestResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])