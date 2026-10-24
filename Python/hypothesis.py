"""
src/hypothesis.py
-----------------
Hypothesis-testing helpers with effect sizes, bootstrap confidence intervals,
and explicit sample-size guards.

Design rules enforced here (from the project specification):
* Never claim causation from an observational association.
* Always report effect size alongside p-value.
* Non-parametric tests where normality is not plausible.
* Bootstrap CI caps each sample at `max_n_per_sample` (default 200_000) so
  that very large inputs (e.g. 27M rejected rows) do not freeze the notebook.
* Every TestResult carries a `summary()` line so notebooks can print a
  human-readable verdict without extra formatting.
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
    n_a: Optional[int] = None
    n_b: Optional[int] = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self, alpha: float = 0.05) -> str:
        """Human-readable one-liner."""
        decision = "REJECT H0" if self.p_value < alpha else "fail to reject H0"
        ci = (
            f", 95% CI of mean diff = [{self.ci_low:.4f}, {self.ci_high:.4f}]"
            if self.ci_low is not None and self.ci_high is not None and
               not (np.isnan(self.ci_low) or np.isnan(self.ci_high))
            else ""
        )
        return (
            f"{self.name} | {self.test} | stat={self.statistic:.4g}, "
            f"p={self.p_value:.3e} -> {decision} | "
            f"{self.effect_size_name}={self.effect_size:.4f} "
            f"| n={self.n}{ci}"
        )


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------
class HypothesisTester:
    """Statistical tests sharing reproducibility and bootstrap configuration.

    Method arguments can override the instance defaults for an individual test.
    Results retain the existing TestResult reporting interface.
    """

    def __init__(
        self, seed: int = 42, max_n_per_sample: int = 200_000,
        n_boot: int = 2000, alpha: float = 0.05,
    ) -> None:
        self.seed = seed
        self.max_n_per_sample = max_n_per_sample
        self.n_boot = n_boot
        self.alpha = alpha

    def cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cohen's d for two independent samples (pooled SD)."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return 0.0
        va, vb = a.var(ddof=1), b.var(ddof=1)
        pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        if pooled == 0:
            return 0.0
        return float((a.mean() - b.mean()) / pooled)

    def cliffs_delta(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cliff's delta (non-parametric effect size in [-1, 1])."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if len(a) == 0 or len(b) == 0:
            return 0.0
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        greater = np.searchsorted(b_sorted, a, side="left").sum()
        less = (len(b) - np.searchsorted(b_sorted, a, side="right")).sum()
        n = len(a) * len(b)
        return float((greater - less) / n) if n else 0.0

    def cramers_v(self, confusion: np.ndarray) -> float:
        """Cramér's V for a contingency table."""
        confusion = np.asarray(confusion)
        if confusion.size == 0:
            return 0.0
        chi2 = stats.chi2_contingency(confusion, correction=False)[0]
        n = confusion.sum()
        r, k = confusion.shape
        if n == 0 or min(r - 1, k - 1) == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))

    # ---------------------------------------------------------------------------
    # Bootstrap CI (capped)
    # ---------------------------------------------------------------------------
    def bootstrap_diff_ci(
        self,
        a: np.ndarray,
        b: np.ndarray,
        n_boot: Optional[int] = None,
        alpha: Optional[float] = None,
        seed: Optional[int] = None,
        max_n_per_sample: Optional[int] = None,
    ) -> tuple[float, float]:
        """
        Bootstrap CI for mean(a) - mean(b).

        Caps each sample at `max_n_per_sample` rows before bootstrapping so that
        very large inputs (e.g. 27M rejected rows) do not freeze the notebook.
        """
        seed = self.seed if seed is None else seed
        max_n_per_sample = self.max_n_per_sample if max_n_per_sample is None else max_n_per_sample
        n_boot = self.n_boot if n_boot is None else n_boot
        alpha = self.alpha if alpha is None else alpha
        rng = np.random.default_rng(seed)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

        if len(a) > max_n_per_sample:
            a = rng.choice(a, size=max_n_per_sample, replace=False)
        if len(b) > max_n_per_sample:
            b = rng.choice(b, size=max_n_per_sample, replace=False)

        if len(a) == 0 or len(b) == 0:
            return (float("nan"), float("nan"))

        diffs = np.empty(n_boot)
        for i in range(n_boot):
            diffs[i] = (
                rng.choice(a, size=len(a), replace=True).mean()
                - rng.choice(b, size=len(b), replace=True).mean()
            )
        return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))

    # ---------------------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------------------
    def mannwhitney_test(
        self,
        a: np.ndarray,
        b: np.ndarray,
        name: str,
        h0: str,
        h1: str,
        note: str = "",
        bootstrap: bool = True,
        seed: Optional[int] = None,
        max_n_per_sample: Optional[int] = None,
    ) -> Optional[TestResult]:
        """Mann-Whitney U with Cliff's delta and a bootstrap CI of the mean diff."""
        seed = self.seed if seed is None else seed
        max_n_per_sample = self.max_n_per_sample if max_n_per_sample is None else max_n_per_sample
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

        if len(a) < 2 or len(b) < 2:
            print(f"[SKIP] {name}: not enough data (n_a={len(a)}, n_b={len(b)})")
            return None

        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        eff = self.cliffs_delta(a, b)
        ci = (
            self.bootstrap_diff_ci(a, b, seed=seed, max_n_per_sample=max_n_per_sample)
            if bootstrap else (None, None)
        )

        return TestResult(
            name=name, h0=h0, h1=h1, test="Mann-Whitney U",
            statistic=float(stat), p_value=float(p),
            effect_size=eff, effect_size_name="Cliff's delta",
            ci_low=ci[0], ci_high=ci[1],
            n=len(a) + len(b), n_a=len(a), n_b=len(b), note=note,
        )

    def chi_square_test(
        self,
        table: pd.DataFrame,
        name: str,
        h0: str,
        h1: str,
        note: str = "",
        min_expected: float = 5.0,
    ) -> Optional[TestResult]:
        """Chi-square with Cramér's V and an expected-count sanity check."""
        if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
            print(f"[SKIP] {name}: table has too few rows/columns.")
            return None

        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        v = self.cramers_v(table.values)

        if expected.min() < min_expected:
            warn = (
                f"WARNING: min expected count = {expected.min():.2f} < {min_expected}. "
                f"Chi-square approximation may be unreliable."
            )
            print(f"[WARN] {name}: {warn}")
            note = (note + " " + warn).strip()

        if v < 0.10:
            print(
                f"[INFO] {name}: Cramér's V = {v:.4f} is negligible "
                f"(p is small because n = {int(table.values.sum()):,})."
            )

        return TestResult(
            name=name, h0=h0, h1=h1, test="Chi-square",
            statistic=float(chi2), p_value=float(p),
            effect_size=v, effect_size_name="Cramér's V",
            n=int(table.values.sum()), note=note,
        )

    def welch_ttest(
        self,
        a: np.ndarray,
        b: np.ndarray,
        name: str,
        h0: str,
        h1: str,
        note: str = "",
        bootstrap: bool = True,
        seed: Optional[int] = None,
        max_n_per_sample: Optional[int] = None,
    ) -> Optional[TestResult]:
        """Welch's t-test with Cohen's d and a bootstrap CI of the mean diff."""
        seed = self.seed if seed is None else seed
        max_n_per_sample = self.max_n_per_sample if max_n_per_sample is None else max_n_per_sample
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

        if len(a) < 2 or len(b) < 2:
            print(f"[SKIP] {name}: not enough data (n_a={len(a)}, n_b={len(b)})")
            return None

        stat, p = stats.ttest_ind(a, b, equal_var=False)
        eff = self.cohens_d(a, b)
        ci = (
            self.bootstrap_diff_ci(a, b, seed=seed, max_n_per_sample=max_n_per_sample)
            if bootstrap else (None, None)
        )

        return TestResult(
            name=name, h0=h0, h1=h1, test="Welch t-test",
            statistic=float(stat), p_value=float(p),
            effect_size=eff, effect_size_name="Cohen's d",
            ci_low=ci[0], ci_high=ci[1],
            n=len(a) + len(b), n_a=len(a), n_b=len(b), note=note,
        )

    # ---------------------------------------------------------------------------
    # Normality check
    # ---------------------------------------------------------------------------
    def normality_report(self, x: np.ndarray, sample_n: int = 5000, seed: Optional[int] = None) -> dict:
        """Shapiro-Wilk on a reproducible subsample (Shapiro is unstable for huge n)."""
        seed = self.seed if seed is None else seed
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return {"shapiro_stat": float("nan"), "shapiro_p": float("nan"),
                    "skew": float("nan"), "n_used": 0}
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

    # ---------------------------------------------------------------------------
    # Collecting results
    # ---------------------------------------------------------------------------
    def results_to_dataframe(self, results: list[Optional[TestResult]]) -> pd.DataFrame:
        """
        Convert a list that may contain None to a DataFrame.
        Prints a warning for every None entry so silent failures are impossible.
        """
        clean = []
        for i, r in enumerate(results):
            if r is None:
                print(f"[MISSING] result #{i} is None — check the corresponding cell.")
            else:
                clean.append(r.to_dict())
        return pd.DataFrame(clean)