"""
src/eda_utils.py
----------------
Utilities for EDA in the Lending Club project.

Design rules enforced here (from the project specification):
* EDA is done on train/validation only — never on the held-out test set.
* Reproducible sampling (no `head()` as a representative sample).
* Linear dimensionality reduction: PCA.
* Non-linear dimensionality reduction: UMAP.
* All figures are saved to outputs/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EDA analysis
# ---------------------------------------------------------------------------
class LendingClubEDA:
    """Reusable EDA operations with a shared reproducible sampling seed.

    Inputs remain explicit so one instance can analyze different cohorts.
    Pass train/validation data only; this class does not select the cohort.
    Method-level seeds override the instance seed without changing it.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def reproducible_sample(self, df: pd.DataFrame, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Return a reproducible random sample of n rows (never use df.head())."""
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        if n >= len(df):
            return df.copy()
        idx = rng.choice(len(df), size=n, replace=False)
        return df.iloc[np.sort(idx)].copy()

    # ---------------------------------------------------------------------------
    # Descriptive statistics
    # ---------------------------------------------------------------------------
    def describe_numeric(self, df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        """Extended descriptive stats: count, missing%, mean, std, quartiles, skew, kurtosis."""
        present = [c for c in cols if c in df.columns]
        desc = df[present].describe().T
        desc["missing_pct"] = df[present].isna().mean() * 100
        desc["skew"] = df[present].skew()
        desc["kurtosis"] = df[present].kurtosis()
        return desc.round(4)

    # ---------------------------------------------------------------------------
    # Plotting helpers
    # ---------------------------------------------------------------------------
    def plot_histograms(
        self,
        df: pd.DataFrame,
        cols: Sequence[str],
        out_path: Path,
        bins: int = 50,
        log_scale_cols: Optional[Iterable[str]] = None,
    ) -> None:
        """Grid of histograms for numeric columns."""
        log_scale_cols = set(log_scale_cols or [])
        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, cols):
            data = df[col].dropna()
            if col in log_scale_cols and (data > 0).all():
                data = np.log1p(data)
                ax.set_title(f"log1p({col})")
            else:
                ax.set_title(col)
            ax.hist(data, bins=bins, color="steelblue", edgecolor="white")
            ax.set_ylabel("count")
        for ax in axes[len(cols):]:
            ax.axis("off")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved histogram grid -> %s", out_path)

    def plot_scatter(
        self,
        df: pd.DataFrame, x: str, y: str, out_path: Path, hue: Optional[str] = None,
        sample_n: int = 20_000, seed: Optional[int] = None,
    ) -> None:
        """Scatter plot (subsampled reproducibly) with optional hue."""
        seed = self.seed if seed is None else seed
        sub = self.reproducible_sample(df, n=sample_n, seed=seed)
        fig, ax = plt.subplots(figsize=(7, 5))
        if hue and hue in sub.columns:
            sns.scatterplot(data=sub, x=x, y=y, hue=hue, s=8, alpha=0.5, ax=ax)
        else:
            ax.scatter(sub[x], sub[y], s=6, alpha=0.4, color="steelblue")
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved scatter -> %s", out_path)

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame, cols: Sequence[str], out_path: Path, method: str = "spearman",
    ) -> pd.DataFrame:
        """Correlation heatmap for numeric columns; returns the correlation matrix."""
        present = [c for c in cols if c in df.columns]
        corr = df[present].corr(method=method, numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(present)),
                                        max(6, 0.35 * len(present))))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
        ax.set_title(f"{method.title()} correlation")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved heatmap -> %s", out_path)
        return corr

    # ---------------------------------------------------------------------------
    # Linear DR: PCA
    # ---------------------------------------------------------------------------
    def run_pca(
        self,
        df: pd.DataFrame,
        cols: Sequence[str],
        out_dir: Path,
        n_components: int = 10,
        sample_n: Optional[int] = 50_000,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Fit PCA on a reproducible sample of the training data.
        Saves a scree plot and a 2D projection plot.
        Returns a dict with explained variance ratios and the fitted model.
        """
        seed = self.seed if seed is None else seed
        present = [c for c in cols if c in df.columns]
        sub = self.reproducible_sample(df[present].dropna(), n=sample_n or len(df), seed=seed)
        X = StandardScaler().fit_transform(sub.values)

        pca = PCA(n_components=n_components, random_state=seed)
        Z = pca.fit_transform(X)

        # Scree
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
               pca.explained_variance_ratio_, color="steelblue")
        ax.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                np.cumsum(pca.explained_variance_ratio_), "o-", color="darkred")
        ax.set_xlabel("component"); ax.set_ylabel("explained variance ratio")
        ax.set_title("PCA scree plot")
        fig.tight_layout()
        fig.savefig(out_dir / "pca_scree.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2D projection
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.35, color="steelblue")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("PCA — first two components")
        fig.tight_layout()
        fig.savefig(out_dir / "pca_2d.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            index=present,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        )
        loadings.to_csv(out_dir / "pca_loadings.csv")

        return {
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
            "pca": pca,
        }

    # ---------------------------------------------------------------------------
    # Non-linear DR: UMAP
    # ---------------------------------------------------------------------------
    def run_umap(
        self,
        df: pd.DataFrame,
        cols: Sequence[str],
        out_dir: Path,
        hue: Optional[str] = None,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        sample_n: int = 30_000,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fit UMAP on a reproducible sample. Saves a 2D embedding plot and
        returns the embedding DataFrame.
        """
        seed = self.seed if seed is None else seed
        try:
            import umap  # umap-learn
        except ImportError as e:
            raise ImportError(
                "umap-learn is required. Install with: pip install umap-learn"
            ) from e

        present = [c for c in cols if c in df.columns]
        keep = present + ([hue] if hue and hue in df.columns else [])
        sub = self.reproducible_sample(df[keep].dropna(), n=sample_n, seed=seed)
        X = StandardScaler().fit_transform(sub[present].values)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=seed,
        )
        Z = reducer.fit_transform(X)

        emb = pd.DataFrame(Z, columns=["umap1", "umap2"], index=sub.index)
        if hue and hue in sub.columns:
            emb[hue] = sub[hue].values

        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        if hue and hue in emb.columns:
            sns.scatterplot(data=emb, x="umap1", y="umap2", hue=hue,
                            s=8, alpha=0.5, ax=ax)
        else:
            ax.scatter(emb["umap1"], emb["umap2"], s=6, alpha=0.4, color="steelblue")
        ax.set_title("UMAP — 2D embedding")
        fig.tight_layout()
        fig.savefig(out_dir / "umap_2d.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        return emb

    def iqr_outliers(self, s: pd.Series) -> int:
        """Count observations beyond 1.5 IQR from the quartiles."""
        s = s.dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return int(((s < lo) | (s > hi)).sum())
