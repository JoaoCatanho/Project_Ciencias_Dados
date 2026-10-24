"""
Lending Club project source package.

Modules
-------
data_loader   : loaders + schema reports for accepted/rejected CSVs
preprocessing : cleaning, parsing, and leak-safe transformation helpers
splits        : reusable chronological train/val/test split policy
eda_utils     : reproducible EDA, visualizations, PCA, and UMAP
hypothesis    : statistical tests, effect sizes, and confidence intervals
feature_validation : engineered-feature diagnostics
"""

from .data_loader import LendingClubLoader
from .preprocessing import LendingClubPreprocessor, REGRESSION_BANNED, CLASSIFICATION_BANNED
from .splits import ChronologicalSplitter, ChronologicalSplits
from .eda_utils import LendingClubEDA
from .hypothesis import HypothesisTester
from .feature_validation import FeatureValidator

__all__ = [
    "LendingClubLoader",
    "LendingClubPreprocessor",
    "REGRESSION_BANNED",
    "CLASSIFICATION_BANNED",
    "ChronologicalSplitter",
    "ChronologicalSplits",
    "LendingClubEDA",
    "HypothesisTester",
    "FeatureValidator",
]
