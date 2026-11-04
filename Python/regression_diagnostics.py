"""Descriptive regression diagnostics; never alter fitted models or evaluation rows."""
from __future__ import annotations

import numpy as np
import pandas as pd


class RegressionDiagnostics:
    """Trace extreme errors and linear feature contributions without refitting."""

    @staticmethod
    def extreme_error_summary(frame, target='int_rate'):
        residual = frame[target] - frame['prediction']
        squared = residual.pow(2).sort_values(ascending=False)
        total = squared.sum()
        summary = dict(min_prediction=float(frame.prediction.min()),
                       max_prediction=float(frame.prediction.max()),
                       median_absolute_error=float(residual.abs().median()))
        for n in [1, 10, 100]:
            summary[f'mse_share_top_{n}'] = float(squared.head(n).sum() / total) if total else 0.
        return summary

    @staticmethod
    def linear_contributions(bundle, input_rows, top_features=5):
        """Return largest signed contributions per row in target units.

        Rows must retain their validation partition indices. Contributions use
        fitted standardized/encoded values; the model intercept is separate.
        """
        model = bundle['model']
        if np.asarray(model.coef_).ndim != 1:
            raise ValueError('Expected a single-output linear regressor.')
        matrix = bundle['preprocessing'].transform(input_rows[bundle['predictors']])
        names = bundle['preprocessing']._pipeline.named_steps['pre'].get_feature_names_out()
        contributions = matrix * model.coef_
        records = []
        for row_position, index in enumerate(input_rows.index):
            for feature in np.argsort(-np.abs(contributions[row_position]))[:top_features]:
                name = str(names[feature])
                original = name.removeprefix('num__')
                records.append(dict(partition_row=int(index), feature=name,
                                    contribution_percentage_points=float(contributions[row_position, feature]),
                                    raw_value=input_rows.loc[index, original] if original in input_rows else np.nan))
        return pd.DataFrame(records)
