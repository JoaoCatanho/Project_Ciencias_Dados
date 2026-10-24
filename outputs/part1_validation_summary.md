# Part 1 validation results

All six Python notebooks were executed in order on the full input datasets.
All 92 code cells completed without errors. All 17 unit and artifact integration tests passed.

## Corrected development populations

| Population | Rows | Latest development date |
|---|---:|---|
| classification | 1,129,813 | 2016-12-01 |
| funded | 1,321,847 | 2016-12-01 |
| rejected | 11,032,222 | 2016-12-31 |

## Features and transformations

- Eligible engineered features: 12 for regression and 15 for classification.
- Regression predictor allowlists exclude funding-date history, verification status, rate-determining variables and post-origination data.
- Inquiries per open account and closed credit share replace the two simple rescalings.
- Employment below one year maps to the documented 0.5-year convention; exactly one year remains 1, and 10+ years is capped at its lower bound of 10.
- All 697 missing classification-cohort bankruptcy counts remain missing in the binary flag.
- DTI sentinels become missing. Nonnegative heavy-tailed plotting variables use log1p, including zeros.
- PCA and UMAP fit training samples only; validation samples use transform without refitting.
- Training-only PCA cumulative variance at ten components: 81.67%.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg .venv/bin/python -B -m unittest discover -s tests -v
```

The integration checks validate saved notebook execution, development-period boundaries, funded-cohort inclusion, predictor eligibility, encoding, missingness, finite projections and statistical output ranges.

## Remaining limitations

- The snapshot does not establish historical label availability or a verified common observation cutoff. This is a chronological holdout with retrospective labels, not a historical backtest.
- Earlier exploratory runs exposed test-period distributions. Corrected outputs avoid further test-driven development, but cannot undo that exposure.
- Cross-file funded proportions and amount comparisons describe different recorded populations and are not causal approval models.
- Complete-case dimensionality-reduction samples may underrepresent earlier vintages.
- Feature screening recommendations require subsequent fold-wise predictive validation; high correlation alone does not establish uselessness.
