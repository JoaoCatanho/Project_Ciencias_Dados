Development setup completed successfully.

- Cohort counts match Part 1, and all 10 task/fold combinations preserve month boundaries.
- Mean-predictor validation RMSE: 4.752 percentage points; MAE: 3.838.
- Median-predictor validation MAE: 3.791; RMSE: 4.720.
- Majority-class validation accuracy: 77.36%; macro F1: 0.291. Minority-class recall is zero, so accuracy alone is inadequate.
- Prior baseline macro OVR AUC: 0.500; micro OVR AUC: 0.880. The latter reflects class prevalence rather than individual risk ranking.
- Classification CV uses March–December 2014 evaluation blocks because earlier At risk labels are absent or sparse; this is a substantial label-availability and follow-up limitation.
- Test outcomes remain unread. These are reference scores, not evidence that a useful predictive model has been built.

## 6. What the observed results imply

The training mean interest rate is **13.828%**, versus **12.824%** in validation: a decline of about **1.005 percentage points**. The mean baseline consequently has negative validation R² (−0.0468); the median baseline is slightly better on this later cohort (RMSE 4.720 versus 4.752). Although the mean minimizes squared error on its fitting data, that guarantee does not extend to a shifted future distribution.

Classification also changes over time. Good loans fall from **82.93% to 77.36%**, Default increases from **16.94% to 21.24%**, and At risk rises from **0.135% to 1.400%**. The majority baseline achieves 77.36% accuracy while missing every At risk and Default loan. Its macro F1 is only **0.291**. The prior baseline's micro AUC of **0.880** despite constant borrower scores illustrates why macro AUC, per-class recall, and confusion matrices are essential.

Only **611** training observations belong to At risk. Before December 2013 there are none. The adjusted classification CV starts with **33** At risk training examples; its estimates will still be noisy and constrained to 2014 validation vintages. Later models should report per-fold class support and minority recall, compare training-only class weighting, and avoid claiming that these folds establish availability of historical outcomes. The observed pattern is consistent with snapshot/follow-up effects but does not prove their cause.

All development counts match Part 1. The preprocessing smoke checks produced **101 regression** and **108 classification** encoded columns on the bounded samples, with finite values and matching fit/transform widths. Full-fold category availability can change these widths; downstream estimators must use their own fitted pipelines.
