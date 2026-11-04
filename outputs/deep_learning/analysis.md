# Notebook 10 — observed deep-learning results

- Regression: validation RMSE **3.592 ± 0.022 percentage points** across training seeds; MAE **2.893**, R² **0.402**. Histogram boosting RMSE: **3.588**.
- Classification: validation macro F1 **0.371 ± 0.004**, accuracy **58.70%**, macro OVR AUC **0.653**. Histogram boosting macro F1: **0.422**.
- At risk precision **3.18%**, recall **47.47%**, AP **0.0305**. Default recall **27.80%**. Read these alongside the confusion matrix and term-specific breakdown rather than accuracy alone.
- Every final classification model retained all **451,671 training loans**, including **611 At risk examples**. Regression used up to **50,000 seeded fitting rows**. All original validation rows were scored and their targets/dates matched previous notebooks.
- Epochs came from the final calendar month inside each fitting partition. Each scored model was freshly refitted on its full designated fitting partition after epoch selection; external validation did not determine stopping or seed choice.
- Numerical conditioning was an explicit training-fitted revision. Architectural and preprocessing/training-budget effects are not isolated by comparison with previous model families.
- Runtime: **2.43 minutes**. Model save/reload checks passed. Original classes and external splits were preserved; final test data remain unread.

## Assessment of the observed results

**The MLP is a competitive regression reference, not a demonstrated improvement over boosting.** Its mean RMSE is 3.592 percentage points versus 3.588 for boosting; the difference is much smaller than the MLP's training-seed spread (0.022). Selecting the best seed would exaggerate the comparison, so retain all three. Regression seed variation includes both the 50,000-row fitting sample and neural randomness. Numeric clipping/log transforms differ from previous model families, so this is not an isolated architecture comparison. Extreme rates are still compressed toward the middle, and monthly errors must be considered before generalizing to later vintages.

**The classification network does not replace the ensemble benchmark.** Mean macro F1 is 0.371, below boosting's 0.422; mean Default recall is 27.8%, versus 55.0%. Balanced loss selected by training CV increases At risk detection, but its mean precision is only 3.18% at recall 47.5%. Average precision is 0.0305, consistent with the weak ranking signal found in prior diagnostics. More At risk alerts alone are not evidence of an effective risk detector.

**The term-support gap persists.** For primary seed 42, At risk recall is 0.00% for 36-month loans and 89.7% for 60-month loans. The original training cohort has no 36-month At risk examples. A nonlinear model has not supplied the missing training support or established a common outcome horizon. Keep these findings alongside the overall confusion matrix.

**Early stopping is conservative and unstable for the rare-class task.** Final classification stopping chooses 1–3 epochs across seeds. The original training tail contains only 34 At risk cases; earlier CV inner fitting has as few as 13. Training loss can decrease while stopping loss rises. This indicates poor transfer to that stopping cohort; temporal prior changes, sparse minority support, weighting and model fitting can all contribute. It is not proof that every neural architecture would fail. The architecture, learning rates and weighting search are deliberately limited. A small selected epoch count is reported rather than hidden or replaced by validation-selected training duration.

**Protocol and artifacts:** every scored network is freshly refitted after selecting its epoch count within training data. All final classification models use all 451,671 training rows and all 611 At risk examples. Complete validation cohorts were retained, original classes/splits were preserved, and all six saved bundles reproduce predictions. Model/input/output cards and fitted bounds are saved for reproducible inference. Earlier models and the canonical predictor plans remain intact. The final test set is still reserved for the final comparison phase.

**Handoff:** retain boosting as the current classification benchmark and both boosting/MLP as regression references. Proceed with the required clustering notebook, then freeze choices before final test evaluation. The successful implementation satisfies the deep-learning comparison requirement; it does not establish calibration, fairness or deployment readiness.
