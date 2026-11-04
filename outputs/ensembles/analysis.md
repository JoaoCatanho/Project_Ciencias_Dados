# Notebook 09 — observed ensemble development results

- **Regression / hist_gradient_boosting**, {"learning_rate": 0.1, "max_leaf_nodes": 15}: CV RMSE **2.527 ± 0.678**; validation MAE **2.889**, RMSE **3.588 percentage points**, R² **0.403**. RMSE change versus the single tree: **-3.9%**; reduction versus mean baseline: **24.5%**.
- **Regression / random_forest**, {"max_depth": 20}: CV RMSE **2.656 ± 0.597**; validation MAE **2.951**, RMSE **3.660 percentage points**, R² **0.379**. RMSE change versus the single tree: **-2.0%**; reduction versus mean baseline: **23.0%**.
- **Classification / hist_gradient_boosting**, {"class_weight": "balanced", "learning_rate": 0.1, "max_leaf_nodes": 31}: CV macro F1 **0.397 ± 0.009**; validation macro F1 **0.422**, accuracy **66.65%**, macro OVR AUC **0.696**. At risk precision/recall: **3.75% / 9.93%**; Default precision/recall: **35.74% / 55.02%**.
- **Classification / random_forest**, {"class_weight": "balanced", "max_depth": 20}: CV macro F1 **0.383 ± 0.003**; validation macro F1 **0.409**, accuracy **63.87%**, macro OVR AUC **0.676**. At risk precision/recall: **3.56% / 23.55%**; Default precision/recall: **34.64% / 46.22%**.

- Runtime: **26.47 minutes**. Every selected fold and external fit completed its declared tree/iteration budget; saved-model checks passed.
- All training rows are retained; each random-forest tree receives a 50%-size bootstrap draw. Validation targets/order match the prior notebooks. Final test data remain unread.
- The joint-applicant feature definition remains unchanged. Application-type and extreme-ratio diagnostics describe remaining weaknesses rather than repairing the evaluation retrospectively.
- Evaluate class-specific precision and recall before recommending a model; improved aggregate scores do not establish usable risk decisions or fairness.
- Next: the planned deep-learning notebook, followed by clustering and final comparisons. Keep prediction-time eligibility and final-test isolation throughout.

## Assessment and recommendations

**Regression improves, but the gain over the single tree is modest.** The lowest observed ensemble validation RMSE is 3.588 percentage points (hist_gradient_boosting), a 3.9% reduction versus the previous single tree. Its R² is 0.403 and MAE is 2.889 percentage points. These errors remain material for rate prediction. CV error is lower than later-vintage validation error, and the scatterplots show predictions compressed toward the middle of the rate range. These development comparisons do not establish statistically significant superiority or final test generalization.

**The large Ridge extrapolation failure is avoided, but joint-applicant handling is not solved.** Ensemble predictions on the previous extreme-ratio rows remain in an ordinary interest-rate range instead of producing Ridge's enormous values. That diagnostic slice was selected by Ridge's validation errors and is not an independent benchmark. Group metrics below retain all validation loans and show the remaining application-type differences:

```text
                 model application_type      n      mae     rmse  mean_residual  min_prediction  max_prediction
         random_forest       Individual 846202 2.949591 3.657885      -1.027220        6.358212       23.029102
         random_forest        Joint App   9300 3.068793 3.806684      -0.988379        6.441917       22.942695
hist_gradient_boosting       Individual 846202 2.886416 3.585494      -1.076088        5.436116       25.317142
hist_gradient_boosting        Joint App   9300 3.111074 3.828016      -1.285310        5.553660       25.565252
```

Joint applications were absent from the training cohort. More stable tree predictions do not establish that the primary-income ratio is an appropriate affordability measure for them. Revisit application-appropriate features as an explicit validation-informed development experiment before relying on models that extrapolate beyond their training feature support. No such repair was applied in this notebook.

**Classification still requires substantial caution.** The highest observed ensemble validation macro F1 is 0.422 (hist_gradient_boosting), a change of +0.033 versus the single tree's 0.389. These are scores under the existing default decision rules; no validation thresholds were tuned.

- **hist_gradient_boosting:** At risk precision 3.75%, recall 9.93%; Default precision 35.74%, recall 55.02%. The share of At risk predictions that are false positives is 96.25%.

- **random_forest:** At risk precision 3.56%, recall 23.55%; Default precision 34.64%, recall 46.22%. The share of At risk predictions that are false positives is 96.44%.

**Training fit is not the objective.** Selected classification settings show these mean training-to-fold comparisons: hist_gradient_boosting: training macro F1 0.560, fold-validation macro F1 0.397; random_forest: training macro F1 0.388, fold-validation macro F1 0.383. Large gaps indicate limited transfer to later folds; small gaps do not establish adequate minority detection. Macro F1 can improve through Default detection while At risk remains weak. Always retain both minority classes in the assessment.

**Handoff:** retain both ensemble families as development references for the planned deep-learning notebook. Preserve the frozen target definitions, input eligibility, complete evaluation cohorts, and final-test isolation. The primary-income ratio issue is especially relevant for models that can extrapolate. None of these development results establishes calibrated probabilities, fairness, or suitability for automated lending decisions.
