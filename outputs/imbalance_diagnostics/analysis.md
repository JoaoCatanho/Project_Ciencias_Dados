# Notebook 09b — observed diagnostic results

Training-CV sampler choice: **hybrid**; feature choice: **joint_robust**; risk threshold: **0.3** (None means argmax).
- Sampling **hybrid**: validation macro F1 **0.412**, At risk precision **7.75%**, recall **1.26%**, AP **0.0324**.
- Sampling **none**: validation macro F1 **0.319**, At risk precision **5.38%**, recall **0.05%**, AP **0.0314**.
- Sampling **over**: validation macro F1 **0.325**, At risk precision **6.75%**, recall **1.47%**, AP **0.0329**.
- Sampling **under**: validation macro F1 **0.405**, At risk precision **5.22%**, recall **0.06%**, AP **0.0316**.
- Sampling **weighted**: validation macro F1 **0.377**, At risk precision **3.22%**, recall **45.93%**, AP **0.0317**.

- Features **current**, decision **argmax**: macro F1 **0.412**, At risk AP **0.0324**, recall **1.26%**; top-1% review precision **6.09%**, recall **4.35%**.
- Features **joint**, decision **argmax**: macro F1 **0.412**, At risk AP **0.0324**, recall **1.40%**; top-1% review precision **6.09%**, recall **4.35%**.
- Features **joint_robust**, decision **argmax**: macro F1 **0.412**, At risk AP **0.0325**, recall **1.09%**; top-1% review precision **5.94%**, recall **4.25%**.
- Features **joint_robust**, decision **risk_threshold_0.3**: macro F1 **0.415**, At risk AP **0.0325**, recall **1.99%**; top-1% review precision **5.94%**, recall **4.25%**.
- Features **original**, decision **argmax**: macro F1 **0.409**, At risk AP **0.0319**, recall **1.06%**; top-1% review precision **6.00%**, recall **4.28%**.

Boosting confirmation: macro F1 **0.414**, versus **0.422** for the matched Notebook 09 reference; reference reused: **False**.
- Interpret precision and recall jointly. Oversampling duplicates independent observations; it does not increase their number. Partial undersampling can discard useful majority information. Balanced weights and sampling need not improve the same classes.
- AP and review-budget metrics assess ranking separately from argmax/threshold decisions. A threshold can redistribute errors without creating predictive signal. Joint/current CV ties are expected when no joint applications occur in training/CV.
- Validation already informed this diagnostic design. These are development comparisons, not final-test claims; no final test rows were read.
- Experiment runtime: **5.93 minutes**.

## What this experiment tells us

**Sampling affects the fitting objective, but higher macro F1 is not the same as better At risk detection.** At current features, undersampling increases validation macro F1 from 0.319 to 0.405. Default recall rises from 4.4% to 28.2%, while At risk recall remains about 0.06%. The hybrid gives macro F1 0.412, but At risk recall only 1.26%. Balanced weights instead give 45.9% At risk recall with only 3.22% precision. These are different operating tradeoffs, not evidence that one configuration solves the rare class.

**Ranking signal barely changes.** At risk average precision stays around 0.031–0.033 across the logistic sampling policies, against validation prevalence 0.014. Oversampling alone makes only a small difference. Changing how often existing records are fitted does not supply new independent minority examples or guarantee better separation. This supports investigating predictor information, temporal/snapshot labels and representation as well as imbalance; it does not prove that the data are intrinsically unlearnable or that every resampling method would fail.

**Feature handling matters for numerical validity, but the tested revision is not a large classification improvement.** Joint/current have identical CV results because there are no joint applications in those training/CV periods. Robust treatment improves CV modestly, yet validation macro F1 is 0.412 versus 0.412 for current features with the same selected sampler. Removing engineered columns gives a similarly small change. These results do not implicate engineered features as the main source of weak At risk detection. The primary-income ratio issue remains a real input-support problem, especially for extrapolating models; its correction should not be described as a demonstrated rare-class solution.

**Decision rules move errors, with limited practical gain here.** The selected treatment has argmax validation At risk recall 1.09% and selected-rule recall 1.99%, while precision changes from 7.55% to 6.89%. Nearly all At risk cases are still missed at this selected operating point. The fixed 1% review budget obtains precision 5.94% and recall 4.25%: above a random-prevalence reference, but still weak. A review budget and acceptable precision/recall must be chosen for the intended use; macro F1 alone can favor a model that effectively neglects one class.

**The selected treatment does not transfer as an overall boosting improvement.** Boosting validation macro F1 is 0.414, versus 0.422 for Notebook 09. At risk recall is 0.40%, versus 9.93%; Default recall is 31.15%, versus 55.02%. Better logistic CV scores did not identify a universally better boosting treatment. Keep the original ensemble results; do not overwrite them with a claimed repair.

**Recommended next steps:** retain this diagnostic in the report, predeclare class-specific or review-budget objectives, and continue model comparisons without claiming imbalance has been solved. Probability calibration, broader sampler strengths or representations, and genuinely additional independent At risk examples remain possible investigations. A fixed-horizon label audit would help assess unequal follow-up if suitable outcome timing exists; do not invent missing dates or silently replace the project's three-class target. Final test data remain unread.
