# Notebook 09c — target/predictor audit results

- No confirmed common observation cutoff is established by the file. Scheduled-term scenarios and post-event date ranges are audit-only; no historical backtest is claimed.
- Frozen classification cohort coverage was verified against known outcomes in the audit source. No final test rows were read and no target labels or splits were changed.
- The explicit leakage guard now covers the source payment-amount alias and three hardship/payment auxiliaries; earlier allowlists already excluded these inputs.

- **logistic/current**: macro F1 **0.377**; At risk AP **0.0317**, recall **45.93%**, precision **3.22%**; Default AP **0.2872**.
- **boosting/current**: macro F1 **0.422**; At risk AP **0.0284**, recall **9.93%**, precision **3.75%**; Default AP **0.3730**.
- **logistic/pricing_grade**: macro F1 **0.372**; At risk AP **0.0266**, recall **41.87%**, precision **3.04%**; Default AP **0.2868**.
- **boosting/pricing_grade**: macro F1 **0.419**; At risk AP **0.0277**, recall **9.90%**, precision **3.54%**; Default AP **0.3703**.
- **logistic/pricing_bureau**: macro F1 **0.372**; At risk AP **0.0266**, recall **41.87%**, precision **3.04%**; Default AP **0.2871**.
- **boosting/pricing_bureau**: macro F1 **0.419**; At risk AP **0.0273**, recall **10.28%**, precision **3.46%**; Default AP **0.3695**.

Training-CV goal-specific selections:
- **At risk: logistic/current**, CV mean AP **0.0149**.
- **Default: boosting/pricing_grade**, CV mean AP **0.3410**.

- Inspect selected_goal_operating_points.csv for infeasible training precision targets and any validation failures to meet feasible targets. Do not retune those cutoffs using validation.
- New field availability assumptions, deferred inputs, vintage missingness and joint non-applicability are recorded in the column inventory and audit tables. Conditional source provenance is not established by a higher predictive score.
- Experiment runtime: **14.81 minutes**.

## Assessment and development recommendations

**The strongest finding is a target-support gap.** Training contains 0 At risk examples among 36-month loans, versus 4,586 in validation. Those validation examples constitute 48.3% of all validation At risk cases. All 611 training examples are from 60-month loans. For the current weighted logistic model, validation At risk recall is 0.07% for 36-month loans versus 88.8% for 60-month loans. This is not repaired by duplicating existing minority examples. It is consistent with unequal follow-up and transient snapshot status, but does not prove which mechanism generated each label. Earlier delinquencies that have subsequently resolved cannot be reconstructed from a final status alone.

The term-only training-prior diagnostic already gives validation At risk AP **0.0228**, versus prevalence **0.0140**. Full models still need to add ranking information beyond this coarse segmentation. This comparison illustrates the importance of term/follow-up support; it does not establish a causal decomposition of model performance.

**A defensible historical observation cutoff cannot be recovered here.** The source contains later payment/credit dates, but they do not timestamp every label. Scheduled maturity scenarios are not actual observation times, event histories, or evidence that labels were observable at a past training date. The target and frozen chronological partitions are retained. A fixed-horizon redesign would require verified outcome observation dates/event histories and a documented observation policy, not an assumed date chosen to improve scores.

**Adding available-looking inputs is not an established solution.** At risk training-CV AP is highest for logistic/current. The Default CV choice is boosting/pricing_grade; its mean AP advantage over the current boosting control is approximately 0.000113, much smaller than fold variability. A numerical first place is not convincing evidence of feature superiority. Conditional bureau timestamp provenance also cannot be certified by higher scores. Earlier plans and models are not overwritten. Both current-input controls reproduced their earlier validation probabilities within numerical tolerance after the leakage-guard fix.

**Review objectives expose limits that macro F1 hides.** No tested At risk rule satisfied any declared precision floor (starting at 5%) with at least 20 alerts in every CV fold. This means infeasible within this finite model/grid/support experiment, not a universal impossibility theorem. Those rules were not relaxed using validation. Default rules can meet these low floors, but some alert nearly everyone: floors of 5%/10% are below the Default base rate and are weak constraints. Alert volume and fixed 1%/5% review-budget results must accompany precision and recall. These are separate review queues, not a replacement target or an approved lending policy.

**Changes to carry forward:** retain the stronger leakage guard, explicit input-availability inventory, class-specific AP and precision–recall reporting, review workloads, and a joint-applicant treatment that respects missing/non-applicable values. Do not promote the new predictor scenarios as a proven performance repair. Keep prior ensemble benchmarks and the canonical predictor plan intact; preserve the conditional scenarios as research evidence. Before claiming deployment-grade At risk prediction, obtain an agreed horizon and outcome-timing provenance, or state clearly that the assignment models snapshot status with unequal follow-up. The planned neural-network comparison can still be completed on the mandated task, with these limitations and training-only robust preprocessing stated explicitly. Final test data remain unread.
