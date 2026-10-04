# =============================================================================
# R/05_feature_engineering_model_plan.R
# 05 — Feature Engineering & Model Plan (Phase 3)
# =============================================================================
#
# Run from the PROJECT ROOT (the folder that contains R/, data/ and outputs/):
#   Rscript R/05_feature_engineering_model_plan.R      or      source("R/05_feature_engineering_model_plan.R", print.eval = TRUE)
# =============================================================================

if (!file.exists("R/src/load_all.R")) {
  stop("Set the working directory to the project root (folder with R/, data/, outputs/).")
}

source("R/src/load_all.R")
split_cfg <- read_split_config("outputs/split_definition.json")
TRAIN_END <- split_cfg$train_end
VAL_END   <- split_cfg$val_end

# R counterpart of `notebooks/05_feature_engineering_model_plan.ipynb`. Covers:

# * at least 10 new features, each with source columns, formula, rationale, missing-value
#   handling and eligibility per prediction time;
# * task-to-model plan, validation strategy, metrics and computational budget;
# * reproducible sampling.

# ==== Load cleaned data (from script 02) ==================================

reg_cohort <- readRDS(processed_path("accepted_regression"))
clf_cohort <- readRDS(processed_path("accepted_classification"))

# ==== Feature engineering ===================================================

# All engineered features are **row-wise formulas** (`add_features()` in
# `R/src/feature_engineering.R`): no statistic is learned from other rows, so computing them
# before the split does not leak information. Their missing values are imputed later inside the
# recipe, fitted on training data only.
reg_cohort <- add_features(reg_cohort)
clf_cohort <- add_features(clf_cohort)
fcat <- feature_catalog()
save_table(fcat, "feature_catalog")
fcat[, .(feature, source_columns, formula, regression_eligible, classification_eligible)]

fcat[, .(feature, rationale, missing_handling)]

# **Not allowed (spec):** any feature built from excluded fields — e.g. `installment / income` in
# the regression task, recovery flags, hardship counts. `payment_to_income`,
# `credit_hist_years_at_issue` and `issue_year` are therefore classification-only.

# ---- Quick check on the training partition ---------------------------------
# Missingness and rank association with each target, on TRAIN only.
ftr_reg <- reg_cohort[split == "train"]
ftr_clf <- clf_cohort[split == "train"]
fcheck <- rbindlist(lapply(fcat$feature, function(f) {
  data.table(feature = f,
             missing_pct = round(100 * mean(is.na(ftr_clf[[f]])), 2),
             spearman_int_rate = if (fcat[feature == f, regression_eligible] == "yes" &&
             uniqueN(na.omit(ftr_reg[[f]])) > 1)
               round(cor(ftr_reg[[f]], ftr_reg$int_rate, method = "spearman",
                         use = "complete.obs"), 3) else NA_real_,
             mean_diff_default_vs_good = round(
               mean(ftr_clf[target_class == 2L][[f]], na.rm = TRUE) -
               mean(ftr_clf[target_class == 0L][[f]], na.rm = TRUE), 3))
}))
save_table(fcheck, "feature_check_train")
fcheck

# This describes the training data; it is not a selection step. Any feature selection will be
# done inside the cross-validation folds in Part 2.

# ==== Task-to-model plan ====================================================

plan <- data.table(
  family = c("Baseline", "kNN (from scratch, NumPy)", "Supervised (non-ensemble) #1",
             "Supervised (non-ensemble) #2", "Ensemble — bagging", "Ensemble — boosting",
             "Deep learning", "Clustering"),
  regression = c("predict train median int_rate", "kNN regressor (mean of k neighbours)",
                 "Ridge / linear regression", "Decision tree regressor",
                 "Random forest regressor", "HistGradientBoosting / XGBoost regressor",
                 "MLP (dense, ReLU, dropout) with linear output", "—"),
  classification = c("predict majority class (Good)", "kNN classifier (majority vote, distance-weighted)",
                     "Multinomial logistic regression (class weights)", "Decision tree classifier (class weights)",
                     "Random forest classifier", "HistGradientBoosting / XGBoost classifier",
                     "MLP with softmax output, weighted cross-entropy", "—"),
  clustering = c("—", "—", "—", "—", "—", "—", "—",
                 "k-means (k = 2..10) and Gaussian mixture / DBSCAN on borrower profile; states via accepted+rejected aggregates"),
  justification = c("reference that every model must beat",
                    "required; sensitive to scaling -> uses the train-fitted recipe; subsample for speed",
                    "interpretable linear baseline; coefficients explain drivers",
                    "captures non-linear thresholds (FICO bands) and interactions",
                    "reduces variance of trees; robust to outliers and skew",
                    "state of the art for tabular data; handles missing values natively",
                    "required; tabular MLP suits ~100 standardised inputs",
                    "unsupervised structure; validated by silhouette / Davies-Bouldin and stability across seeds")
)
save_table(plan, "model_plan")
plan

# ==== Validation strategy ===================================================

# * **Hold-out:** train ≤ <TRAIN_END>, validation to <VAL_END>, test after <VAL_END>
#   (frozen, used once at the end of Part 2).
# * **Time-respecting cross-validation inside the training period** (expanding window, one
#   calendar year per assessment fold — never random K-fold):
train_years  <- sort(unique(year(reg_cohort[split == "train", issue_d])))
assess_years <- tail(train_years, 4)
folds <- rbindlist(lapply(assess_years, function(y) {
  data.table(fold = paste0("F", match(y, assess_years)),
             fit_on = paste0("issue year <= ", y - 1),
             assess_on = paste0("issue year == ", y),
             n_fit_reg = reg_cohort[split == "train" & year(issue_d) < y, .N],
             n_assess_reg = reg_cohort[split == "train" & year(issue_d) == y, .N],
             n_fit_clf_matured = clf_cohort[split == "train" & year(issue_d) < y &
                                              maturity_d <= as.Date(paste0(y, "-01-01")), .N])
}))
save_table(folds, "cv_folds")
folds

# `n_fit_clf_matured` = classification training loans whose full term had elapsed at the start of
# each assessment year, i.e. whose outcome would really have been known then (label-availability
# rule from script 02). It is small for early folds — a real limitation of a historical backtest
# on this snapshot.

# * All learned preprocessing (the recipe) and any resampling for class imbalance are fitted
#   **inside each training fold** only; validation and test sets are never rebalanced.
# * **Metrics.** Regression: MAE, RMSE, R² (mean ± SD across folds, plus the validation/test
#   value). Classification: confusion matrix; macro and micro precision, recall and F1;
#   one-vs-rest AUC; accuracy. Macro-F1 is the main selection metric because *At risk* is tiny.
# * **Computational budget.** The regression cohort has ~2.2M rows. Hyperparameter tuning uses a
#   reproducible random sample of the training period (seed <SEED>, e.g. 200k rows) and a small
#   grid/random search (~20 configurations per model); the final model is refit on the full
#   training period. The from-scratch kNN uses a 50k–100k reference sample because prediction
#   cost grows with *n × m*.
# * **Sampling.** Every sample uses `sample_rows()` / `cap_sample()` with a fixed seed. No analysis
#   uses the first rows of a file.

# ==== Save the feature-enriched data ========================================

saveRDS(reg_cohort, processed_path("accepted_regression_features"))
saveRDS(clf_cohort, processed_path("accepted_classification_features"))

# ==== Session information ===================================================

sessionInfo()
