# =============================================================================
# R/02_data_cleaning.R
# 02 — Data Analysis & Cleaning (Phase 2, Pre-processing)
# =============================================================================
#
# Run from the PROJECT ROOT (the folder that contains R/, data/ and outputs/):
#   Rscript R/02_data_cleaning.R      or      source("R/02_data_cleaning.R", print.eval = TRUE)
# =============================================================================

if (!file.exists("R/src/load_all.R")) {
  stop("Set the working directory to the project root (folder with R/, data/, outputs/).")
}

# ---- Parameters ------------------------------------------------------------
params <- list(
  use_cache = TRUE,
  rej_dti_max = 100,
  observation_cutoff = "2018-12"
)

source("R/src/load_all.R")
library(recipes)
split_cfg  <- read_split_config("outputs/split_definition.json")
TRAIN_END  <- split_cfg$train_end
VAL_END    <- split_cfg$val_end
OBS_CUTOFF <- month_end(params$observation_cutoff)

# R counterpart of `notebooks/02_data_cleaning.ipynb`. Covers:

# 1. source files, schema, units, missing values, duplicates and quality checks;
# 2. exclusions, encodings and transformations (documented);
# 3. chronological split, label availability and leak-safe preprocessing (fit on train only).

# Outputs: tables in `outputs/R/tables/`, processed data in `data/processed/R/` (read by
# scripts 03–05).

# ==== Load the raw files ====================================================

dir.create(PROCESSED_DIR, recursive = TRUE, showWarnings = FALSE)
acc_cache <- processed_path("accepted_parsed")
rej_cache <- processed_path("rejected_parsed")

if (params$use_cache && file.exists(acc_cache)) {
  acc <- readRDS(acc_cache)
} else {
  acc <- load_accepted(file.path("accepted_2007_to_2018Q4.csv"))
  saveRDS(acc, acc_cache)
}
if (params$use_cache && file.exists(rej_cache)) {
  rej <- readRDS(rej_cache)
} else {
  rej <- load_rejected(file.path("rejected_2007_to_2018Q4.csv"))
  saveRDS(rej, rej_cache)
}
n_risk_zero <- attr(rej, "n_risk_score_zero") %||% NA_integer_

data.table(file = c("accepted_2007_to_2018Q4.csv", "rejected_2007_to_2018Q4.csv"),
           rows_loaded = c(nrow(acc), nrow(rej)),
           columns_loaded = c(ncol(acc), ncol(rej)),
           footer_rows_removed = c(attr(acc, "n_footer_rows") %||% NA, NA),
           size_in_R_MB = round(c(object.size(acc), object.size(rej)) / 1024^2))

# Only a fixed list of columns is read from the accepted file (`ACCEPTED_COLS` in
# `R/src/data_loader.R`): every column that is eligible as an input in at least one task, the
# target, the split date and a handful of leakage columns (read only to document their removal).
# Parsing is **deterministic** (fixed date format `%b-%Y`, removing `%`, extracting the number
# of months from `term`) — nothing is estimated from the data at this stage.

# **Units.** Money columns are in USD; `int_rate`, `dti`, `revol_util`, `bc_util` are
# percentages (0–100 scale); `term` is in months; FICO is a score in [300, 850];
# `mo_sin_old_rev_tl_op` is in months.

# ==== Schema reports ========================================================

schema_acc <- data.table(
  column = names(acc),
  type = vapply(acc, function(x) class(x)[1], character(1)),
  missing_pct = round(100 * vapply(acc, function(x) mean(is.na(x)), numeric(1)), 2),
  n_unique = vapply(acc, uniqueN, integer(1))
)[order(-missing_pct)]
save_table(schema_acc, "schema_accepted")
head(schema_acc, 20)

schema_rej <- data.table(
  column = names(rej),
  type = vapply(rej, function(x) class(x)[1], character(1)),
  missing_pct = round(100 * vapply(rej, function(x) mean(is.na(x)), numeric(1)), 2)
)
save_table(schema_rej, "schema_rejected")
schema_rej

# Missingness is not random over time: several bureau fields only start being
# collected around 2012. This matters for the split (train is older than test).
miss_cols <- intersect(c("tot_cur_bal", "tot_hi_cred_lim", "bc_util", "mo_sin_old_rev_tl_op",
                         "mort_acc", "pct_tl_nvr_dlq", "emp_length", "revol_util", "dti"),
                       names(acc))
miss_year <- acc[, lapply(.SD, function(x) mean(is.na(x))),
                 by = .(year = year(issue_d)), .SDcols = miss_cols]
save_table(miss_year, "missingness_by_year")
miss_long <- melt(miss_year, id.vars = "year", variable.name = "column", value.name = "missing")
p <- ggplot(miss_long, aes(factor(year), column, fill = missing)) +
  geom_tile(colour = "white") +
  scale_fill_gradient(low = "#F7F7F7", high = "#C44E52", labels = percent) +
  labs(x = "issue year", y = NULL, fill = "% missing",
       title = "Missingness by issue year (accepted file)")
save_plot(p, "missingness_by_year")
if (interactive()) print(p)

# ==== Duplicates and identifier checks ======================================

dup_tbl <- data.table(
  check = c("exact duplicate rows (accepted)", "duplicate id (accepted)",
            "same (loan_amnt, issue_d, annual_inc, zip_code, earliest_cr_line)",
            "exact duplicate rows (rejected)"),
  count = c(sum(duplicated(acc)), sum(duplicated(acc$id)),
            sum(duplicated(acc, by = c("loan_amnt", "issue_d", "annual_inc",
                                       "zip_code", "earliest_cr_line"))),
            sum(duplicated(rej)))
)
save_table(dup_tbl, "duplicates")
dup_tbl
# Exact duplicates carry no new information -> removed.
acc <- unique(acc)
acc <- acc[!duplicated(id)]
rej <- unique(rej)

# There is no borrower identifier in the public file (`member_id` is empty), so repeated
# borrowers cannot be identified reliably. The 5-column key above is only a rough proxy and is
# reported, not used to remove rows — a limitation to state in the report.

# ==== Quality checks ========================================================

quality <- data.table(
  check = c("accepted: dti < 0 or >= 999 (sentinel values)",
            "accepted: annual_inc == 0",
            "accepted: revol_util > 100",
            "accepted: fico_range_high < fico_range_low",
            "accepted: earliest_cr_line after issue_d",
            "rejected: Risk_Score == 0 (set to NA by the loader)",
            "rejected: risk_score missing (after 0 -> NA)",
            paste0("rejected: DTI > ", params$rej_dti_max, "%"),
            "rejected: DTI > 1000% (clear artefacts, e.g. 4042.11%)",
            "rejected: DTI < 0"),
  count = c(acc[dti < 0 | dti >= 999, .N],
            acc[annual_inc == 0, .N],
            acc[revol_util > 100, .N],
            acc[fico_range_high < fico_range_low, .N],
            acc[earliest_cr_line > issue_d, .N],
            n_risk_zero,
            rej[is.na(risk_score), .N],
            rej[dti > params$rej_dti_max, .N],
            rej[dti > 1000, .N],
            rej[dti < 0, .N])
)
save_table(quality, "quality_checks")
quality

# Decisions (all **fixed rules**, not learned from data, so they can be applied before the split):

# * `dti < 0` or `dti >= 999` in the accepted file are sentinel/invalid values → `NA`.
# * `annual_inc == 0` makes ratios undefined → kept; ratios become `NA` (see `safe_div`).
# * `revol_util > 100` is possible (balance above limit) → kept; extreme values are an outlier
#   question handled later inside the training fold.
# * Rejected `DTI > <params$rej_dti_max>%` → `NA` (same rule as the Python notebook 02).
#   This rule also removes plausible high-DTI rejections; the 1000% threshold would be a less
#   aggressive alternative (both counts are reported above).
acc <- clean_accepted(acc)
rej <- clean_rejected(rej, dti_max = params$rej_dti_max)

# ---- Joint / secondary-applicant fields: not applicable ≠ missing ----------
joint_cols <- intersect(c("annual_inc_joint", "dti_joint", "verification_status_joint"), names(acc))
joint_miss <- acc[, lapply(.SD, function(x) round(100 * mean(is.na(x)), 1)),
                  by = application_type, .SDcols = joint_cols]
save_table(joint_miss, "joint_missingness_by_application_type")
joint_miss

# For `Individual` applications the joint fields are 100% `NA` because they are **not
# applicable**; they are not set to zero. The engineered feature `dti_effective` (script 05)
# uses `dti_joint` for joint applications and `dti` otherwise.

# ==== Targets, cohorts and exclusions =======================================

status_tbl <- acc[, .N, by = loan_status][order(-N)]
status_tbl[, class := fifelse(loan_status %in% STATUS_GOOD, "0 Good",
                       fifelse(loan_status %in% STATUS_AT_RISK, "1 At risk",
                       fifelse(loan_status %in% STATUS_DEFAULT, "2 Default",
                               "excluded (outcome unknown / policy)")))]
save_table(status_tbl, "loan_status_mapping")
status_tbl

acc <- add_target(acc)
acc[, split := assign_split(issue_d, TRAIN_END, VAL_END)]
acc <- add_maturity(acc, OBS_CUTOFF)

exclusions <- data.table(
  step = c("accepted rows after cleaning", "regression cohort (int_rate present)",
           "classification: removed 'Current'",
           "classification: removed 'Does not meet the credit policy' statuses",
           "classification: removed other / missing status",
           "classification cohort"),
  rows = c(nrow(acc), acc[!is.na(int_rate), .N],
           acc[loan_status == "Current", .N],
           acc[grepl("^Does not meet", loan_status), .N],
           acc[is.na(target_class) & loan_status != "Current" &
                 !grepl("^Does not meet", loan_status), .N],
           acc[!is.na(target_class), .N])
)
save_table(exclusions, "exclusions_log")
exclusions

# The "Does not meet the credit policy" loans are older loans issued under a previous credit
# policy. They have a known outcome, but they are excluded (same as in Python) because they
# belong to a population that the current policy would no longer fund. Including them as
# Good/Default would be a valid alternative.

# ==== Chronological split and label availability ============================

sizes <- rbind(
  acc[!is.na(int_rate), .(task = "regression", n = .N), by = split],
  acc[!is.na(target_class), .(task = "classification", n = .N), by = split]
)
sizes <- dcast(sizes, task ~ split, value.var = "n")
save_table(sizes, "split_sizes")
sizes

cls_split <- acc[!is.na(target_class), .N, by = .(split, target_label)]
cls_split[, pct := round(100 * N / sum(N), 2), by = split]
dcast(cls_split, target_label ~ split, value.var = "pct")

# Share of each status group by issue year. Recent vintages are still mostly
# 'Current'; among the ones already resolved, early defaults are over-represented.
st_year <- acc[, .N, by = .(year = year(issue_d),
                            group = fcase(loan_status == "Current", "Current (excluded)",
                                          target_class == 0L, "Good",
                                          target_class == 1L, "At risk",
                                          target_class == 2L, "Default",
                                          default = "Other (excluded)"))]
st_year[, share := N / sum(N), by = year]
years <- sort(unique(st_year$year))
cut_pos <- c(match(as.integer(substr(TRAIN_END, 1, 4)), years),
             match(as.integer(substr(VAL_END, 1, 4)), years)) + 0.5
p <- ggplot(st_year, aes(factor(year), share, fill = group)) +
  geom_col() +
  scale_fill_manual(values = c(CLASS_COLS, "Current (excluded)" = "grey80",
                               "Other (excluded)" = "grey55")) +
  scale_y_continuous(labels = percent) +
  geom_vline(xintercept = cut_pos, linetype = 2) +
  labs(x = "issue year", y = "share of loans", fill = NULL,
       title = "Loan status at the 2018 snapshot, by issue year",
       subtitle = "dashed lines = train | validation | test cutoffs")
save_plot(p, "status_by_issue_year", w = 8.5)
if (interactive()) print(p)

# Default rate among RESOLVED loans by vintage and term, and whether the full
# term has elapsed by the observation cutoff.
fu <- acc[!is.na(target_class),
          .(n_resolved = .N,
            default_rate = round(100 * mean(target_class == 2L), 1),
            pct_full_term_elapsed = round(100 * mean(matured_by_cutoff), 1)),
          by = .(year = year(issue_d), term)][order(term, year)]
save_table(fu, "followup_by_vintage")
fu

# **Unequal follow-up — why this matters.** For recent vintages most loans are still `Current`
# and are removed; the ones that remain are those that *finished early* — prepaid in full or
# already charged off. The default rate among resolved loans therefore looks inflated for later
# years (a property of the snapshot, not of the borrowers). Consequences:

# 1. A split on `issue_d` alone is **not** a historical backtest: a model "trained" at the end of
#    <TRAIN_END> could not have seen the 2018 outcome of a loan issued in 2014.
# 2. A label-availability rule that respects this: when simulating training at date *T*, keep
#    only loans with `issue_d + term ≤ T` (`matured_by_cutoff`, with *T* = observation cutoff).
# 3. The class distribution of the test period (after <VAL_END>) is not comparable with the
#    training period; this must be discussed when reporting test metrics.

# ==== Leakage removal =======================================================

leak <- leakage_columns(names(acc))
exclusions_cols <- data.table(dropped_column = leak, reason = "post-origination leakage")
save_table(exclusions_cols, "exclusions_columns")
acc[, (leak) := NULL]
stopifnot(!any(leak %in% names(acc)))
exclusions_cols

# Task-specific exclusions (`grade`, `sub_grade`, `installment`, … for regression) are not
# dropped from the data — they are simply never selected as regression inputs
# (see `REGRESSION_EXTRA_BANNED` and the eligibility table in script 01).

# ==== Leak-safe preprocessing with `recipes` (fit on train only) ============

# `recipes` is the R equivalent of a scikit-learn `Pipeline`/`ColumnTransformer`:
# `prep()` **learns** the statistics (medians, modes, rare levels, means and SDs) and
# `bake()` **applies** them. We `prep()` on the training partition only and `bake()` the
# validation partition with the stored statistics. The test partition is not touched.
# The recipe itself is defined in `make_recipe()` (`R/src/preprocessing.R`).
reg_cols  <- c("int_rate", NUM_INPUTS, CAT_INPUTS)
reg_train <- as.data.frame(acc[split == "train" & !is.na(int_rate), ..reg_cols])
reg_val   <- as.data.frame(acc[split == "val"   & !is.na(int_rate), ..reg_cols])

rec_reg   <- prep(make_recipe(reg_train, "int_rate"), training = reg_train)
val_baked <- bake(rec_reg, new_data = reg_val)

dir.create("outputs/R/models", recursive = TRUE, showWarnings = FALSE)
saveRDS(rec_reg, "outputs/R/models/recipe_regression_train.rds")

# The normalisation statistics come from the training set only:
norm_step <- tidy(rec_reg, number = length(rec_reg$steps))
head(norm_step[norm_step$statistic == "mean", c("terms", "value")], 8)

# Sanity check: baked TRAIN is centred (mean ~ 0) but baked VALIDATION is not
# exactly 0 — exactly what we expect when the transformation is not refit.
train_baked <- bake(rec_reg, new_data = NULL)
vars <- c("fico_range_low", "dti", "loan_amnt")
data.table(variable = vars,
           mean_train_baked = round(sapply(vars, function(v) mean(train_baked[[v]])), 4),
           mean_val_baked   = round(sapply(vars, function(v) mean(val_baked[[v]])), 4))

# Within cross-validation (Part 2) the same recipe is re-`prep()`ed inside each training fold.

# ==== Persist cleaned data for scripts 03–05 ==============================

reg_cohort <- acc[!is.na(int_rate)]
clf_cohort <- acc[!is.na(target_class)]
saveRDS(reg_cohort, processed_path("accepted_regression"))
saveRDS(clf_cohort, processed_path("accepted_classification"))
saveRDS(rej, processed_path("rejected_clean"))
data.table(file = paste0(c("accepted_regression", "accepted_classification", "rejected_clean"), ".rds"),
           rows = c(nrow(reg_cohort), nrow(clf_cohort), nrow(rej)))

# ==== Summary of decisions (for the mid-journey report) =====================

# * Deterministic parsing (percent signs, dates, term) is done in the loader.
# * Footer rows and exact duplicates removed; duplicate `id`s removed; repeated borrowers cannot
#   be identified (no borrower ID).
# * `Risk_Score == 0` treated as missing; rejected DTI above <params$rej_dti_max>% and accepted
#   DTI sentinels treated as invalid.
# * Joint-applicant fields kept as `NA` where not applicable (no zero-filling).
# * Leakage columns enumerated in `exclusions_columns.csv` and dropped.
# * Split: train ≤ <TRAIN_END>, validation to <VAL_END>, test after (from
#   `split_definition.json`, shared with Python).
# * Unequal follow-up quantified (`followup_by_vintage.csv`).
# * Learned transforms fitted on training only (`recipe_regression_train.rds`).
