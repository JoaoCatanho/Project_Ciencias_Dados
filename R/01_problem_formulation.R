# =============================================================================
# R/01_problem_formulation.R
# 01 — Problem Formulation (Phase 1)
# =============================================================================
#
# Run from the PROJECT ROOT (the folder that contains R/, data/ and outputs/):
#   Rscript R/01_problem_formulation.R      or      source("R/01_problem_formulation.R", print.eval = TRUE)
# =============================================================================

if (!file.exists("R/src/load_all.R")) {
  stop("Set the working directory to the project root (folder with R/, data/, outputs/).")
}

# ---- Parameters ------------------------------------------------------------
params <- list(
  observation_cutoff = "2018-12"
)

source("R/src/load_all.R")

# This script fixes the **tasks, cohorts, targets, prediction times and split design**
# before any data-dependent preprocessing or outcome-informed selection (R counterpart of
# `notebooks/01_problem_formulation.ipynb`). The final test set is reserved here and is not
# used until Phase 5.

# ==== Tasks, goals and populations ==========================================

tasks <- data.table(
  task = c("Regression", "Classification", "Clustering"),
  goal = c("Estimate the interest rate a new applicant would likely be offered",
           "Predict the repayment outcome of a funded loan (3 classes)",
           "Find interpretable groups of borrowers / states"),
  target = c("int_rate (%)",
             "Good (Fully Paid) / At risk (Late 16-30, Late 31-120, In Grace) / Default (Charged Off, Default)",
             "none (unsupervised)"),
  population = c("All funded loans with a valid int_rate",
                 "Funded loans whose status maps to one of the 3 classes; 'Current' excluded (outcome unknown)",
                 "Borrowers: funded loans (train period); States: accepted + rejected aggregates"),
  prediction_time = c("Application, BEFORE pricing (no grade/sub_grade/installment)",
                      "Origination, AFTER pricing (int_rate, grade, installment allowed)",
                      "n/a (descriptive)")
)
save_table(tasks, "task_definitions")
tasks

# ==== Information available at each prediction time (column eligibility) ====

# The rules follow the specification ("Note on data leakage" and the decision-rule table).
# `EXCLUDE` means the column is never used as a predictor for that task.
elig <- rbindlist(list(
  data.table(column = c("loan_amnt", "term", "purpose"),
             regression = "input", classification = "input",
             reason = "requested loan characteristics, known at application"),
  data.table(column = c("emp_length", "home_ownership", "annual_inc", "dti",
                        "fico_range_low", "fico_range_high", "delinq_2yrs",
                        "inq_last_6mths", "open_acc", "total_acc", "pub_rec",
                        "pub_rec_bankruptcies", "revol_bal", "revol_util",
                        "mort_acc", "tot_cur_bal", "tot_hi_cred_lim", "bc_util",
                        "bc_open_to_buy", "mo_sin_old_rev_tl_op",
                        "num_actv_rev_tl", "pct_tl_nvr_dlq", "tax_liens",
                        "collections_12_mths_ex_med", "chargeoff_within_12_mths",
                        "acc_now_delinq", "tot_coll_amt"),
             regression = "input", classification = "input",
             reason = "application / credit-bureau history at application (prior credit, not this loan's outcome)"),
  data.table(column = "earliest_cr_line", regression = "input (via mo_sin_old_rev_tl_op)",
             classification = "input (age relative to issue_d)",
             reason = "credit-history age must not use a future funding date for the pre-pricing task"),
  data.table(column = c("application_type", "annual_inc_joint", "dti_joint"),
             regression = "input", classification = "input",
             reason = "joint fields are not leakage; NA = not applicable for individual apps (no zero-filling)"),
  data.table(column = c("addr_state", "zip_code"),
             regression = "input (fairness-sensitive)", classification = "input (fairness-sensitive)",
             reason = "geography may proxy protected attributes; analysed separately for fairness"),
  data.table(column = c("verification_status", "verification_status_joint"),
             regression = "EXCLUDE", classification = "input",
             reason = "verification result exists at origination, not before pricing"),
  data.table(column = c("int_rate", "grade", "sub_grade", "installment"),
             regression = "EXCLUDE (target / rate-determining)", classification = "input",
             reason = "fixed by pricing; installment reverse-engineers int_rate"),
  data.table(column = c("funded_amnt", "funded_amnt_inv"),
             regression = "EXCLUDE", classification = "EXCLUDE",
             reason = "funding outcome; availability at origination not defensible"),
  data.table(column = c("initial_list_status", "disbursement_method"),
             regression = "EXCLUDE", classification = "EXCLUDE",
             reason = "conditional process information; availability uncertain"),
  data.table(column = "issue_d", regression = "split only",
             classification = "split + input (issue_year)",
             reason = "funding month, not application date"),
  data.table(column = "loan_status", regression = "EXCLUDE",
             classification = "TARGET", reason = "defines the class label"),
  data.table(column = LEAKAGE_EXACT, regression = "EXCLUDE", classification = "EXCLUDE",
             reason = "post-origination performance (leakage)"),
  data.table(column = c("hardship_*", "settlement_*"), regression = "EXCLUDE",
             classification = "EXCLUDE", reason = "post-origination field groups (leakage)"),
  data.table(column = c("id", "member_id", "url", "desc", "policy_code", "pymnt_plan", "emp_title", "title"),
             regression = "EXCLUDE", classification = "EXCLUDE",
             reason = "identifier / administrative / free text")
))
save_table(elig, "column_eligibility")
elig

# ==== Leakage guard lists ===================================================

# The lists in `R/src/preprocessing.R` are the single source of truth for what must never
# become a predictor (same content as `src/preprocessing.py`).
list(leakage_exact = LEAKAGE_EXACT,
     leakage_prefixes = LEAKAGE_PREFIXES,
     regression_extra_banned = REGRESSION_EXTRA_BANNED,
     administrative = ADMIN_IRRELEVANT)

# ==== Split definition ======================================================

# The cutoffs are written by the Python notebook 01 (`outputs/split_definition.json`); R reads
# the same file so both languages use the **same** partitions. If the file does not exist yet,
# it is created here with the same default values.
split_file <- "outputs/split_definition.json"
if (!file.exists(split_file)) {
  dir.create("outputs", showWarnings = FALSE)
  jsonlite::write_json(list(date_col = "issue_d", train_end = "2014-12", val_end = "2016-12"),
                       split_file, auto_unbox = TRUE, pretty = TRUE)
}
split_cfg <- read_split_config(split_file)
data.table(partition = c("train", "validation", "test (held out)"),
           issue_d = c(paste("<=", split_cfg$train_end),
                       paste(">", split_cfg$train_end, "and <=", split_cfg$val_end),
                       paste(">", split_cfg$val_end)))

# * Split variable: `issue_d` (funding month). Cutoffs are month ends, so loans issued in the
#   same month always stay in the same partition.
# * The test partition is **not** used for EDA, hypothesis tests, feature design or fitting any
#   transformation.
# * Observation cutoff for labels: **<format(month_end(params$observation_cutoff))>**. The file
#   is the 2018Q4 release, so `loan_status` is a snapshot taken at the end of 2018; we use that
#   date (not the latest scheduled-payment date) as the common observation point. The
#   consequences of this (unequal follow-up) are analysed in script 02.

# ==== Notes on assumptions and limitations ==================================

# * `issue_d` is the funding month, not the application date. For the pre-pricing regression it
#   is used only to order the data chronologically, never as a predictor.
# * Joint / secondary-applicant fields are not automatically leakage; *not applicable* is kept
#   separate from *missing* and never replaced by zero.
# * No external temporal variables are introduced.
# * Geography (`addr_state`, `zip_code`) is a potential proxy for protected attributes and is
#   treated as a separate fairness consideration.
