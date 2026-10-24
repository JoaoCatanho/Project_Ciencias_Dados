# =============================================================================
# R/src/preprocessing.R
# -----------------------------------------------------------------------------
# Leakage lists, target derivation, fixed cleaning rules and the leak-safe
# preprocessing recipe. Mirrors src/preprocessing.py.
# =============================================================================

# Post-origination leakage (spec, "Note on data leakage").
# NB: the real Kaggle column is `last_pymnt_amnt` (with an "n").
# As colunas que dão leakage
LEAKAGE_EXACT <- c(
  "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
  "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
  "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
  "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
  "last_fico_range_low", "debt_settlement_flag", "debt_settlement_flag_date"
)
LEAKAGE_PREFIXES <- c("^hardship_", "^settlement_")

# Rate-determining fields: banned for the pre-pricing regression only.
REGRESSION_EXTRA_BANNED <- c("grade", "sub_grade", "installment",
                             "funded_amnt", "funded_amnt_inv",
                             "issue_d", "verification_status",
                             "verification_status_joint",
                             "initial_list_status", "disbursement_method")

# Administrative fields considered irrelevant.
ADMIN_IRRELEVANT <- c("member_id", "url", "desc", "policy_code", "pymnt_plan")

# 3-class target mapping (identical to src/preprocessing.py).
STATUS_GOOD    <- c("Fully Paid")
STATUS_AT_RISK <- c("Late (16-30 days)", "Late (31-120 days)", "In Grace Period")
STATUS_DEFAULT <- c("Charged Off", "Default")
CLASS_LABELS   <- c("0" = "Good", "1" = "At risk", "2" = "Default")

#' Return every leakage column present in `cols` (exact + prefix matches).
leakage_columns <- function(cols) {
  exact  <- intersect(LEAKAGE_EXACT, cols)
  prefix <- grep(paste(LEAKAGE_PREFIXES, collapse = "|"), cols, value = TRUE)
  sort(unique(c(exact, prefix)))
}



#' loan_status -> integer class {0 Good, 1 At risk, 2 Default}; NA otherwise
#' ("Current" and the "Does not meet the credit policy" statuses -> NA).
derive_target <- function(status) {
  out <- rep(NA_integer_, length(status))
  out[status %in% STATUS_GOOD]    <- 0L
  out[status %in% STATUS_AT_RISK] <- 1L
  out[status %in% STATUS_DEFAULT] <- 2L
  out
}

# -----------------------------------------------------------------------------
# Fixed cleaning rules (thresholds set by domain knowledge, NOT learned from
# data, so they can be applied before the split)
# -----------------------------------------------------------------------------

#' Accepted file: dti < 0 or >= 999 are sentinel / invalid values -> NA.
clean_accepted <- function(acc) {
  acc[dti < 0 | dti >= 999, dti := NA_real_]
  acc[]
}

#' Rejected file: DTI below 0 or above `dti_max` (%) -> NA
#' (same rule as the Python notebook 02, dti_max = 100).
clean_rejected <- function(rej, dti_max = 100) {
  rej[dti < 0 | dti > dti_max, dti := NA_real_]
  rej[]
}

#' Add the classification target (integer + labelled factor) to the accepted data.
add_target <- function(acc) {
  acc[, target_class := derive_target(loan_status)]
  acc[, target_label := factor(CLASS_LABELS[as.character(target_class)],
                               levels = CLASS_LABELS)]
  acc[]
}

# -----------------------------------------------------------------------------
# Input lists and leak-safe preprocessing recipe
# -----------------------------------------------------------------------------

NUM_INPUTS <- c("loan_amnt", "term", "annual_inc", "dti", "fico_range_low",
                "fico_range_high", "delinq_2yrs", "inq_last_6mths", "open_acc",
                "total_acc", "pub_rec", "pub_rec_bankruptcies", "mort_acc",
                "revol_bal", "revol_util", "tot_cur_bal", "tot_hi_cred_lim",
                "bc_util", "bc_open_to_buy", "mo_sin_old_rev_tl_op",
                "num_actv_rev_tl", "pct_tl_nvr_dlq")
CAT_INPUTS <- c("home_ownership", "purpose", "addr_state", "application_type")

# Bureau fields that only start being collected around 2012: their missingness
# means "not collected yet", so we keep an explicit indicator.
BUREAU_LATE <- c("tot_cur_bal", "tot_hi_cred_lim", "bc_util", "bc_open_to_buy",
                 "mo_sin_old_rev_tl_op", "num_actv_rev_tl", "pct_tl_nvr_dlq",
                 "mort_acc")

#' Build the preprocessing recipe (the R equivalent of a sklearn Pipeline).
#' Nothing is learned here: statistics are estimated only when `prep()` is
#' called, and we call it on the TRAINING data only.
make_recipe <- function(train_df, outcome) {
  recipes::recipe(as.formula(paste(outcome, "~ .")), data = train_df) |>
    recipes::step_indicate_na(tidyselect::all_of(intersect(BUREAU_LATE, names(train_df)))) |>
    recipes::step_impute_median(recipes::all_numeric_predictors()) |>   # medians from TRAIN
    recipes::step_impute_mode(recipes::all_nominal_predictors()) |>     # modes from TRAIN
    # rare levels (< 1% in TRAIN) merged; "other_rare" because purpose already has "other"
    recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01,
                        other = "other_rare") |>
    # remove variáveis constantes no TREINO antes de criar dummies
    # (application_type só tem "Individual" até 2014: os pedidos conjuntos
    #  só começaram no fim de 2015)
    recipes::step_zv(recipes::all_predictors()) |>
    recipes::step_dummy(recipes::all_nominal_predictors()) |>
    recipes::step_zv(recipes::all_predictors()) |>
    recipes::step_normalize(recipes::all_numeric_predictors())         # mean/sd from TRAIN
}
