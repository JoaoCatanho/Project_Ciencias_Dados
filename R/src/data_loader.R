library(data.table)
# =============================================================================
# R/src/data_loader.R
# -----------------------------------------------------------------------------
# Loading and DETERMINISTIC parsing of the accepted / rejected CSV files.
# Mirrors src/data_loader.py. No data-dependent transformation happens here
# (no imputation, no scaling): only fixed-format parsing.
# =============================================================================

# Columns read from the accepted file. We do NOT read all 151 columns:
# the excluded ones are either post-origination leakage that is never used,
# administrative noise, or free text. Reading a fixed list keeps memory low
# (~1 GB instead of ~6 GB) and makes the analysis reproducible.
ACCEPTED_COLS <- c(
  # identifiers / dates
  "id", "issue_d",
  # loan characteristics
  "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate",
  "installment", "grade", "sub_grade", "purpose", "initial_list_status",
  "application_type", "disbursement_method",
  # outcome (target for classification only)
  "loan_status",
  # borrower / bureau profile at application
  "emp_length", "home_ownership", "annual_inc", "verification_status",
  "zip_code", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
  "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
  "pub_rec", "pub_rec_bankruptcies", "revol_bal", "revol_util", "total_acc",
  "mort_acc", "tot_cur_bal", "tot_hi_cred_lim", "bc_util", "bc_open_to_buy",
  "mo_sin_old_rev_tl_op", "num_actv_rev_tl", "pct_tl_nvr_dlq", "tax_liens",
  "collections_12_mths_ex_med", "chargeoff_within_12_mths",
  "acc_now_delinq", "tot_coll_amt",
  # joint / secondary applicant
  "annual_inc_joint", "dti_joint", "verification_status_joint",
  # a few leakage columns, read ONLY to document that they exist and are dropped
  "total_pymnt", "recoveries", "last_pymnt_amnt", "last_fico_range_low",
  "hardship_flag", "debt_settlement_flag"
)

# -----------------------------------------------------------------------------
# Deterministic parsing helpers
# -----------------------------------------------------------------------------

#Transforms the percentage strings into numeric values
parse_percent <- function(x) {
  if (is.numeric(x)) return(x)
  suppressWarnings(as.numeric(gsub("%", "", trimws(x), fixed = TRUE)))
}

# 
#' "Dec-2015" -> Date 2015-12-01 (fixed format "%b-%Y", English month names).
parse_month_year <- function(x) {
  old <- Sys.getlocale("LC_TIME")
  on.exit(Sys.setlocale("LC_TIME", old), add = TRUE)
  invisible(Sys.setlocale("LC_TIME", "C"))   # month abbreviations in English, C apppears in every operative System# no lintr
  as.Date(paste0("01-", x), format = "%d-%b-%Y") # paste0 não deixa espaços
}

#' " 36 months" -> 36
parse_term <- function(x) {
  suppressWarnings(as.numeric(gsub("\\D", "", x))) #retira todo o que não seja numeros
}

# 3. Loading
# -----------------------------------------------------------------------------

#' Load and deterministically parse the accepted-loans file.
#'
#' * reads only ACCEPTED_COLS (those that exist in the file)
#' * removes the summary/footer rows at the end of the Kaggle CSV
#'   (they have an `id` like "Total amount funded..." and no loan data)
#' * parses term, percentages and dates with FIXED formats
load_accepted <- function(path, cols = ACCEPTED_COLS) {
  header <- names(fread(path, nrows = 0))
  use    <- intersect(cols, header) # escoplhe apenas as colunas escolhidas
  dt <- fread(path, select = use, na.strings = c("", "NA"),
              colClasses = list(character = c("id", "zip_code")), # estas colunas não podem ser lidas como numeros, por isso passam para strings
              showProgress = FALSE)
  n_raw <- nrow(dt)
  dt <- dt[!is.na(loan_amnt)]# porque para ser empréstimo tem que ter uma valor emprestado
  attr(dt, "n_footer_rows") <- n_raw - nrow(dt)

  dt[, term := parse_term(term)]
  for (col in intersect(c("int_rate", "revol_util", "bc_util"), names(dt))) {
    set(dt, j = col, value = parse_percent(dt[[col]]))
  }
  dt[, issue_d := parse_month_year(issue_d)]
  dt[, earliest_cr_line := parse_month_year(earliest_cr_line)]
  dt[]
}

#' Load and deterministically parse the rejected-applications file.
#'
#' Spec notes applied here:
#'  * Risk_Score == 0 is MISSING, not a real low score
#'  * Debt-To-Income Ratio is a text percentage ("10%") -> numeric
#' Columns are renamed to snake_case for convenience.
load_rejected <- function(path) {
  dt <- fread(path, na.strings = c("", "NA"),
              select = c("Amount Requested", "Application Date",
                         "Risk_Score", "Debt-To-Income Ratio",
                         "State", "Employment Length"),
              showProgress = FALSE)
  setnames(dt,
           c("Amount Requested", "Application Date", "Risk_Score",
             "Debt-To-Income Ratio", "State", "Employment Length"),
           c("amount_requested", "application_date", "risk_score",
             "dti", "state", "emp_length"))
  dt[, application_date := as.Date(application_date, format = "%Y-%m-%d")]
  dt[, dti := parse_percent(dti)]
  n_zero <- dt[risk_score == 0, .N]
  dt[risk_score == 0, risk_score := NA_real_]
  setattr(dt, "n_risk_score_zero", n_zero)
  dt[]
}
