# =============================================================================
# R/src/feature_engineering.R
# -----------------------------------------------------------------------------
# Engineered features (row-wise formulas only) and their documentation table.
# =============================================================================

#' "10+ years" -> 10, "< 1 year" -> 0, "3 years" -> 3, "n/a"/"" -> NA
parse_emp_length <- function(x) {
  x <- trimws(x)
  out <- rep(NA_real_, length(x))
  out[x %in% "< 1 year"] <- 0
  out[x %in% "10+ years"] <- 10
  idx <- grepl("^[0-9]+ years?$", x)
  out[idx] <- as.numeric(sub(" .*$", "", x[idx]))
  out
}

# -----------------------------------------------------------------------------
# Feature engineering (deterministic, row-wise; no statistics learned)
# -----------------------------------------------------------------------------

#' Safe division: NA when the denominator is 0 or NA.
safe_div <- function(num, den) ifelse(is.na(den) | den == 0, NA_real_, num / den)

#' Add the engineered features. Every feature is a row-wise formula, so it can
#' be computed before the split without leaking information between rows.
#' Anything that LEARNS from data (imputation, rare-level grouping, scaling)
#' is done later inside a recipe fitted on the training split only.
add_features <- function(dt) {
  dt <- copy(dt)
  dt[, `:=`(
    # F1  mid-point of the FICO band
    fico_mid             = (fico_range_low + fico_range_high) / 2,
    # F2  log income (heavy right tail)
    log_annual_inc       = log1p(annual_inc),
    # F3  requested amount relative to income
    loan_to_income       = safe_div(loan_amnt, annual_inc),
    # F4  revolving balance relative to income
    revol_bal_to_income  = safe_div(revol_bal, annual_inc),
    # F5  employment length in years (numeric)
    emp_length_years     = parse_emp_length(emp_length),
    # F6  explicit flag for unknown employment length
    emp_length_missing   = as.integer(is.na(parse_emp_length(emp_length))),
    # F7  share of credit lines that are currently open
    open_acc_ratio       = safe_div(open_acc, total_acc),
    # F8  any delinquency in the last 2 years
    any_delinq_2yrs      = as.integer(delinq_2yrs > 0),
    # F9  any public derogatory record (pub rec, bankruptcy or tax lien)
    any_derog            = as.integer((pub_rec > 0) |
                                      (!is.na(pub_rec_bankruptcies) & pub_rec_bankruptcies > 0) |
                                      (!is.na(tax_liens) & tax_liens > 0)),
    # F10 total balance over total credit limit (all bureau trade lines)
    total_bal_to_limit   = safe_div(tot_cur_bal, tot_hi_cred_lim),
    # F11 joint application flag
    is_joint             = as.integer(application_type == "Joint App"),
    # F12 DTI that applies to the application (joint DTI for joint apps).
    #     Keeps "not applicable" (individual) separate from "missing".
    dti_effective        = ifelse(application_type == "Joint App" & !is.na(dti_joint),
                                  dti_joint, dti),
    # F13 60-month term flag
    term_60              = as.integer(term == 60),
    # F14 inquiries per open account (credit-seeking intensity)
    inq_per_open_acc     = safe_div(inq_last_6mths, open_acc),
    # F15 [classification only] credit history length at issue date (years)
    credit_hist_years_at_issue = months_between(earliest_cr_line, issue_d) / 12,
    # F16 [classification only] annual payment burden relative to income
    payment_to_income    = safe_div(12 * installment, annual_inc),
    # F17 [classification only] issue year (origination-time context)
    issue_year           = as.integer(format(issue_d, "%Y")),
    # F18 (Ramos) interação entre a dívida atual e o peso do novo empréstimo
    dti_x_loan_to_income = dti * safe_div(loan_amnt, annual_inc),
    # F19 (Ramos) [classificação apenas] idade do histórico de crédito em meses
    credit_hist_months   = as.numeric(issue_d - earliest_cr_line) / 30.44
  )]
  dt[]
}

#' Documentation table for the engineered features (exported to CSV).
feature_catalog <- function() {
  out <- data.table(
    feature = c("fico_mid", "log_annual_inc", "loan_to_income",
                "revol_bal_to_income", "emp_length_years",
                "emp_length_missing", "open_acc_ratio", "any_delinq_2yrs",
                "any_derog", "total_bal_to_limit", "is_joint",
                "dti_effective", "term_60", "inq_per_open_acc",
                "credit_hist_years_at_issue", "payment_to_income",
                "issue_year"),
    source_columns = c("fico_range_low, fico_range_high", "annual_inc",
                       "loan_amnt, annual_inc", "revol_bal, annual_inc",
                       "emp_length", "emp_length", "open_acc, total_acc",
                       "delinq_2yrs",
                       "pub_rec, pub_rec_bankruptcies, tax_liens",
                       "tot_cur_bal, tot_hi_cred_lim", "application_type",
                       "dti, dti_joint, application_type", "term",
                       "inq_last_6mths, open_acc",
                       "earliest_cr_line, issue_d",
                       "installment, annual_inc", "issue_d"),
    formula = c("(low + high) / 2", "log(1 + annual_inc)",
                "loan_amnt / annual_inc", "revol_bal / annual_inc",
                "'< 1 year'->0, '10+ years'->10, 'k years'->k",
                "1 if emp_length unparseable/missing",
                "open_acc / total_acc", "1 if delinq_2yrs > 0",
                "1 if any of the three > 0", "tot_cur_bal / tot_hi_cred_lim",
                "1 if application_type == 'Joint App'",
                "dti_joint if joint, else dti", "1 if term == 60",
                "inq_last_6mths / open_acc",
                "months(issue_d - earliest_cr_line) / 12",
                "12 * installment / annual_inc", "year(issue_d)"),
    rationale = c("single score instead of two collinear bounds",
                  "reduces right skew; income effect is multiplicative",
                  "affordability of the requested loan",
                  "revolving leverage independent of income scale",
                  "employment stability as a number",
                  "missing employment may be informative (self-employed etc.)",
                  "how much of the credit history is still active",
                  "recent repayment problems",
                  "serious past credit events, rare individually",
                  "overall utilisation of available credit",
                  "joint apps use different income/DTI fields",
                  "uses the DTI relevant to the application; avoids zero-filling joint fields",
                  "longer term = higher risk and higher rate",
                  "recent credit-seeking relative to active accounts",
                  "credit age at origination (spec: relative to issue_d)",
                  "real repayment burden at origination",
                  "captures policy/macro drift across vintages"),
    missing_handling = c(
      "NA if any bound NA -> median impute (train fit)",
      "NA if income NA -> median impute (train fit)",
      "NA if income 0/NA -> median impute (train fit)",
      "NA if income 0/NA -> median impute (train fit)",
      "NA kept, flagged by F6 -> median impute (train fit)",
      "never missing",
      "NA if total_acc 0/NA -> median impute (train fit)",
      "NA if delinq_2yrs NA -> mode impute (train fit)",
      "NA only if pub_rec NA -> mode impute (train fit)",
      "NA before ~2012 (bureau field not collected) -> median impute + missing indicator",
      "never missing",
      "NA only if dti NA -> median impute (train fit)",
      "never missing",
      "NA if open_acc 0/NA -> median impute (train fit)",
      "NA if earliest_cr_line NA -> median impute (train fit)",
      "NA if income 0/NA -> median impute (train fit)",
      "never missing"),
    regression_eligible = c(rep("yes", 14), "no (uses issue_d, a future funding date)",
                            "no (installment encodes int_rate)",
                            "no (issue_d is not a pre-pricing input)"),
    classification_eligible = rep("yes", 17)
  )
  novas <- data.table(
    feature = c("dti_x_loan_to_income", "credit_hist_months"),
    source_columns = c("dti, loan_amnt, annual_inc", "earliest_cr_line, issue_d"),
    formula = c("dti * loan_amnt / annual_inc", "(issue_d - earliest_cr_line) em dias / 30.44"),
    rationale = c("dívida existente x peso do novo empréstimo (alavancagem composta)",
                  "histórico de crédito mais longo = mais estabilidade"),
    missing_handling = c("NA se dti/income NA ou income 0 -> median impute (train fit)",
                         "NA se earliest_cr_line NA -> median impute (train fit)"),
    regression_eligible = c("yes", "no (uses issue_d, a future funding date)"),
    classification_eligible = c("yes", "yes"))
  out <- rbind(out, novas)
  out[, selected_regression := feature %in% FEATURES_REG]
  out[, selected_classification := feature %in% FEATURES_CLF]
  out[]
}

# Conjunto final acordado com o Ramos (10 por tarefa, ordenadas por |Spearman| com int_rate)
FEATURES_REG <- c("fico_mid", "dti_x_loan_to_income", "loan_to_income",
                  "inq_per_open_acc", "total_bal_to_limit", "any_delinq_2yrs",
                  "open_acc_ratio", "log_annual_inc", "emp_length_years",
                  "emp_length_missing")
FEATURES_CLF <- c(setdiff(FEATURES_REG, "open_acc_ratio"), "credit_hist_months")
