# =============================================================================
# R/src/splits.R
# -----------------------------------------------------------------------------
# Chronological split utilities. Mirrors src/splits.py:
#   * explicit calendar cutoffs (never file order)
#   * same-month records stay in the same partition
#   * label-availability helpers (loan maturity vs observation cutoff)
# =============================================================================

#' Read the split cutoffs written by the Python notebook 01
#' (outputs/split_definition.json), so both languages use the same split.
read_split_config <- function(path = "outputs/split_definition.json") {
  if (file.exists(path)) {
    cfg <- jsonlite::fromJSON(path)
  } else {
    warning("split_definition.json not found; using default cutoffs.")
    cfg <- list(date_col = "issue_d", train_end = "2014-12", val_end = "2016-12")
  }
  cfg
}

#' Last calendar day of a "YYYY-MM" month.
month_end <- function(ym) {
  first <- as.Date(paste0(ym, "-01"))
  seq(first, by = "1 month", length.out = 2)[2] - 1
}

#' Assign each row to train / val / test using explicit calendar cutoffs.
#' Same-month loans always land in the same partition (cutoffs are month ends).
assign_split <- function(dates, train_end, val_end) {
  te <- month_end(train_end)
  ve <- month_end(val_end)
  factor(ifelse(dates <= te, "train", ifelse(dates <= ve, "val", "test")),
         levels = c("train", "val", "test"))
}


#' Whole months between two Dates (b - a).
months_between <- function(a, b) {
  (as.integer(format(b, "%Y")) - as.integer(format(a, "%Y"))) * 12 +
    (as.integer(format(b, "%m")) - as.integer(format(a, "%m")))
}

#' Add the scheduled maturity date and whether the full term had elapsed by
#' the observation cutoff (label-availability diagnostics).
add_maturity <- function(acc, obs_cutoff) {
  acc[, maturity_d := issue_d + round(term * 30.4375)]   # term in months -> days
  acc[, matured_by_cutoff := maturity_d <= obs_cutoff]
  acc[]
}

#' Paths of the processed files shared between the R notebooks
#' (written by 02_data_cleaning.Rmd, read by 03, 04 and 05).
PROCESSED_DIR <- "data/processed/R"
processed_path <- function(name) file.path(PROCESSED_DIR, paste0(name, ".rds"))
