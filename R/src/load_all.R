# =============================================================================
# R/src/load_all.R
# -----------------------------------------------------------------------------
# Loads every module of the R codebase. Each notebook starts with
#   source("R/src/load_all.R")
# which is the R equivalent of the `from src.xxx import ...` lines in Python.
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)   # fast loading and manipulation of the large CSVs
  library(ggplot2)      # plots
  library(scales)       # axis formatting
})

# `a %||% b`: a if not NULL, otherwise b (built into base R only from 4.4).
`%||%` <- function(a, b) if (is.null(a)) b else a

for (f in c("data_loader.R", "preprocessing.R", "splits.R", "eda_utils.R",
            "hypothesis.R", "feature_engineering.R")) {
  source(file.path("R", "src", f))
}

# Shared plot settings
theme_set(theme_minimal(base_size = 11))
CLASS_COLS <- c("Good" = "#4C72B0", "At risk" = "#DD8452", "Default" = "#C44E52")
SEED <- 42
