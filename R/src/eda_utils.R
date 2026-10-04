# =============================================================================
# R/src/eda_utils.R
# -----------------------------------------------------------------------------
# Reproducible sampling, descriptive statistics and output helpers.
# Mirrors src/eda_utils.py.
# =============================================================================

# -----------------------------------------------------------------------------
# Reproducible sampling (never head())
# -----------------------------------------------------------------------------

#' Random sample of n rows with a fixed seed. Returns all rows if n >= nrow.
sample_rows <- function(dt, n, seed = 42) {
  if (n >= nrow(dt)) return(copy(dt))
  set.seed(seed)
  dt[sort(sample.int(nrow(dt), n))]
}

#' Random sample of a vector (capped), reproducible.
cap_sample <- function(x, n_max, seed = 42) {
  x <- x[!is.na(x)]
  if (length(x) <= n_max) return(x)
  set.seed(seed)
  x[sample.int(length(x), n_max)]
}

# -----------------------------------------------------------------------------
# Descriptive statistics
# -----------------------------------------------------------------------------

skewness <- function(x) {
  x <- x[!is.na(x)]; m <- mean(x); s <- sd(x)
  if (is.na(s) || s == 0) return(NA_real_)
  mean((x - m)^3) / s^3
}
kurtosis_excess <- function(x) {
  x <- x[!is.na(x)]; m <- mean(x); s <- sd(x)
  if (is.na(s) || s == 0) return(NA_real_)
  mean((x - m)^4) / s^4 - 3
}

#' Extended describe(): n, missing %, mean, sd, quartiles, skew, kurtosis.
describe_numeric <- function(dt, cols) {
  rbindlist(lapply(cols, function(cl) {
    x <- dt[[cl]]
    q <- quantile(x, c(0, .25, .5, .75, 1), na.rm = TRUE, names = FALSE)
    data.table(variable = cl, n = sum(!is.na(x)),
               missing_pct = round(100 * mean(is.na(x)), 2),
               mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE),
               min = q[1], q25 = q[2], median = q[3], q75 = q[4], max = q[5],
               skew = skewness(x), kurtosis = kurtosis_excess(x))
  }))
}

#' Number of IQR (1.5 x IQR) outliers per column - documentation only.
iqr_outliers <- function(x) {
  x <- x[!is.na(x)]
  q <- quantile(x, c(.25, .75), names = FALSE); i <- q[2] - q[1]
  sum(x < q[1] - 1.5 * i | x > q[2] + 1.5 * i)
}

# -----------------------------------------------------------------------------
# Output helpers (tables and figures)
# -----------------------------------------------------------------------------

save_table <- function(x, name, dir = "outputs/R/tables") {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  fwrite(x, file.path(dir, paste0(name, ".csv")))
  invisible(x)
}

save_plot <- function(p, name, dir = "outputs/R/figures", w = 7, h = 4.5) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(file.path(dir, paste0(name, ".png")), p,
                  width = w, height = h, dpi = 150)
  invisible(p)
}
