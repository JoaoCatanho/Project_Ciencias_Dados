# =============================================================================
# R/src/hypothesis.R
# -----------------------------------------------------------------------------
# Hypothesis-testing helpers: effect sizes, bootstrap CIs, capped samples.
# Mirrors src/hypothesis.py.
# =============================================================================

# -----------------------------------------------------------------------------
# Hypothesis-testing helpers (effect sizes + bootstrap CI)
# -----------------------------------------------------------------------------

#' Cliff's delta from the Mann-Whitney W statistic:
#' delta = P(X > Y) - P(X < Y) = 2W/(n1*n2) - 1   (ties count 1/2)
#' (as.numeric avoids integer overflow: 200,000 x 45,000 > .Machine$integer.max)
cliffs_delta_from_w <- function(w, n1, n2) {
  2 * as.numeric(w) / (as.numeric(n1) * as.numeric(n2)) - 1
}

#' Cramer's V for a contingency table.
cramers_v <- function(tab) {
  chi2 <- suppressWarnings(chisq.test(tab, correct = FALSE)$statistic)
  n <- sum(tab); k <- min(dim(tab)) - 1
  as.numeric(sqrt(chi2 / (n * k)))
}

#' Percentile bootstrap CI for a statistic of two independent samples.
boot_ci_two_sample <- function(a, b, stat = function(a, b) mean(a) - mean(b),
                               B = 1000, seed = 42, conf = 0.95) {
  set.seed(seed)
  na <- length(a); nb <- length(b)
  reps <- vapply(seq_len(B), function(i) {
    stat(a[sample.int(na, na, replace = TRUE)],
         b[sample.int(nb, nb, replace = TRUE)])
  }, numeric(1))
  alpha <- (1 - conf) / 2
  unname(quantile(reps, c(alpha, 1 - alpha)))
}

#' Mann-Whitney U test + Cliff's delta + bootstrap CI of mean and median diff.
#' Both samples are capped (reproducibly) at n_max rows each.
mw_test <- function(a, b, name, h0, h1, n_max = 200000, seed = 42, B = 1000) {
  a <- cap_sample(a, n_max, seed); b <- cap_sample(b, n_max, seed + 1)
  wt <- wilcox.test(a, b, alternative = "two.sided", exact = FALSE)
  d  <- cliffs_delta_from_w(unname(wt$statistic), length(a), length(b))
  ci_mean <- boot_ci_two_sample(a, b, B = B, seed = seed)
  # median bootstrap on a smaller subsample (medians are slower to resample)
  a_s <- cap_sample(a, 20000, seed); b_s <- cap_sample(b, 20000, seed + 1)
  ci_med <- boot_ci_two_sample(a_s, b_s, function(x, y) median(x) - median(y),
                               B = B, seed = seed)
  data.table(test_id = name, h0 = h0, h1 = h1, test = "Mann-Whitney U",
             statistic = unname(wt$statistic), p_value = wt$p.value,
             effect_size = d, effect_name = "Cliff's delta",
             n_a = length(a), n_b = length(b),
             mean_a = mean(a), mean_b = mean(b),
             median_a = median(a), median_b = median(b),
             ci_mean_diff = sprintf("[%.3f, %.3f]", ci_mean[1], ci_mean[2]),
             ci_median_diff = sprintf("[%.3f, %.3f]", ci_med[1], ci_med[2]))
}

#' Magnitude labels (Romano et al. 2006 for Cliff's delta; Cohen for V).
label_cliff <- function(d) {
  a <- abs(d)
  ifelse(a < 0.147, "negligible", ifelse(a < 0.33, "small",
         ifelse(a < 0.474, "medium", "large")))
}
label_cramer <- function(v) {
  ifelse(v < 0.1, "negligible", ifelse(v < 0.3, "small",
         ifelse(v < 0.5, "medium", "large")))
}
