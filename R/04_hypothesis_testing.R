# =============================================================================
# R/04_hypothesis_testing.R
# 04 — Hypothesis Testing (Phase 2)
# =============================================================================
#
# Run from the PROJECT ROOT (the folder that contains R/, data/ and outputs/):
#   Rscript R/04_hypothesis_testing.R      or      source("R/04_hypothesis_testing.R", print.eval = TRUE)
# =============================================================================

if (!file.exists("R/src/load_all.R")) {
  stop("Set the working directory to the project root (folder with R/, data/, outputs/).")
}

# ---- Parameters ------------------------------------------------------------
params <- list(
  test_cap = 200000,
  boot_B = 1000
)

source("R/src/load_all.R")
split_cfg <- read_split_config("outputs/split_definition.json")
VAL_END   <- split_cfg$val_end

# R counterpart of `notebooks/04_hypothesis_testing.ipynb`. For each hypothesis:

# * **H0 / H1** stated explicitly;
# * the **test** is justified and its **assumptions** checked;
# * test statistic, p-value, **effect size** and a bootstrap 95% CI are reported;
# * every interpretation carries a **non-causation** caveat.

# **Scope:** train + validation only (rejected file: application date ≤ end of validation).
# Groups larger than <format(params$test_cap, big.mark = ",")> rows are reduced to a
# reproducible random sample (seed <SEED>); the reported *n* is the number actually tested.
# Because *n* is huge, almost every difference is "significant": **the effect size, not the
# p-value, drives the conclusion.**

# | Test | Python notebook 04 | This script |
# |---|---|---|
# | H1 | DTI vs default | same |
# | H2 | purpose vs default | same |
# | H3 | Risk_Score by state approval group | **score: rejected vs accepted** (spec example) |
# | H4 | amount accepted vs rejected | same (restricted to development period) |
# | H5 | FICO vs default | same |
# | H6 | revol_util vs default | same |
# | H7 | — | int_rate by purpose (Kruskal–Wallis) |

# ==== Load cleaned data and reuse the frozen split ==========================

reg_cohort <- readRDS(processed_path("accepted_regression"))
clf_cohort <- readRDS(processed_path("accepted_classification"))
rej        <- readRDS(processed_path("rejected_clean"))

eda_reg <- reg_cohort[split != "test"]
eda_clf <- clf_cohort[split != "test"]
VAL_END_DATE <- month_end(VAL_END)
acc_dev <- reg_cohort[issue_d <= VAL_END_DATE]
rej_dev <- rej[application_date <= VAL_END_DATE]
data.table(set = c("classification (train+val)", "regression (train+val)",
                   "accepted (dev period)", "rejected (dev period)"),
           rows = c(nrow(eda_clf), nrow(eda_reg), nrow(acc_dev), nrow(rej_dev)))

# ==== Assumption checks (normality) =========================================

norm_tbl <- rbindlist(lapply(c("dti", "fico_range_low", "revol_util", "annual_inc", "int_rate"), function(v) {
  x <- cap_sample(eda_reg[[v]], 5000, SEED)
  st <- shapiro.test(x)
  data.table(variable = v, shapiro_W = unname(st$statistic), shapiro_p = st$p.value,
             skew = skewness(x))
}))
save_table(norm_tbl, "hypothesis_normality")
norm_tbl

# Shapiro–Wilk (on a 5,000-row sample, its maximum size in R) rejects normality and the
# distributions are skewed/discrete, so rank-based tests are used for numeric comparisons
# (Mann–Whitney U for 2 groups, Kruskal–Wallis for k groups) and chi-square for categorical
# associations. Observations are treated as independent loans (no borrower ID to check repeated
# borrowers — limitation).

# ==== H1 — DTI and default ==================================================

# * H0: the distribution of DTI is the same for Default and Good loans.
# * H1: the distributions differ (two-sided). Expected: higher DTI for Default.
# * Test: Mann–Whitney U. Effect size: Cliff's δ.
h1 <- mw_test(eda_clf[target_class == 2L, dti], eda_clf[target_class == 0L, dti],
              "H1 DTI: Default vs Good", "same DTI distribution", "different (two-sided)",
              n_max = params$test_cap, seed = SEED, B = params$boot_B)
h1[, .(statistic, p_value, effect_size, magnitude = label_cliff(effect_size),
       n_a, n_b, median_a, median_b, ci_mean_diff, ci_median_diff)]

# ==== H2 — Loan purpose and default =========================================

# * H0: `purpose` and the 3-class outcome are independent.
# * H1: they are associated.
# * Test: chi-square of independence; purposes with < 1% of loans are merged into `other_rare`
#   so all expected counts are ≥ 5. Effect size: Cramér's V.
pur <- copy(eda_clf[, .(purpose, target_label)])
rare <- pur[, .N, by = purpose][N < 0.01 * nrow(pur), purpose]
pur[purpose %in% rare, purpose := "other_rare"]
tab2 <- table(pur$purpose, pur$target_label)
ct2 <- chisq.test(tab2, correct = FALSE)
v2 <- cramers_v(tab2)
h2 <- data.table(test_id = "H2 purpose x class", h0 = "independent", h1 = "associated",
                 test = "Chi-square", statistic = unname(ct2$statistic), p_value = ct2$p.value,
                 effect_size = v2, effect_name = "Cramer's V", n_a = sum(tab2), n_b = NA_integer_)
save_table(as.data.table(tab2), "hypothesis_purpose_vs_class_table")
data.table(chi2 = ct2$statistic, df = ct2$parameter, p = ct2$p.value,
           cramers_v = v2, magnitude = label_cramer(v2), min_expected = min(ct2$expected))

pur[, .(n = .N, default_share = round(100 * mean(target_label == "Default"), 1)),
    by = purpose][order(-default_share)]

# ==== H3 — Lower risk scores are more likely to be rejected =================

# * H0: the score distribution is the same for rejected and accepted applicants.
# * H1: they differ (expected: lower scores among rejected).
# * Accepted: mid of the FICO band at application; rejected: `Risk_Score` (0 treated as missing).
# * Caveat: the two files may not use exactly the same scoring model over the whole period,
#   and the accepted file only contains *funded* loans (selection). This is a proxy comparison.
acc_score <- acc_dev[, (fico_range_low + fico_range_high) / 2]
rej_score <- rej_dev[!is.na(risk_score), risk_score]
h3 <- mw_test(rej_score, acc_score, "H3 score: Rejected vs Accepted",
              "same score distribution", "different (two-sided)",
              n_max = params$test_cap, seed = SEED, B = params$boot_B)
h3[, .(statistic, p_value, effect_size, magnitude = label_cliff(effect_size),
       n_a, n_b, median_a, median_b, ci_median_diff)]

band <- function(x) cut(x, breaks = seq(500, 850, 25), right = FALSE)
ar <- merge(data.table(band = band(acc_score))[, .(accepted = .N), by = band],
            data.table(band = band(rej_score))[, .(rejected = .N), by = band],
            by = "band", all = TRUE)[!is.na(band)]
ar[is.na(accepted), accepted := 0][is.na(rejected), rejected := 0]
ar[, acceptance_rate := accepted / (accepted + rejected)]
save_table(ar, "acceptance_by_score_band")
p <- ggplot(ar[accepted + rejected >= 200], aes(band, acceptance_rate, group = 1)) +
  geom_line(colour = "#4C72B0") + geom_point(colour = "#4C72B0") +
  scale_y_continuous(labels = percent) +
  labs(x = "score band", y = "share accepted",
       title = "Share of applications accepted by score band (development period)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(p, "acceptance_by_score_band")
if (interactive()) print(p)

# ==== H4 — Requested amount: accepted vs rejected ===========================

# * H0: requested amounts have the same distribution in accepted and rejected applications.
# * H1: they differ (two-sided).
# * `loan_amnt` is the amount applied for (not the funded amount).
h4 <- mw_test(acc_dev$loan_amnt, rej_dev$amount_requested,
              "H4 amount: Accepted vs Rejected", "same distribution", "different (two-sided)",
              n_max = params$test_cap, seed = SEED, B = params$boot_B)
h4[, .(statistic, p_value, effect_size, magnitude = label_cliff(effect_size),
       n_a, n_b, median_a, median_b, ci_median_diff)]

# ==== H5 — FICO and default =================================================

# * H0: the distribution of `fico_range_low` is the same for Default and Good loans.
# * H1: they differ (two-sided). Expected: lower FICO for Default.
h5 <- mw_test(eda_clf[target_class == 2L, fico_range_low], eda_clf[target_class == 0L, fico_range_low],
              "H5 FICO: Default vs Good", "same FICO distribution", "different (two-sided)",
              n_max = params$test_cap, seed = SEED, B = params$boot_B)
h5[, .(statistic, p_value, effect_size, magnitude = label_cliff(effect_size),
       n_a, n_b, median_a, median_b, ci_median_diff)]

# ==== H6 — Revolving utilisation and default ================================

# * H0: the distribution of `revol_util` is the same for Default and Good loans.
# * H1: they differ (two-sided).
h6 <- mw_test(eda_clf[target_class == 2L, revol_util], eda_clf[target_class == 0L, revol_util],
              "H6 revol_util: Default vs Good", "same distribution", "different (two-sided)",
              n_max = params$test_cap, seed = SEED, B = params$boot_B)
h6[, .(statistic, p_value, effect_size, magnitude = label_cliff(effect_size),
       n_a, n_b, median_a, median_b, ci_median_diff)]

# ==== H7 — Interest rate differs by loan purpose (regression target) ========

# * H0: `int_rate` has the same distribution for every purpose.
# * H1: at least one purpose differs.
# * Test: Kruskal–Wallis (k groups, non-normal). Effect size: ε² = (H − k + 1)/(n − k).
kw_df <- sample_rows(eda_reg[, .(int_rate, purpose)], params$test_cap, SEED)
kw <- kruskal.test(int_rate ~ factor(purpose), data = kw_df)
k <- uniqueN(kw_df$purpose); n <- nrow(kw_df)
eps2 <- (unname(kw$statistic) - k + 1) / (n - k)
h7 <- data.table(test_id = "H7 int_rate by purpose", h0 = "same distribution for all purposes",
                 h1 = "at least one differs", test = "Kruskal-Wallis",
                 statistic = unname(kw$statistic), p_value = kw$p.value,
                 effect_size = eps2, effect_name = "epsilon squared", n_a = n, n_b = NA_integer_)
data.table(H = kw$statistic, df = kw$parameter, p = kw$p.value, epsilon2 = eps2,
           magnitude = fifelse(eps2 < 0.01, "negligible", fifelse(eps2 < 0.06, "small",
                       fifelse(eps2 < 0.14, "medium", "large"))))

# ==== Consolidated results ==================================================

cols <- c("test_id", "test", "statistic", "p_value", "effect_name", "effect_size", "n_a", "n_b")
hyp <- rbindlist(lapply(list(h1, h2, h3, h4, h5, h6, h7), function(x) x[, ..cols]))
hyp[, p_holm := p.adjust(p_value, method = "holm")]   # correction for 7 tests
hyp[, decision := fifelse(p_holm < 0.05, "reject H0", "do not reject H0")]
save_table(hyp, "hypothesis_results")
hyp[, .(test_id, test, p_holm = signif(p_holm, 3), effect_name,
        effect_size = round(effect_size, 3), decision, n_a, n_b)]

# ==== Interpretation guidelines for the mid-journey report ==================

# For each hypothesis write one paragraph: *test and why → assumptions checked → decision at
# α = 0.05 (Holm-adjusted) → effect size and CI → practical meaning → "observational association,
# not evidence of causation"*.

# **Effect-size conventions.** Cliff's δ: |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium,
# otherwise large. Cramér's V: 0.1 small, 0.3 medium, 0.5 large. ε²: 0.01 small, 0.06 medium,
# 0.14 large.

# **Limitations to mention.** Selection bias (accepted = funded loans only; the rejected file is a
# different credit snapshot, so H3/H4 are proxy tests); unequal follow-up (see script 02);
# fairness (state-level results are not policy recommendations); class imbalance (*At risk* is a
# very small class).
