# =============================================================================
# R/03_eda.R
# 03 — Exploratory Data Analysis (Phase 2)
# =============================================================================
#
# Run from the PROJECT ROOT (the folder that contains R/, data/ and outputs/):
#   Rscript R/03_eda.R      or      source("R/03_eda.R", print.eval = TRUE)
# =============================================================================

if (!file.exists("R/src/load_all.R")) {
  stop("Set the working directory to the project root (folder with R/, data/, outputs/).")
}

# ---- Parameters ------------------------------------------------------------
params <- list(
  plot_sample_n = 20000,
  pca_sample_n = 50000,
  umap_sample_n = 15000
)

source("R/src/load_all.R")
library(uwot)   # UMAP
split_cfg <- read_split_config("outputs/split_definition.json")
VAL_END   <- split_cfg$val_end

# R counterpart of `notebooks/03_eda.ipynb`. Scope:

# * descriptive statistics and visualisations on **train + validation only** (test untouched);
# * outliers, associations and patterns;
# * linear (PCA) and non-linear (UMAP) dimensionality reduction;
# * initial insights.

# ==== Load cleaned data (from script 02) ==================================

reg_cohort <- readRDS(processed_path("accepted_regression"))
clf_cohort <- readRDS(processed_path("accepted_classification"))
rej        <- readRDS(processed_path("rejected_clean"))

# EDA is done on train + validation ONLY. Test is untouched.
eda_reg <- reg_cohort[split != "test"]
eda_clf <- clf_cohort[split != "test"]
EDA_NUM <- intersect(NUM_INPUTS, names(eda_reg))
data.table(set = c("regression EDA", "classification EDA"),
           rows = c(nrow(eda_reg), nrow(eda_clf)))

# ==== Descriptive statistics ================================================

desc_reg <- describe_numeric(eda_reg, c("int_rate", EDA_NUM))
save_table(desc_reg, "eda_describe_regression")
desc_reg[, lapply(.SD, function(x) if (is.numeric(x)) signif(x, 4) else x)]

# ==== Histograms ============================================================

# Heavy-tailed money variables are shown on `log1p` scale.
heavy <- c("annual_inc", "revol_bal", "tot_cur_bal", "tot_hi_cred_lim", "bc_open_to_buy")
h_sample <- sample_rows(eda_reg, params$plot_sample_n * 5, SEED)
h_long <- melt(h_sample[, c("int_rate", EDA_NUM), with = FALSE],
               measure.vars = c("int_rate", EDA_NUM), na.rm = TRUE)
h_long[, variable := as.character(variable)]
h_long[variable %in% heavy, `:=`(value = log1p(value), variable = paste0("log1p(", variable, ")"))]
p <- ggplot(h_long, aes(value)) +
  geom_histogram(bins = 40, fill = "#4C72B0", colour = "white", linewidth = 0.1) +
  facet_wrap(~ variable, scales = "free", ncol = 4) +
  labs(x = NULL, y = "count",
       title = "Distributions of numeric inputs (train + validation, random sample)")
save_plot(p, "histograms_numeric", w = 11, h = 10)
if (interactive()) print(p)

# ==== Interest rate: drift over time and relation with key inputs ===========

p <- ggplot(eda_reg[, .(int_rate, year = factor(year(issue_d)))], aes(year, int_rate)) +
  geom_boxplot(outlier.size = 0.3, fill = "#4C72B0", alpha = 0.4) +
  labs(x = "issue year", y = "int_rate (%)",
       title = "Interest rate by vintage (pricing policy drifts over time)")
save_plot(p, "int_rate_by_year")
if (interactive()) print(p)

s <- sample_rows(eda_reg, params$plot_sample_n, SEED)
s[, fico_mid := (fico_range_low + fico_range_high) / 2]
p <- ggplot(s, aes(fico_mid, int_rate)) +
  geom_bin_2d(bins = 40) +
  geom_smooth(method = "loess", colour = "#C44E52", se = FALSE, linewidth = 0.8) +
  scale_fill_gradient(low = "#DCE6F2", high = "#1F3B63") +
  labs(x = "FICO (mid of band)", y = "int_rate (%)",
       title = "Interest rate vs FICO (random sample)")
save_plot(p, "int_rate_vs_fico")
if (interactive()) print(p)

p <- ggplot(s[!is.na(dti)], aes(dti, int_rate)) +
  geom_bin_2d(bins = 40) +
  geom_smooth(method = "loess", colour = "#C44E52", se = FALSE, linewidth = 0.8) +
  scale_fill_gradient(low = "#DCE6F2", high = "#1F3B63") +
  labs(x = "DTI (%)", y = "int_rate (%)", title = "Interest rate vs DTI (random sample)")
save_plot(p, "int_rate_vs_dti")
if (interactive()) print(p)

purpose_order <- eda_reg[, .(m = median(int_rate)), by = purpose][order(m), purpose]
p <- ggplot(eda_reg[, .(purpose = factor(purpose, purpose_order), int_rate,
                        term = paste(term, "months"))],
            aes(purpose, int_rate, fill = term)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) +
  coord_flip() +
  scale_fill_manual(values = c("#4C72B0", "#DD8452")) +
  labs(x = NULL, y = "int_rate (%)", fill = NULL, title = "Interest rate by purpose and term")
save_plot(p, "int_rate_by_purpose_term", h = 5.5)
if (interactive()) print(p)

# ==== Classification target: balance and risk gradients =====================

cb <- eda_clf[, .N, by = target_label][, pct := N / sum(N)]
save_table(cb, "eda_class_balance")
p <- ggplot(cb, aes(target_label, N, fill = target_label)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = percent(pct, 0.1)), vjust = -0.3) +
  scale_fill_manual(values = CLASS_COLS) +
  scale_y_continuous(labels = comma, expand = expansion(c(0, .1))) +
  labs(x = NULL, y = "loans", title = "Class balance (train + validation)")
save_plot(p, "class_balance", h = 3.5)
if (interactive()) print(p)

g <- eda_clf[, .(fico_band = cut(fico_range_low, breaks = seq(600, 850, 20), right = FALSE),
                 default = target_class == 2L, term = paste(term, "m"))]
gf <- g[!is.na(fico_band), .(default_rate = mean(default), n = .N), by = .(fico_band, term)]
p <- ggplot(gf[n >= 100], aes(fico_band, default_rate, colour = term, group = term)) +
  geom_line() + geom_point() +
  scale_y_continuous(labels = percent) +
  scale_colour_manual(values = c("#4C72B0", "#DD8452")) +
  labs(x = "FICO band", y = "default share", colour = NULL,
       title = "Default share by FICO band and term (train + validation)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(p, "default_by_fico_term")
if (interactive()) print(p)

# ==== Correlation heatmap (Spearman) ========================================

# Spearman is used because it is robust to skew, outliers and monotone non-linearity.
cs <- sample_rows(eda_reg, params$pca_sample_n * 2, SEED)
cm <- cor(cs[, c("int_rate", EDA_NUM), with = FALSE], method = "spearman",
          use = "pairwise.complete.obs")
save_table(as.data.table(cm, keep.rownames = "variable"), "corr_spearman_regression")
cm_long <- as.data.table(as.table(cm))
setnames(cm_long, c("x", "y", "rho"))
p <- ggplot(cm_long, aes(x, y, fill = rho)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", limits = c(-1, 1)) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  labs(x = NULL, y = NULL, title = "Spearman correlation (train + validation)")
save_plot(p, "corr_heatmap", w = 8.5, h = 7.5)
if (interactive()) print(p)

data.table(variable = rownames(cm), rho_with_int_rate = round(cm[, "int_rate"], 3))[
  variable != "int_rate"][order(-abs(rho_with_int_rate))][1:10]

# ==== Linear dimensionality reduction — PCA =================================

# PCA is fitted on a reproducible random sample of the **training** partition (complete cases of
# the numeric inputs, standardised). The scaling inside `prcomp` is learned from that sample only.
pca_df <- sample_rows(na.omit(reg_cohort[split == "train", c("int_rate", EDA_NUM), with = FALSE]),
                      params$pca_sample_n, SEED)
pca <- prcomp(pca_df[, ..EDA_NUM], center = TRUE, scale. = TRUE)
ev <- data.table(PC = seq_along(pca$sdev), var_explained = pca$sdev^2 / sum(pca$sdev^2))
ev[, cumulative := cumsum(var_explained)]
save_table(ev, "pca_explained_variance")
p <- ggplot(ev[PC <= 15], aes(PC)) +
  geom_col(aes(y = var_explained), fill = "#4C72B0") +
  geom_line(aes(y = cumulative), colour = "#C44E52") +
  geom_point(aes(y = cumulative), colour = "#C44E52") +
  scale_y_continuous(labels = percent) +
  labs(y = "variance explained", title = "PCA scree plot (bars) and cumulative (line)")
save_plot(p, "pca_scree")
if (interactive()) print(p)

load_tbl <- as.data.table(pca$rotation[, 1:3], keep.rownames = "variable")
save_table(load_tbl, "pca_loadings")
load_tbl[order(-abs(PC1))][1:8, .(variable, PC1 = round(PC1, 3), PC2 = round(PC2, 3))]

scores <- data.table(pca$x[, 1:2], int_rate = pca_df$int_rate)
p <- ggplot(sample_rows(scores, params$plot_sample_n, SEED), aes(PC1, PC2, colour = int_rate)) +
  geom_point(size = 0.4, alpha = 0.5) +
  scale_colour_viridis_c() +
  labs(title = "Training sample projected on PC1-PC2, coloured by int_rate")
save_plot(p, "pca_pc1_pc2")
if (interactive()) print(p)

# ==== Non-linear dimensionality reduction — UMAP ============================

u_df <- sample_rows(na.omit(clf_cohort[split == "train", c("target_label", EDA_NUM), with = FALSE]),
                    params$umap_sample_n, SEED)
X <- scale(as.matrix(u_df[, ..EDA_NUM]))
set.seed(SEED)
emb <- umap(X, n_neighbors = 30, min_dist = 0.1, n_threads = 1, seed = SEED)
emb_dt <- data.table(UMAP1 = emb[, 1], UMAP2 = emb[, 2], class = u_df$target_label)
p <- ggplot(emb_dt[order(class)], aes(UMAP1, UMAP2, colour = class)) +
  geom_point(size = 0.4, alpha = 0.5) +
  scale_colour_manual(values = CLASS_COLS) +
  guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  labs(title = "UMAP of numeric inputs (training sample), coloured by class")
save_plot(p, "umap_classes", h = 5)
if (interactive()) print(p)

# ==== Outlier inspection (documentation only) ===============================

out_tbl <- data.table(variable = EDA_NUM,
                      iqr_outliers = sapply(EDA_NUM, function(v) iqr_outliers(eda_reg[[v]])))
out_tbl[, pct := round(100 * iqr_outliers / nrow(eda_reg), 2)]
save_table(out_tbl[order(-iqr_outliers)], "eda_iqr_outliers")
out_tbl[order(-iqr_outliers)][1:10]

# Outliers are **not** removed here. Clipping/winsorising learns thresholds from data, so if used
# it will be a step inside the recipe, fitted on training folds only.

# ==== Rejected applications (application date ≤ end of validation period) ===

VAL_END_DATE <- month_end(VAL_END)
rej_dev <- rej[application_date <= VAL_END_DATE]
acc_dev <- reg_cohort[issue_d <= VAL_END_DATE]

amt <- rbind(data.table(group = "accepted (loan_amnt)", amount = cap_sample(acc_dev$loan_amnt, 1e5, SEED)),
             data.table(group = "rejected (Amount Requested)", amount = cap_sample(rej_dev$amount_requested, 1e5, SEED)))
p <- ggplot(amt, aes(amount, fill = group)) +
  geom_density(alpha = 0.45, colour = NA) +
  scale_x_continuous(labels = dollar, limits = c(0, 40000)) +
  scale_fill_manual(values = c("#4C72B0", "#C44E52")) +
  labs(x = "requested amount", fill = NULL,
       title = "Requested amount: accepted vs rejected (development period)")
save_plot(p, "amount_accepted_vs_rejected")
if (interactive()) print(p)

p <- ggplot(rej_dev[!is.na(risk_score)][sample.int(.N, min(.N, 1e5))], aes(risk_score)) +
  geom_histogram(bins = 50, fill = "#C44E52", colour = "white", linewidth = 0.1) +
  labs(x = "Risk_Score (0 treated as missing)", y = "count",
       title = "Risk score of rejected applications (development period)")
save_plot(p, "rejected_risk_score")
if (interactive()) print(p)

# ==== Initial insights ======================================================

# The values below are computed by the code; the bullet points are what to check and write up.
top_corr <- data.table(variable = rownames(cm), rho = cm[, "int_rate"])[
  variable != "int_rate"][order(-abs(rho))][1:3]
miss_year <- eda_reg[, .(tot_cur_bal_missing = mean(is.na(tot_cur_bal))), by = .(year = year(issue_d))]
insights <- data.table(
  insight = c("most heavy-tailed inputs (skew)",
              "strongest Spearman correlates of int_rate",
              "PCs needed for 80% of variance",
              "minority class share (At risk)",
              "vintages where bureau fields are mostly missing"),
  value = c(paste(desc_reg[order(-abs(skew))][1:3, variable], collapse = ", "),
            paste(sprintf("%s (%.2f)", top_corr$variable, top_corr$rho), collapse = ", "),
            as.character(ev[cumulative >= 0.8, min(PC)]),
            percent(cb[target_label == "At risk", pct], 0.01),
            paste(sort(miss_year[tot_cur_bal_missing > 0.5, year]), collapse = ", "))
)
save_table(insights, "eda_insights")
insights

# * Heavy-tailed money variables → log transform (`log_annual_inc` in script 05) or robust scaling.
# * FICO, term, DTI and utilisation dominate the (non-banned) association with `int_rate`;
#   the rate also drifts by vintage, so a chronological validation is essential.
# * PCA: if many components are needed for 80% of the variance, the inputs carry partly
#   independent information and no single direction dominates.
# * UMAP: strong class overlap → the 3-class problem is hard, especially *At risk*.
# * Associations seen here are observational and may be confounded by vintage
#   (riskier products and different policies in different years).
