# =============================================================================
# R/run_all.R
# Runs the five R scripts in order. From the project root:
#   Rscript R/run_all.R      or      source("R/run_all.R")
# 01 -> 02 (writes data/processed/R/*.rds) -> 03, 04, 05 (read them).
# =============================================================================
scripts <- c("01_problem_formulation", "02_data_cleaning", "03_eda",
             "04_hypothesis_testing", "05_feature_engineering_model_plan")
for (s in scripts) {
  message("\n========== Running ", s, " ==========")
  source(file.path("R", paste0(s, ".R")), print.eval = TRUE, local = new.env())
}
message("\nDone. Tables in outputs/R/tables/, figures in outputs/R/figures/.")
