# R — Part 1 (Phases 1–3)

R implementation of Part 1 as plain R scripts, organised like the Python code:
a codebase of modules (`R/src/`) used by five numbered scripts.

```
R/
├── 00_install_packages.R                 install required packages (run once)
├── run_all.R                             runs the 5 scripts in order
├── 01_problem_formulation.R              <-> notebooks/01_problem_formulation.ipynb
├── 02_data_cleaning.R                    <-> notebooks/02_data_cleaning.ipynb  (writes data/processed/R/*.rds)
├── 03_eda.R                              <-> notebooks/03_eda.ipynb
├── 04_hypothesis_testing.R               <-> notebooks/04_hypothesis_testing.ipynb
├── 05_feature_engineering_model_plan.R   <-> notebooks/05_feature_engineering_model_plan.ipynb
└── src/                                  <-> src/*.py
    ├── load_all.R               sources every module (like `from src... import`)
    ├── data_loader.R            loading + deterministic parsing
    ├── preprocessing.R          leakage lists, target, cleaning rules, recipe
    ├── splits.R                 chronological split, maturity / label availability
    ├── eda_utils.R              sampling, descriptive stats, save tables/plots
    ├── hypothesis.R             effect sizes, bootstrap CI, Mann-Whitney wrapper
    └── feature_engineering.R    17 features + documentation table
```

## How to run

1. Put `accepted_2007_to_2018Q4.csv` and `rejected_2007_to_2018Q4.csv` in `data/raw/`.
2. Set the working directory to the **project root** (the folder with `R/`, `data/`, `outputs/`).
3. Run:

```r
source("R/00_install_packages.R")
source("R/run_all.R")            # or in a terminal: Rscript R/run_all.R
```

Scripts can also be run one at a time, but **02 must run before 03, 04 and 05**, because it
writes the processed data they read. Parameters (sample sizes, thresholds, cache) are at the
top of each script in `params <- list(...)`.

The first run of 02 parses the CSVs (a few minutes, ~3–4 GB RAM) and caches them in
`data/processed/R/`. Set `use_cache = FALSE` in 02 to force a reload.

## Outputs

* `outputs/R/tables/*.csv` — every table (also printed in the console)
* `outputs/R/figures/*.png` — every figure
* `outputs/R/models/recipe_regression_train.rds` — preprocessing recipe fitted on train only

## Consistency with Python

Same split cutoffs (`outputs/split_definition.json`, written by Python notebook 01 — or by
`01_problem_formulation.R` if it does not exist yet), same 3-class mapping, same leakage lists,
same seed (42) and same 200k-per-group cap for the hypothesis tests.

`data/processed/` and `outputs/` should stay out of Git (large / regenerable).
