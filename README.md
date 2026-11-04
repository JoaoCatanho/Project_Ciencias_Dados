# Projeto de Ciência de Dados — Lending Club

Projeto da unidade curricular de **Ciência de Dados** (Universidade da Madeira).
Usamos os dados públicos da Lending Club (empréstimos aceites e pedidos rejeitados, 2007–2018) para duas tarefas:

* **Regressão:** prever a taxa de juro (`int_rate`) de um empréstimo aceite;
* **Classificação:** prever o estado do empréstimo em 3 classes (Good / At risk / Default).

A análise está feita em **Python** (notebooks Jupyter + módulos) e em **R** (scripts numerados + módulos), com as mesmas regras: mesmo split temporal (treino até 2014-12, validação até 2016-12, teste depois), mesmas listas de *leakage* e a mesma seed (42).

## Estrutura do repositório

```
Project_Ciencias_Dados/
├── README.md
├── requirements.txt       dependências Python
├── notebooks/             notebooks Jupyter (Python), 01 a 05b
├── Python/                módulos Python usados pelos notebooks
├── R/                     scripts R (00 a 05, run_all.R) e módulos em R/src/
├── outputs_Python/        resultados do Python (figuras, JSON do split e do plano de modelos)
├── outputs_R/             resultados do R (figuras, tabelas, recipe)
└── Relatorio/             relatório final (PDF)
```

## Dados

Os ficheiros são grandes e não estão no repositório. Descarregar do Kaggle
([Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)):

* `accepted_2007_to_2018Q4.csv`
* `rejected_2007_to_2018Q4.csv`

O Python lê-os de `data/raw/` e o R lê-os da raiz do projeto.

## Como correr

### Python

Na pasta do projeto:

```bash
pip install -r requirements.txt
jupyter lab
```

Correr os notebooks em `notebooks/` por ordem: 01 → 02 → 03 → 04 → 05 → 05b.

### R

No RStudio, com o diretório de trabalho na raiz do projeto:

```r
source("R/00_install_packages.R")   # só da primeira vez
source("R/run_all.R")               # corre os scripts 01 a 05 por ordem
```

O script 02 tem de correr antes do 03, 04 e 05. Mais detalhes em [`R/README.md`](R/README.md).

## Fases do trabalho

| # | Fase | Python | R |
|---|------|--------|---|
| 01 | Formulação do problema e split temporal | `01_problem_formulation.ipynb` | `01_problem_formulation.R` |
| 02 | Limpeza dos dados | `02_data_cleaning.ipynb` | `02_data_cleaning.R` |
| 03 | Análise exploratória (EDA, PCA, UMAP) | `03_eda.ipynb` | `03_eda.R` |
| 04 | Testes de hipóteses | `04_hypothesis_testing.ipynb` | `04_hypothesis_testing.R` |
| 05 | Feature engineering e plano de modelos | `05_feature_engineering_model_plan.ipynb`, `05b_feature_validation.ipynb` | `05_feature_engineering_model_plan.R` |

# Project_Ciencias_Dados

Projeto no âmbito da cadeira de Ciências de Dados — Lending Club.

Part 1 notebooks are in `notebooks/01_*` through `05b_*`. Reusable object-oriented Python implementations are in `Python/`.

## Part 2: modelling setup

Run `notebooks/06_modelling_setup.ipynb` from the repository root or the notebooks directory using the project Python environment. It requires the feature-engineered parquet files produced by Notebook 05, plus `outputs/predictor_plan.json`, `outputs/split_definition.json`, and `outputs/split_ids.json`.

Notebook 06 validates the development cohorts and eligible inputs, defines expanding calendar-month CV, checks preprocessing on bounded training/validation samples, and evaluates dummy baselines on all development evaluation rows. It reads no held-out test outcomes. Tables, configuration, plots, and generated interpretation are saved in `outputs/modelling_setup/`.

The baseline models intentionally ignore predictors. Later modelling notebooks must build fresh preprocessing-plus-model pipelines within each CV fold. The month-based CV utility accepts origination dates through `groups`. Chronological evaluation uses retrospective labels and does not establish historical label availability.

Run focused checks with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Execute the notebook with JupyterLab, or programmatically with `nbclient`. Install dependencies from `requirements.txt`; explicit notebook-execution dependencies are included.

## Part 2: custom kNN

Run `notebooks/07_knn_from_scratch.ipynb` after Notebook 06. `Python/knn.py` implements exact batched Euclidean neighbour search, classification voting/probabilities, and regression averaging in NumPy. `Python/knn_experiments.py` handles fold-local preprocessing, reproducible training sampling, CV selection, full validation scoring, and model persistence.

The declared budget is 10,000 training rows per fold, six k/weight settings, five chronological folds, and two BLAS threads. All evaluation rows are retained. Classification sampling approximately preserves prevalence while ensuring every available training class is represented. Full training and sample-matched dummy comparisons are both included. Test rows remain reserved for final evaluation.

Outputs are in `outputs/knn/`: experiment configuration, sampled training positions, CV/tuning tables, validation predictions, plots, interpretation, and saved preprocessing/model bundles. Bundles are development artifacts fitted on training samples, not final deployment models. `joblib` and `threadpoolctl` are used directly for persistence and BLAS resource control. The notebook can take several minutes; exact search scales linearly with training rows, evaluation rows, and encoded features.

## Part 2: non-ensemble supervised models

Run `notebooks/08_supervised_models.ipynb` after Notebooks 06 and 07. The reusable classes in `Python/supervised_models.py` define experiment settings, construct Ridge/logistic regression and individual decision trees, and run fold-aware training, CV selection, validation scoring, and persistence. No additional packages beyond `requirements.txt` are needed.

Notebook 08 retains every training and validation row. It compares nine regression and twelve classification configurations across the same five chronological folds used previously. Classification compares ordinary and balanced class weights. All learned preprocessing and class weights use the current training partition only. Candidates must converge in all CV folds to be eligible for selection. The selected parameters are saved before external validation scoring; saved-model prediction and probability checks run automatically.

Outputs in `outputs/supervised/` include configuration, candidate and tuning tables, per-fold/full-validation metrics and timing, predictions, coefficients/importances, confusion matrices, plots, analysis, and four development model bundles. Comparisons with kNN verify identical validation targets and order, but kNN used 10,000 training rows while these models use the full training cohorts. Final test rows remain unread. Full-cohort CV may take several minutes; runtime depends on the machine and optimizer convergence.

Notebook 08 also audits an observed Ridge extrapolation failure using `Python/regression_diagnostics.py`. It traces large errors to the primary-income loan-to-income ratio on joint applications and saves diagnostic rows, coefficient contributions, and train/validation feature-support summaries. Joint income is used only for this audit. The original fitted models, evaluation rows, and scores are retained; the notebook documents future feature-policy work rather than silently repairing validation results.

## Part 2: ensemble models

Run `notebooks/09_ensemble_models.ipynb` after Notebook 08. `Python/ensemble_models.py` provides bagging (random forests) and histogram gradient boosting for both tasks, reusing the full-cohort, fold-aware runner and save/reload checks from Notebook 08. No additional packages are required beyond `requirements.txt`.

The declared search has six regression and eight classification configurations across five chronological folds (70 CV fits), plus four selected full-training fits. Forests use 64 trees, a 50%-size bootstrap draw per tree, and two workers. Boosting uses 100 iterations with random internal early stopping disabled. All evaluation rows are retained; class weights and preprocessing are fitted within training folds. The fixed budget is checked explicitly and does not establish optimum convergence.

`outputs/ensembles/` contains tuning/metric tables, comparisons with Notebooks 06–08, predictions, confusion matrices, plots, forest importances, boosting structure summaries, application-type diagnostics, analysis, and four development model bundles. Cohort targets and order are verified against previous predictions. The joint-applicant feature policy is deliberately unchanged; its effects are documented rather than repaired using validation results. Final test data remain unread.

## Part 2: imbalance and treatment diagnostics

Run `notebooks/09b_imbalance_diagnostics.ipynb` after Notebook 09. The POO implementation in `Python/imbalance_diagnostics.py` compares unchanged fitting, balanced class weights, partial random undersampling, partial oversampling, and a hybrid. Random policies use three seeds. Training-fold preprocessing is held constant across samplers; uniform fitting weights sum to the original fold size. All minority originals are retained, replication is capped, and evaluation rows are never sampled. No new packages or synthetic-data dependencies are required.

The staged diagnostic selects sampling by training CV, then compares current features, removal of engineered columns, joint-income affordability, and joint affordability with training-fitted clipping/log transforms. It selects an At risk decision threshold using training out-of-fold predictions, reports per-class average precision and a fixed 1% review budget, and confirms the selected combined treatment with fixed-parameter boosting. An unchanged boosting reference is reused transparently. This is a staged ablation, not an exhaustive interaction study.

Outputs in `outputs/imbalance_diagnostics/` contain the declared configuration, CV predictions/summaries, sampling provenance, seed variability, development validation comparisons, confusions, plots, selected model bundles, and analysis. Joint income is an explicitly declared auxiliary origination-time input; absent/non-applicable values are not replaced with zero. The existing predictor plan and earlier datasets/results remain unchanged. Previous validation findings motivated the design, so results are development comparisons; the final test set remains unread.

## Part 2: target/predictor audit and class-specific review

Run `notebooks/09c_target_predictor_audit.ipynb` after 09b. `Python/target_predictor_audit.py` audits development statuses by vintage and loan term, Current/unknown exclusions, event-date metadata, hypothetical scheduled-term scenarios, predictor missingness, joint non-applicability, and every available source column. It does not infer an observation cutoff from payment maxima or change the target/cohorts.

The controlled experiment compares current inputs, adding original grade/subgrade for the after-pricing classifier, and a conditional extension with three original-bureau counts. The issuer prospectus supports the concepts at underwriting, but cannot certify every historical record's timestamp; the bureau extension remains conditional and other fields remain deferred. Grades remain forbidden for pre-pricing regression. Fixed balanced logistic and histogram boosting models use the same five chronological folds (30 CV fits and six final training fits).

Model/scenario choices maximize training-CV average precision separately for At risk and Default review. Illustrative precision floors must be met with at least 20 alerts in every CV fold or are reported infeasible. Validation evaluates those fixed rules and 1%/5% review budgets without replacing the three-class prediction task. Outputs, saved models, input/availability manifests, PR curves, operating tables, and analysis are in `outputs/target_predictor_audit/`. No final test rows are read and earlier results/plans are retained.

The general leakage guard now explicitly includes `last_pymnt_amnt`, `deferral_term`, `payment_plan_start_date`, and `orig_projected_additional_accrued_interest`. Previous modelling allowlists already excluded these, so the fix does not change their inputs or scores.

## Part 2: deep learning

Run `notebooks/10_deep_learning.ipynb` after the diagnostic notebooks. Install the updated `requirements.txt` (PyTorch is required). `Python/deep_learning.py` provides the tabular MLP, training-only numeric conditioning, calendar early stopping, trainer, and reloadable model bundle; `Python/deep_learning_experiments.py` runs chronological CV and three final training seeds per task.

The original splits, three classes, and predictor allowlists are retained. Numerical conditioning uses training-fitted 0.5–99.5% clipping, declared nonnegative log transforms, and the existing imputation/scaling/encoding pipeline. No grade or joint-income extension is introduced. Classification retains every available fitting row, including all 611 At risk examples in final refits. Regression fitting is capped at a seeded 50,000 rows; all stopping and evaluation rows are retained.

The latest month inside each fitting partition selects the epoch count. Fresh preprocessing and a fresh network then refit that whole partition for the selected count; outer validation never selects epochs. Five original calendar folds compare two configurations per task, followed by seeds 42/43/44. The declared CPU budget is 60 epochs, patience 8, two threads, and a 128–64 ReLU/dropout architecture. Training histories, epoch/support tables, bounds, seed variability, comparisons, predictions, plots, and analysis are saved under `outputs/deep_learning/`.

Model directories contain `model.pt` (state dict and target statistics), fitted preprocessing, and metadata. `NeuralModelBundle.load(directory).predict_frame(data)` performs inference; automatic save/reload checks run during the notebook. Artifacts are development models, not final test-evaluated models. The original test set remains unread, and snapshot/follow-up limitations remain applicable.
