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
