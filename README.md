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

