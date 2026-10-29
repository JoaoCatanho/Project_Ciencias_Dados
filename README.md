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
