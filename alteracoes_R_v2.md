# Alterações ao código R (revisão de 8 de outubro)

Revi o `src.zip` (R: `R/`, Python: `src/` e `notebooks/`). Nenhum dos dois códigos tem ainda as 10 features: o R continua com as 17 em `R/src/feature_engineering.R` e o Python com as 15 em `src/feature_engineering.py`. Também a normalidade e o `dti` ainda estão como antes. Abaixo está o que mudar no **teu R**, por ordem, e no fim o que o Ramos tem de mudar para os dois baterem certo.

O teu R já estava certo em várias coisas, que não precisas de mexer: o `dti` -1/999 → NA (`R/src/preprocessing.R:54`), o PCA sem `int_rate` (`R/03_eda.R:160`), a H7 e o Holm (`R/04_hypothesis_testing.R:174-196`), e a seed 42 com o limite de 200 mil.

---

## 1. As 10 features

### O problema que encontrei: `credit_hist_months` na regressão
Usa o `issue_d` (a data em que o empréstimo foi financiado). O teu próprio código diz que isso não pode entrar na regressão, porque é uma data futura em relação ao momento do pricing (`feature_engineering.R:140` e o `column_eligibility`). O Python marca-a como elegível para a regressão e admite isso como limitação.

**Proposta:** seguir a tua regra. Na regressão, o 7.º lugar passa para o `open_acc_ratio` (ρ = 0,077, praticamente empatado). Na classificação o `issue_d` já é permitido, por isso o `credit_hist_months` fica. O Ramos tem de fazer o mesmo.

| # | Regressão | Classificação |
|---|---|---|
| 1 | fico_mid | fico_mid |
| 2 | dti_x_loan_to_income | dti_x_loan_to_income |
| 3 | loan_to_income | loan_to_income |
| 4 | inq_per_open_acc | inq_per_open_acc |
| 5 | total_bal_to_limit | total_bal_to_limit |
| 6 | any_delinq_2yrs | any_delinq_2yrs |
| 7 | **open_acc_ratio** | **credit_hist_months** |
| 8 | log_annual_inc | log_annual_inc |
| 9 | emp_length_years | emp_length_years |
| 10 | emp_length_missing | emp_length_missing |

### 1a. `R/src/feature_engineering.R`: acrescentar as 2 features do Ramos
Dentro do `dt[, \`:=\`( ... )]` do `add_features()`, a seguir ao F17 (linha 69), põe uma vírgula no fim da linha do `issue_year` e acrescenta:

```r
    # F18 (Ramos) interação entre a dívida atual e o peso do novo empréstimo
    dti_x_loan_to_income = dti * safe_div(loan_amnt, annual_inc),
    # F19 (Ramos) [classificação apenas] idade do histórico de crédito em meses
    credit_hist_months   = as.numeric(issue_d - earliest_cr_line) / 30.44
```

- **`dti_x_loan_to_income`:** calculo o `loan_to_income` outra vez em vez de usar a coluna, porque dentro do mesmo `:=` a coluna ainda não existe.
- **`credit_hist_months`:** o `/ 30.44` é a mesma conversão de dias para meses que o Python usa (`(ref - earliest).dt.days / 30.44`). Assim os valores batem com os do Ramos. O teu `months_between()` dava meses inteiros e os números saíam ligeiramente diferentes.
- **O `dti` já vem limpo** do script 02, por isso o `dti_x_loan_to_income` não apanha os 999.

### 1b. Mesmo ficheiro: as listas finais
No fim do ficheiro, acrescenta:

```r
# Conjunto final acordado com o Ramos (10 por tarefa, ordenadas por |Spearman| com int_rate)
FEATURES_REG <- c("fico_mid", "dti_x_loan_to_income", "loan_to_income",
                  "inq_per_open_acc", "total_bal_to_limit", "any_delinq_2yrs",
                  "open_acc_ratio", "log_annual_inc", "emp_length_years",
                  "emp_length_missing")
FEATURES_CLF <- c(setdiff(FEATURES_REG, "open_acc_ratio"), "credit_hist_months")
```

Com isto, a Fase 4 usa só estas listas, e se mudarem de ideias só mexes num sítio.

### 1c. Mesmo ficheiro: o catálogo
O `feature_catalog()` documenta as 17 por posição (`rep("yes", 14)`, …), por isso acrescentar linhas no meio estraga a correspondência. Deixa a função como está e, no fim dela, troca o `)` final por um `rbind` com as duas novas. Ou seja, substitui as linhas 143-145:

```r
    classification_eligible = rep("yes", 17)
  )
}
```

por:

```r
    classification_eligible = rep("yes", 17)
  )
  novas <- data.table(
    feature = c("dti_x_loan_to_income", "credit_hist_months"),
    source_columns = c("dti, loan_amnt, annual_inc", "earliest_cr_line, issue_d"),
    formula = c("dti * loan_amnt / annual_inc", "(issue_d - earliest_cr_line) em dias / 30.44"),
    rationale = c("dívida existente x peso do novo empréstimo (alavancagem composta)",
                  "histórico de crédito mais longo = mais estabilidade"),
    missing_handling = c("NA se dti/income NA ou income 0 -> median impute (train fit)",
                         "NA se earliest_cr_line NA -> median impute (train fit)"),
    regression_eligible = c("yes", "no (uses issue_d, a future funding date)"),
    classification_eligible = c("yes", "yes"))
  out <- rbind(out, novas)
  out[, selected_regression := feature %in% FEATURES_REG]
  out[, selected_classification := feature %in% FEATURES_CLF]
  out[]
}
```

e muda a primeira linha da função (linha 76) de `data.table(` para `out <- data.table(`.

Porquê: o catálogo continua a mostrar as 19 (o enunciado pede pelo menos 10 documentadas e é bom mostrar o que foi considerado), mas as duas colunas novas dizem quais foram escolhidas. Isso também justifica a escolha no relatório.

### 1d. `R/05_feature_engineering_model_plan.R`: o check com as novas
Não precisas de mexer: o `fcheck` (linhas 267-277) percorre o `fcat$feature`, por isso as 2 novas entram sozinhas. O `credit_hist_months` vai dar NA na coluna do Spearman, o que está certo, porque não é elegível para a regressão.

Para o check refletir a decisão, acrescenta a seguir ao `save_table(fcheck, ...)` (linha 278):

```r
fcheck[feature %in% FEATURES_REG][order(-abs(spearman_int_rate))]
```

Isto imprime as 10 da regressão ordenadas, para conferires que a ordem bate com a tabela acima.

### 1e. `R/README.md`, linha 22
Troca `17 features + documentation table` por `19 features documentadas, 10 selecionadas por tarefa (FEATURES_REG / FEATURES_CLF)`.

---

## 2. Normalidade: `R/04_hypothesis_testing.R`, linhas 64-71

Substitui o bloco por:

```r
norm_vars <- c("dti", "annual_inc", "fico_range_low", "loan_amnt", "revol_util", "int_rate")
norm_tbl <- rbindlist(lapply(norm_vars, function(v) {
  x <- cap_sample(eda_reg[[v]], 5000, SEED)
  st <- shapiro.test(x)
  data.table(column = v, shapiro_stat = unname(st$statistic), shapiro_p = st$p.value,
             skew = skewness(x), n_used = length(x))
}))
save_table(norm_tbl, "hypothesis_normality")
norm_tbl
```

O que mudou e porquê:
- **`norm_vars`:** as 6 variáveis acordadas (as tuas 5 mais o `loan_amnt` do Ramos).
- **`column` / `shapiro_stat`:** os mesmos nomes de coluna do CSV dele.
- **`n_used`:** regista quantos valores entraram no teste. O `cap_sample` já tira os NA, por isso pode ser < 5000 numa variável com muitos NA.
- **`skewness()`:** mantive a tua função de `eda_utils.R`. Divide por `sd()` (n − 1) e o scipy divide pelo desvio-padrão com n, mas com n = 5000 a diferença é de 0,03 %, irrelevante.

Uma coisa a combinar com o Ramos: tu fazes a normalidade no **cohort da regressão** (`eda_reg`) e ele no **da classificação** (`clf_eda`). Como o `int_rate` é o alvo da regressão, eu ficava com o teu e pedia-lhe para mudar.

---

## 3. PCA com `fico_mid` (opcional): `R/03_eda.R`, linhas 158-160

```r
PCA_VARS <- c(setdiff(EDA_NUM, c("fico_range_low", "fico_range_high")), "fico_mid")
pca_base <- reg_cohort[split == "train", c("int_rate", EDA_NUM), with = FALSE]
pca_base[, fico_mid := (fico_range_low + fico_range_high) / 2]
pca_df <- sample_rows(na.omit(pca_base), params$pca_sample_n, SEED)
pca <- prcomp(pca_df[, ..PCA_VARS], center = TRUE, scale. = TRUE)
```

Tira a componente com variância ≈ 0 (a tua antiga PC22). O scree, os loadings e o gráfico PC1-PC2 logo abaixo continuam a funcionar sem mudanças: usam o `pca` e o `pca_df`, que continua a ter o `int_rate` para colorir. O UMAP (linha 186) usa o `EDA_NUM` à parte; no UMAP não faz mal ficar com os dois FICO. Se preferires deixar como está, explica no relatório que a última PC é ≈ 0 por causa do par low/high.

---

## O que o Ramos tem de mudar (para os dois baterem certo)

1. **Features:** passar para as mesmas 10 por tarefa. Tem de criar as tuas 7 (`fico_mid`, `inq_per_open_acc`, `total_bal_to_limit`, `any_delinq_2yrs`, `open_acc_ratio`, `log_annual_inc`, `emp_length_missing`) e marcar o `credit_hist_months` como não elegível para a regressão.
2. **`dti`:** no `02_data_cleaning`, pôr o `dti < 0` ou `>= 999` do accepted a NaN. Neste momento só conta quantos são (no `03_eda`) e não os limpa. Isto também afeta o `dti_x_loan_to_income`.
3. **Normalidade:** as mesmas 6 variáveis e no cohort da regressão.
4. **H7:** acrescentar o Kruskal-Wallis do `int_rate` por `purpose` e a correção de Holm.

O PCA dele já está certo nesta versão: exclui o `int_rate` e tem um `assert` a confirmar.
