# O que dizer ao professor sobre os teus outputs (R)

*Baseado no teu `outputs.zip` de 12/10: 30 tabelas, 16 figuras e 1 recipe.*

**A ideia geral, numa frase:**
> "Os outputs estão organizados pela ordem do pipeline: carregar e limpar os dados, explorar, testar hipóteses e preparar as features e o plano de modelos. Cada tabela e figura é gerada pelos scripts, com seed fixa, por isso são reproduzíveis."

Para cada output: **o que é** → **o que dizer**, com os números do teu run.

---

## 1. Dados e limpeza (scripts 01 e 02)

**`schema_accepted.csv` / `schema_rejected.csv`:** a estrutura dos dois ficheiros.
> "Os aceites têm 151 colunas; os rejeitados só 9. Por isso as comparações aceites vs rejeitados só podem usar o montante, o DTI, o score e o estado."

**`task_definitions.csv`:** os 3 problemas.
> "Regressão: prever a `int_rate`. Classificação: 3 classes (Good, At risk, Default). Clustering: grupos de clientes e de estados."

**`loan_status_mapping.csv`:** como cada estado do empréstimo virou uma classe.
> "O Fully Paid é Good (1 076 751). O Charged Off e o Default são Default (268 599). O Late e o In Grace Period são At risk (34 252). Os 878 317 Current foram excluídos, como pede o enunciado, porque ainda não se sabe como vão acabar. Também excluí os 2 749 'Does not meet the credit policy', que são de uma política antiga."

**`exclusions_log.csv`:** quantas linhas saíram em cada passo.
> "Depois da limpeza ficam 2 260 668 empréstimos aceites. Retirei 33 linhas de rodapé que não são empréstimos. A classificação fica com 1 379 602."

**`exclusions_columns.csv` / `column_eligibility.csv`:** que colunas são proibidas e porquê (leakage).
> "Tirei todas as colunas do pós-empréstimo (pagamentos, recuperações, último FICO...). Na regressão tirei também o `grade`, o `sub_grade` e o `installment`, porque a taxa é praticamente calculada a partir deles. A tabela de elegibilidade diz, coluna a coluna, em que tarefa pode ser usada."

**`duplicates.csv`:**
> "Nos aceites não há duplicados nem ids repetidos. Nos rejeitados havia 1 339 274 linhas exatamente repetidas (quase 5 %), e removi-as."

**`quality_checks.csv`:** valores estranhos encontrados.
> "137 DTI com valores-código (−1 ou 999) passaram a NA. Há 1 667 rendimentos iguais a 0 e 7 343 utilizações acima de 100 %, que são possíveis e por isso ficaram. Nos rejeitados, 86 754 scores iguais a 0 passaram a NA, e há DTIs absurdos (ex.: 4042 %)."

**`missingness_by_year.csv` + figura `missingness_by_year.png`:**
> "Os campos do bureau (ex.: `tot_cur_bal`) estão 100 % vazios até 2011, e metade em 2012. O Lending Club só os começou a recolher depois, por isso isto não é erro. É por isso que a recipe junta um indicador de NA."

**`joint_missingness_by_application_type.csv`:**
> "Os campos de pedidos conjuntos estão 100 % vazios nos pedidos individuais. Isso quer dizer 'não se aplica', por isso não os preenchi com 0."

**`split_definition.json` + `split_sizes.csv`:** o split temporal.
> "Treino até 2014, validação de 2015 a 2016 e teste a partir de 2017. Na regressão ficam 466 345 / 855 502 / 938 821 empréstimos; na classificação, 451 671 / 678 142 / 249 789. É temporal porque o modelo vai prever empréstimos futuros."

**`followup_by_vintage.csv` + figura `status_by_issue_year.png`:**
> "Esta tabela mostra a taxa de Default por ano e prazo. Nos 36 meses, desceu de 17,9 % (2007) para 9,9 % (2010) e voltou a subir até 19,5 % (2016). Nos 60 meses é cerca do dobro (21–36 %). Mostra também que, a partir de 2016 (36 meses) e de 2014 (60 meses), nenhum empréstimo tinha chegado ao fim do prazo. É por isso que a classe At risk cresce tanto no teste."

---

## 2. Exploração (script 03)

**`eda_describe_regression.csv` + `histograms_numeric.png`:**
> "As variáveis de dinheiro são muito assimétricas: o `annual_inc`, o `revol_bal` e o `pub_rec` são as piores. Por isso usei o log do rendimento e testes não paramétricos. Nos histogramas, as variáveis de dinheiro aparecem em escala log para se ver a forma."

**`int_rate_by_year.png`:**
> "A taxa de juro muda com os anos (política de preços). É mais uma razão para o split ser temporal."

**`int_rate_vs_fico.png` / `int_rate_vs_dti.png`:**
> "FICO mais alto dá taxa mais baixa, e é a relação mais forte. Com o DTI a relação é positiva mas mais fraca."

**`int_rate_by_purpose_term.png`:**
> "Os empréstimos a 60 meses têm sempre taxa mais alta. A finalidade também influencia a taxa (é o que a H7 testa)."

**`eda_class_balance.csv` + `class_balance.png`:**
> "No treino + validação há 79,6 % Good, 19,5 % Default e só 0,89 % At risk (10 102 empréstimos). Os números são iguais aos do Ramos em Python. As classes estão muito desequilibradas, por isso vamos usar o macro-F1 e pesos de classe."

**`default_by_fico_term.png`:**
> "A percentagem de Default desce à medida que o FICO sobe, e é sempre maior nos 60 meses."

**`corr_spearman_regression.csv` + `corr_heatmap.png`:**
> "Usei Spearman porque os dados são assimétricos e têm outliers. As variáveis mais correlacionadas com a taxa são o FICO (−0,41) e o prazo (+0,40)."

**`pca_explained_variance.csv` + `pca_scree.png` + `pca_loadings.csv` + `pca_pc1_pc2.png`:**
> "São precisas 10 componentes para explicar 80 % da variância, por isso não há uma estrutura simples. A PC1 é o 'tamanho do crédito' (limite total, saldo total, rendimento). A PC2 é 'utilização vs FICO' (utilização alta e FICO baixo). O `int_rate` não entrou no PCA, porque é o alvo."

**`umap_classes.png`:**
> "No UMAP, as três classes aparecem misturadas: não há separação fácil. A classificação vai ser difícil, sobretudo para a At risk."

**`eda_iqr_outliers.csv`:**
> "O `delinq_2yrs` (20 %) e o `pub_rec` (17 %) aparecem com muitos outliers porque a mediana é 0, e qualquer valor acima de 0 conta. São valores reais, por isso não os removi."

**`amount_accepted_vs_rejected.png`:**
> "Os aceites pedem montantes maiores do que os rejeitados. O aviso 'Removed 666 rows' é só porque o eixo do gráfico pára nos 40 000 $."

**`rejected_risk_score.png` + `acceptance_by_score_band.csv` + `acceptance_by_score_band.png`:**
> "Este é um dos resultados mais claros. Abaixo de 650 de score, praticamente ninguém é aceite (0 aceites até 625, 227 entre 625 e 650). A partir de 650, a taxa de aceitação salta para 31–46 %. Isto sugere um corte mínimo de score na política do Lending Club."

**`eda_insights.csv`:** o resumo automático da EDA.
> "Junta as 5 conclusões principais: as variáveis mais assimétricas, as mais correlacionadas com a taxa, as 10 PCs, os 0,89 % de At risk, e os anos em que o bureau estava vazio (2007–2012)."

---

## 3. Testes de hipóteses (script 04)

**`hypothesis_normality.csv`:**
> "Testei a normalidade em 6 variáveis com o Shapiro-Wilk (5000 valores cada). Todas rejeitam a normalidade (p < 0,001), e a pior é o rendimento (W = 0,75). Por isso usei testes não paramétricos."

**`hypothesis_results.csv`:** os 7 testes.
> "Com amostras de 200 mil, tudo dá significativo, por isso olho para o tamanho do efeito. Corrigi os p-valores com o método de Holm. Os resultados:"

| Teste | Efeito | Leitura |
|---|---|---|
| H1 DTI: Default vs Good | δ = 0,16 | pequeno |
| H2 finalidade × classe | V = 0,04 | desprezável |
| **H3 score: rejeitados vs aceites** | **δ = −0,62** | **grande**: o mais forte |
| H4 montante: aceites vs rejeitados | δ = 0,20 | pequeno (os aceites pedem mais) |
| H5 FICO: Default vs Good | δ = −0,19 | pequeno |
| H6 utilização: Default vs Good | δ = 0,08 | desprezável |
| H7 taxa por finalidade | ε² = 0,04 | pequeno |

> "Conclusão: entre os empréstimos aprovados não há nenhuma variável com efeito grande. O sinal está espalhado. O único efeito grande é o score na decisão de aceitar ou rejeitar, o que bate com o corte de score que vimos na EDA."

**`hypothesis_purpose_vs_class_table.csv`:**
> "É a tabela de contingência da H2. No global a associação é fraca, mas há finalidades que se destacam: o `small_business` tem 28,3 % de Default, contra 14,4 % no `car`."

---

## 4. Features e plano de modelos (script 05)

**`feature_catalog.csv`:** as 19 features consideradas, com fórmula, razão e elegibilidade.
> "Criei 19 candidatas e documentei cada uma. Com o Ramos, escolhemos as 10 com maior correlação com a taxa (colunas `selected_regression` / `selected_classification`)."

**`feature_check_train.csv`:** a validação das features no treino.
> "A melhor é o `fico_mid` (ρ = −0,49; os Default têm cerca de 9,6 pontos FICO a menos). O `term_60` tem ρ = 0,44 mas não entrou, porque é só o `term` recodificado. O `credit_hist_months` é só para a classificação, porque usa a data de financiamento. Os Default têm cerca de 6,6 meses a menos de histórico."

**`recipe_regression_train.rds`** (pasta `models/`):
> "É a recipe do pré-processamento (imputação, indicador de NA, normalização, dummies), estimada só com o treino. Fica guardada para aplicar igual à validação e ao teste, sem leakage."

**`cv_folds.csv`:**
> "A validação cruzada é temporal: treina nos anos anteriores e avalia no ano seguinte (4 folds, 2011 a 2014). Na classificação só entram empréstimos já terminados, por isso o 1.º fold só tem 422. É uma limitação que assumo."

**`model_plan.csv`:**
> "O plano para a próxima fase: baseline, kNN feito de raiz, Ridge/Logística, árvore, Random Forest, Gradient Boosting, MLP e, no clustering, k-means/GMM/DBSCAN."

---

## Se o professor perguntar "o que é que correu menos bem?"

> "Três coisas:
> - a classe At risk é minúscula no treino (0,14 %) e cresce muito no teste (9,67 %);
> - o score dos rejeitados só existe em 33 % dos pedidos;
> - o meu R e o Python do Ramos ainda não estão 100 % alinhados (limpeza do dti, features, H7). Isso fica para a próxima entrega."
