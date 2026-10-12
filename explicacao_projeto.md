# O projeto explicado do zero

*Para leres antes de entregar e para conseguires responder ao professor.*

---

## 1. De que trata o projeto

O **Lending Club** é uma plataforma americana que emprestava dinheiro a particulares. Temos dois ficheiros, de 2007 a 2018:

- **Aceites** (cerca de 2,26 milhões de linhas): empréstimos que foram dados. Cada linha tem 151 colunas: quanto pediu, a taxa de juro, o rendimento, o FICO (a "nota de crédito" da pessoa), as dívidas e, no fim, se pagou ou não.
- **Rejeitados** (cerca de 27,6 milhões de linhas): pedidos recusados. Só têm 9 colunas: montante, data, estado, DTI e score de risco.

O enunciado pede três problemas:

| Problema | Pergunta | Tipo |
|---|---|---|
| **Regressão** | Que taxa de juro (`int_rate`) vai ser dada a esta pessoa? | Prever um número |
| **Classificação** | Este empréstimo vai acabar bem, em risco ou em incumprimento? | Prever uma de 3 classes |
| **Clustering** | Há grupos naturais de clientes (ou de estados)? | Encontrar grupos, sem resposta certa |

As 3 classes da classificação:
- **Good:** pagou tudo (*Fully Paid*).
- **At risk:** está atrasado (*Late*, *In Grace Period*).
- **Default:** não pagou (*Charged Off*, *Default*).

Os empréstimos **Current** (ainda a decorrer, sem atraso) são **excluídos**, porque ainda não se sabe como vão acabar. Isto é o que o enunciado pede.

**Esta 1.ª entrega** cobre as fases iniciais: perceber os dados, limpá-los, explorá-los, testar hipóteses, criar features e planear os modelos. Os modelos em si ficam para as próximas entregas.

---

## 2. Limpeza dos dados

- **Linhas lixo:** o ficheiro dos aceites tem 33 linhas de rodapé que não são empréstimos. Foram removidas.
- **Duplicados:** nos rejeitados há 1,34 milhões de linhas repetidas (quase 5 %). Foram removidas.
- **Valores impossíveis:** o `dti` (dívida / rendimento) às vezes vale −1 ou 999. São "códigos" de valor em falta, não valores reais. No R passam a NA (vazio).
- **NA não é sempre "falta":** os campos de pedidos conjuntos estão vazios nos pedidos individuais porque "não se aplica". Por isso não se preenchem com 0.

---

## 3. Data leakage (a parte mais importante a defender)

**Leakage** é usar no modelo informação que, na vida real, ainda não existiria no momento da previsão. O modelo parece ótimo nos testes, mas na prática não funciona.

O que se excluiu:
- **Colunas do futuro:** pagamentos feitos, recuperações, último pagamento, etc. Só se conhecem depois do empréstimo.
- **Na regressão,** também o `grade`/`sub_grade` e o `installment`. O Lending Club calcula a taxa diretamente a partir do grade, e o `installment` é calculado a partir da taxa. Usá-los seria "fazer batota".
- **Features derivadas destas** também são proibidas.

E mais duas regras:
- **Split temporal, não aleatório:**
  - Treino: até 2014.
  - Validação: 2015–2016.
  - Teste: a partir de 2017.

  Na vida real o modelo prevê o futuro, por isso testamos em dados mais recentes do que os de treino.
- **Pré-processamento ajustado só no treino:** a imputação de NA, a normalização, etc. são calculadas só com o treino e depois aplicadas ao resto. No R isto faz-se com as `recipes` (`prep()` no treino, `bake()` no resto).

---

## 4. Exploração (EDA)

- **Distribuições:** as variáveis financeiras são muito assimétricas, com poucas pessoas com valores enormes (ex.: rendimento). Por isso usou-se o log do rendimento e testes não paramétricos.
- **Correlações com a taxa de juro:** a mais forte é o FICO (negativa: melhor nota, juros mais baixos), seguida do prazo (60 meses têm juros mais altos).
- **PCA:** são precisas cerca de 10 componentes para explicar 80 % da variância. Ou seja, não há um "padrão dominante" nos dados.
- **Desequilíbrio de classes:** a *At risk* é muito rara no treino (0,14 %) mas chega a 9,67 % no teste. Isto acontece porque os empréstimos de 2017–2018 ainda estavam a decorrer quando os dados foram recolhidos.

---

## 5. Testes de hipóteses

**A ideia:** formular uma pergunta ("os que não pagam têm mais dívidas?"), escolher o teste certo e ver se a diferença é real.

**Porque testes não paramétricos (Mann-Whitney, Kruskal-Wallis, qui-quadrado):** o teste de Shapiro-Wilk mostrou que nenhuma variável segue a distribuição normal, por isso o teste t não serve.

**Porque o p-valor não chega:** com milhões de linhas, qualquer diferença minúscula dá "significativo" (p < 0,001). Por isso o que interessa é o **tamanho do efeito**:
- Cliff's δ para comparar 2 grupos.
- V de Cramér para tabelas.
- ε² para mais de 2 grupos.

**Holm:** como fizemos vários testes, corrigimos os p-valores para não aparecerem "descobertas" por acaso. Continuaram todos significativos.

| Hipótese | Resultado em linguagem simples |
|---|---|
| H1 DTI vs Default | Quem não paga tem um pouco mais de dívida. Efeito pequeno. |
| H2 Finalidade vs classe | Quase nenhuma relação no global (desprezável). |
| H3a Score por estado (Ramos) | Diferença desprezável entre estados. |
| H3b Score: rejeitados vs aceites (R) | **O único efeito grande:** os rejeitados têm score muito mais baixo (641 vs 692). |
| H4 Montante: aceites vs rejeitados | Os aceites pedem **mais** (pessoas com melhor perfil pedem mais e são aceites). Efeito pequeno. |
| H5 FICO vs Default | Quem não paga tem FICO cerca de 10 pontos mais baixo. Efeito pequeno. |
| H6 Utilização do crédito vs Default | Diferença desprezável. |
| H7 Taxa de juro por finalidade (R) | A finalidade explica cerca de 4 % da taxa. Efeito pequeno. |

**Conclusão:** dentro dos empréstimos aprovados não há uma variável "mágica". O sinal está espalhado por muitas variáveis fracas. Isto justifica modelos que combinam muitas variáveis, como árvores, ensembles e redes neuronais.

**Cuidado:** são associações, não causas.

---

## 6. Features (variáveis novas)

Criámos variáveis novas a partir das existentes e ficámos com as **10 com maior correlação** com a taxa de juro. Foi uma decisão conjunta com o Ramos.

| Feature | O que é |
|---|---|
| `fico_mid` | Média do intervalo FICO. É a mais forte (ρ = 0,49). |
| `dti_x_loan_to_income` | Dívida atual × peso do novo empréstimo. |
| `loan_to_income` | Empréstimo / rendimento. |
| `inq_per_open_acc` | Pedidos de crédito recentes / contas abertas. |
| `total_bal_to_limit` | Saldo total / limite total. |
| `any_delinq_2yrs` | Teve algum atraso nos últimos 2 anos? (0/1) |
| `open_acc_ratio` *(só regressão)* | Contas abertas / contas totais. |
| `credit_hist_months` *(só classificação)* | Meses de histórico de crédito. |
| `log_annual_inc` | Log do rendimento. |
| `emp_length_years` | Anos de emprego (0 a 10). |
| `emp_length_missing` | Não disse os anos de emprego? (0/1) |

**Porque é que o `credit_hist_months` não entra na regressão:** usa a data de financiamento do empréstimo, que ainda não existe quando se decide a taxa.

---

## 7. Plano de modelos (próximas fases)

Para cada problema:
- **Baseline:** prever sempre a mediana ou a classe mais comum. É a referência a bater.
- **kNN feito de raiz:** pedido pelo enunciado.
- **Modelos simples:** Ridge / Regressão Logística e Árvore de Decisão.
- **Ensembles:** Random Forest (bagging) e Gradient Boosting (boosting).
- **Rede neuronal:** MLP.
- **Clustering:** k-means, GMM e DBSCAN.

**Métricas:**
- Regressão: MAE, RMSE e R².
- Classificação: sobretudo o **macro-F1**. Com classes desequilibradas, a accuracy engana: um modelo que diz sempre "Good" acerta muito e não serve para nada.

---

## 8. Limitações (mostra que pensaste criticamente)

- **Viés de seleção:** só vemos os empréstimos que foram aprovados.
- **Empréstimos recentes ainda a decorrer:** é por isso que a *At risk* cresce no teste.
- **O mundo mudou entre 2007 e 2018:** a crise de 2008 e as regras do Lending Club.
- **Equidade:** o estado e o código postal podem servir de proxy para etnia, e os anos de emprego para idade.
- **R e Python ainda não estão 100 % alinhados nesta entrega:** a limpeza do `dti`, as features e a H7 ficam para a próxima.

---

## 9. Perguntas que o professor pode fazer

**"Porque não usaram um split aleatório?"**
Porque o modelo vai prever empréstimos futuros. Um split aleatório misturava anos e dava resultados otimistas que não se repetem na prática.

**"Todos os testes deram significativos. Isso quer dizer que tudo importa?"**
Não. Com milhões de linhas tudo dá significativo. Por isso olhámos para o tamanho do efeito, e quase todos são pequenos ou desprezáveis.

**"Porque tiraram o `grade` e o `installment` da regressão?"**
Porque a taxa é praticamente calculada a partir deles. O modelo estaria a "copiar" a resposta.

**"Porque é que a *At risk* tem 0,14 % no treino e quase 10 % no teste?"**
Os empréstimos de 2017–2018 ainda estavam a decorrer quando os dados foram recolhidos. Muitos aparecem atrasados porque ainda não tinham acabado. Por isso usamos o macro-F1 e pesos de classe.

**"Porque é que o `term_60` não está nas features, se é a 2.ª mais correlacionada?"**
Porque é só o `term` (36/60 meses) recodificado em 0/1. O `term` original já entra no modelo, por isso não acrescentava nada.

**"Qual foi a tua parte?"**
A implementação em R: a limpeza, a EDA, os testes de hipóteses (incluindo a H3b e a H7), as features e as `recipes`. O Ramos fez a mesma pipeline em Python. Comparámos os resultados e alinhámos as features.
