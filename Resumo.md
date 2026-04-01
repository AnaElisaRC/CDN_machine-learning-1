# 💻 Guia Prático de Códigos: Scikit-Learn (Aulas 04, 05 e 06)

## 1. Regressão Linear e Regularização (Aula 04)
Este módulo foca em prever valores contínuos e ajustar o modelo para evitar underfitting e overfitting.

**Comandos Fundamentais:**
* **Regressão Linear (Equação Normal):** Encontra o melhor ajuste matemático exato.
    ```python
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_) # Exibe onde a reta corta o eixo Y e sua inclinação
    ```
* **Regressão Linear com Gradiente Descendente (SGD):** Ideal para datasets gigantes. Atenção à função `.ravel()` que transforma a matriz `y` em um vetor 1D, exigência do `SGDRegressor`.
    ```python
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, eta0=0.01, random_state=42)
    sgd_reg.fit(X, y.ravel())
    ```
* **Regressão Polinomial:** Quando uma reta não resolve, criamos características elevadas ao quadrado, cubo, etc.
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    ```

**⚠️ Pegadinhas Práticas & Regularização:**
* **Identificando Overfitting:** A biblioteca Scikit-Learn possui a função `learning_curve` que permite avaliar visualmente o RMSE de Treino vs. Validação. Se a curva de treino tem erro muito baixo e a de validação muito alto e não convergem, o modelo decorou os dados (Overfitting).
* **Lidando com Overfitting (Regularização):** * **Ridge (L2):** Penaliza coeficientes grandes. `Ridge(alpha=0.1)`. 
    * **Lasso (L1):** Tende a zerar coeficientes inúteis (seleção automática de features). `Lasso(alpha=0.1)`.
    * **Importante:** Um `alpha` muito grande gera *underfitting* (a reta fica muito plana), e um `alpha` próximo de zero não resolve o *overfitting*.

---

## 2. Regressão Logística e Métricas (Aula 05)
O foco aqui não é prever um valor, mas sim a probabilidade de pertencer a uma classe (ex: É fraude ou não?).

**Comandos Fundamentais:**
* **Treinamento do Classificador:** Diferente da Regressão Linear que usa `alpha`, a Logística usa `C`. O parâmetro `C` é o **inverso** da regularização (quanto maior o `C`, *menor* a regularização).
    ```python
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(C=2, random_state=42)
    log_reg.fit(X_train, y_train)
    ```
* **Pegadinha `predict` vs `predict_proba`:** * `log_reg.predict(X_new)` retorna a categoria final (Ex: `True` ou `False`).
    * `log_reg.predict_proba(X_new)` retorna a matriz com as probabilidades exatas (Ex: 15% chance de Falso, 85% chance de Verdadeiro).

**⚠️ Pegadinhas Práticas & Threshold:**
* **Ajuste manual do Threshold (Fronteira de Decisão):** Por padrão, se a probabilidade passar de 50% (0.5), o Scikit-Learn classifica como Positivo. Se a questão pedir para "aumentar o Recall para 82%", você deve ignorar o `.predict()` padrão, acessar as probabilidades reais calculadas pelo modelo com `method="decision_function"` e criar seu próprio limite:
    ```python
    from sklearn.model_selection import cross_val_predict
    # Pegando a pontuação bruta em vez da classe
    y_scores = cross_val_predict(log_reg, X_train, y_train, cv=3, method="decision_function")
    
    # Criando um limite personalizado (ex: 0.3 em vez de 0.0) para priorizar Recall
    threshold = 0.3
    y_amostra_pred = (y_scores > threshold)
    ```

---

## 3. Projeto End-to-End e Pipelines (Aula 06)
Este módulo é sobre como organizar o código de maneira profissional e reprodutível, juntando transformações de dados e o modelo em um único fluxo.

**Comandos Fundamentais de Preparação:**
* **Amostragem Estratificada:** Evita que a separação de treino/teste fique desbalanceada (muito comum em provas).
    ```python
    from sklearn.model_selection import train_test_split
    # stratify força o split a manter a mesma proporção da coluna informada
    train_set, test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    ```
* **Tratando Nulos e Categorias:**
    * Numéricos: `SimpleImputer(strategy="median")` preenche buracos com a mediana.
    * Texto/Categorias: `OneHotEncoder(sparse_output=False)` converte nomes em colunas binárias.

**⚠️ Pegadinhas Práticas & Pipelines:**
* **Orquestração com `ColumnTransformer`:** Aplica transformações numéricas apenas nas colunas de número, e transformações de texto apenas nas colunas de texto, unindo tudo no final.
    ```python
    from sklearn.compose import make_column_transformer
    from sklearn.compose import make_column_selector
    
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object))
    )
    ```
* **Busca pelos Melhores Hiperparâmetros:** O modelo final não é testado na base do "chute".
    * `GridSearchCV`: Testa *todas* as combinações de uma lista que você passar.
    * `RandomizedSearchCV`: Testa combinações aleatórias. É o método recomendado quando a lista de hiperparâmetros é muito vasta, utilizando funções da biblioteca `scipy.stats` (como `randint` e `expon`).
* **Persistência (Salvar o modelo para uso):** A etapa final de um projeto é exportar o modelo treinado com a biblioteca `joblib`.
    ```python
    import joblib
    joblib.dump(final_model, "meu_modelo.pkl") # Salva
    modelo_carregado = joblib.load("meu_modelo.pkl") # Carrega em produção
    ```

---

## 5. Projeto de Aprendizagem de Máquina de Ponta a Ponta (Pipeline Completo)
**Conceitos-Chave:** Em ambientes profissionais, um modelo não existe no vácuo. [cite_start]É necessário criar um fluxo contínuo e reprodutível que englobe a recolha de dados, a análise exploratória, a preparação mecânica dos atributos, a escolha do algoritmo e a otimização final para entrada em produção[cite: 20, 402, 620, 1419, 1502].

** Nível 1: Explicação Simples**
Pensa nisto como a linha de montagem de uma fábrica. Não basta saberes como funciona a máquina que aperta os parafusos (o modelo matemático). Precisas de construir passadeiras rolantes que lavem as peças sujas (limpeza de dados nulos), pintem as peças de cores diferentes (transformação de texto em números) e entreguem tudo formatado para o robô final fazer o trabalho e ser empacotado para venda (colocar em produção).

** Nível 2: Intermediário com Exemplos**
[cite_start]Ao prever o preço de habitações na Califórnia[cite: 4, 22], não injetas os dados brutos no modelo. [cite_start]Primeiro, verificas correlações geográficas (ex: desenhando mapas de dispersão cruzando a latitude/longitude com a população e o valor da casa)[cite: 450]. [cite_start]Depois, geras características novas e mais lógicas, como descobrir a proporção de quartos por habitação em vez de usar os números isolados[cite: 585]. [cite_start]Só depois de os dados estarem limpos e escalonados é que testas candidatos como a Regressão Linear ou a Árvore de Decisão[cite: 1507, 1561].

** Nível 3: Técnica e Profissional**
[cite_start]O cerne da engenharia de Machine Learning moderna é a automação através de **Pipelines**[cite: 1291]. [cite_start]Usa-se a classe `Pipeline` combinada com o `ColumnTransformer` do Scikit-Learn[cite: 1297, 1387]. [cite_start]Isto permite orquestrar paralelamente transformações distintas: por exemplo, aplicar um `SimpleImputer` com a estratégia da mediana aos atributos numéricos [cite: 686][cite_start], simultaneamente aplicar um `OneHotEncoder` aos atributos categóricos textuais [cite: 837][cite_start], e rematar os dados contínuos com um `StandardScaler`[cite: 978]. [cite_start]A seleção do modelo é validada rigorosamente por *Validação Cruzada* (`cross_val_score`) [cite: 1588][cite_start], enquanto a pesquisa pela melhor combinação de hiperparâmetros é delegada a métodos de força bruta (`GridSearchCV`) [cite: 1657] [cite_start]ou estocásticos (`RandomizedSearchCV`)[cite: 1772]. [cite_start]No fim, o objeto completo é exportado usando a biblioteca `joblib`[cite: 2062, 2064].

⚠️ **Pegadinhas Críticas & Códigos para a Prova:**

* [cite_start]**Amostragem Aleatória vs. Estratificada:** Dividir os dados em treino e teste com `train_test_split` puro pode criar amostras pouco representativas se a base for pequena[cite: 287]. [cite_start]Se o rendimento é crucial para o preço da casa, deve-se usar amostragem estratificada (`StratifiedShuffleSplit` ou o argumento `stratify`) para garantir que a proporção das categorias de rendimento é igual no conjunto de treino e no conjunto de teste[cite: 293, 316, 363].
* [cite_start]**O Perigo do RMSE a "Zero":** Se ao treinar um `DecisionTreeRegressor` [cite: 1556] [cite_start]o Root Mean Squared Error (RMSE) der `0.0` nos dados de treino[cite: 1578], isso não é um milagre. [cite_start]O modelo sofreu um severo *overfitting* (memorizou os dados)[cite: 1579, 1645]. [cite_start]É imperativo utilizar Validação Cruzada, que testa o modelo em dados não vistos iterativamente para revelar o verdadeiro erro (que rondará os $66.000)[cite: 1581, 1597].
* [cite_start]**Distribuições Assimétricas (Cauda Longa):** Se um atributo como a "população" tiver uma assimetria forte à direita, aplicar algoritmos diretamente prejudica o modelo[cite: 984]. [cite_start]O truque é aplicar uma transformação logarítmica (`np.log`) antes da normalização standard para tornar a distribuição simétrica[cite: 988, 1426].
* [cite_start]**Encapsulamento do Transformador:** Durante a fase de previsão final nos dados de teste (`X_test`), deves **apenas usar `.predict()` no modelo ou Pipeline já treinado**[cite: 2039]. Nunca deves voltar a fazer `fit()` nos dados de teste, senão causas o temido *Data Leakage* (fuga de dados) e arruínas a avaliação do modelo.

A **teoria** da Regressão Logística (o que é a função Sigmoide, o Log Loss e as métricas de avaliação) foi muito bem mapeada no nosso primeiro grande resumo (lá no tópico 4). Porém, **a parte prática e os códigos** contidos especificamente neste arquivo `05 - Regressão Logística Codigo.pdf` trazem detalhes operacionais essenciais que ainda não aprofundamos no resumo!

Neste arquivo, o foco sai da matemática abstrata e vai direto para a "mão na massa" com o Scikit-Learn. Ele aborda cenários que costumam ser as grandes pegadinhas de provas práticas e também são perguntas clássicas em entrevistas para vagas de nível júnior em Ciência de Dados:

* **Treinamento Real:** O uso do famoso **Dataset Iris** para separar dados de Treino e Validação usando as características (sepal length, petal width, etc.).
* **O Hiperparâmetro `C`:** O arquivo cobra testes variando o `C` ($C=0.01$, $C=1$ e $C=100$). Este parâmetro controla a *regularização* na Regressão Logística no Scikit-Learn (o inverso do alfa da Regressão Linear).
* **Cálculo de Métricas no Código:** Como gerar Acurácia, Precisão e Recall a partir do Conjunto de Validação.
* **Ajuste Manual do *Threshold* (Fronteira de Decisão):** O PDF foca pesadamente em como mudar o ponto de corte do modelo para priorizar Precisão ou Recall dependendo do objetivo de negócio (por exemplo, exigindo que o Recall fique acima de 82%). 

**Resumindo:** A teoria você já tem, mas a execução técnica desses códigos de Regressão Linear e Logística é o que vai garantir que você não trave na hora de implementar a solução. 

---

# 💻 Guia Prático de Códigos: Scikit-Learn (Aulas 04, 05 e 06)

## 1. Regressão Linear e Regularização (Aula 04)
Este módulo foca em prever valores contínuos e ajustar o modelo para evitar underfitting e overfitting.

**Comandos Fundamentais:**
* **Regressão Linear (Equação Normal):** Encontra o melhor ajuste matemático exato.
    ```python
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_) # Exibe onde a reta corta o eixo Y e sua inclinação
    ```
* **Regressão Linear com Gradiente Descendente (SGD):** Ideal para datasets gigantes. Atenção à função `.ravel()` que transforma a matriz `y` em um vetor 1D, exigência do `SGDRegressor`.
    ```python
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, eta0=0.01, random_state=42)
    sgd_reg.fit(X, y.ravel())
    ```
* **Regressão Polinomial:** Quando uma reta não resolve, criamos características elevadas ao quadrado, cubo, etc.
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    ```

**⚠️ Pegadinhas Práticas & Regularização:**
* **Identificando Overfitting:** A biblioteca Scikit-Learn possui a função `learning_curve` que permite avaliar visualmente o RMSE de Treino vs. Validação. Se a curva de treino tem erro muito baixo e a de validação muito alto e não convergem, o modelo decorou os dados (Overfitting).
* **Lidando com Overfitting (Regularização):** * **Ridge (L2):** Penaliza coeficientes grandes. `Ridge(alpha=0.1)`. 
    * **Lasso (L1):** Tende a zerar coeficientes inúteis (seleção automática de features). `Lasso(alpha=0.1)`.
    * **Importante:** Um `alpha` muito grande gera *underfitting* (a reta fica muito plana), e um `alpha` próximo de zero não resolve o *overfitting*.

---

## 2. Regressão Logística e Métricas (Aula 05)
O foco aqui não é prever um valor, mas sim a probabilidade de pertencer a uma classe (ex: É fraude ou não?).

**Comandos Fundamentais:**
* **Treinamento do Classificador:** Diferente da Regressão Linear que usa `alpha`, a Logística usa `C`. O parâmetro `C` é o **inverso** da regularização (quanto maior o `C`, *menor* a regularização).
    ```python
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(C=2, random_state=42)
    log_reg.fit(X_train, y_train)
    ```
* **Pegadinha `predict` vs `predict_proba`:** * `log_reg.predict(X_new)` retorna a categoria final (Ex: `True` ou `False`).
    * `log_reg.predict_proba(X_new)` retorna a matriz com as probabilidades exatas (Ex: 15% chance de Falso, 85% chance de Verdadeiro).

**⚠️ Pegadinhas Práticas & Threshold:**
* **Ajuste manual do Threshold (Fronteira de Decisão):** Por padrão, se a probabilidade passar de 50% (0.5), o Scikit-Learn classifica como Positivo. Se a questão pedir para "aumentar o Recall para 82%", você deve ignorar o `.predict()` padrão, acessar as probabilidades reais calculadas pelo modelo com `method="decision_function"` e criar seu próprio limite:
    ```python
    from sklearn.model_selection import cross_val_predict
    # Pegando a pontuação bruta em vez da classe
    y_scores = cross_val_predict(log_reg, X_train, y_train, cv=3, method="decision_function")
    
    # Criando um limite personalizado (ex: 0.3 em vez de 0.0) para priorizar Recall
    threshold = 0.3
    y_amostra_pred = (y_scores > threshold)
    ```

---

## 3. Projeto End-to-End e Pipelines (Aula 06)
Este módulo é sobre como organizar o código de maneira profissional e reprodutível, juntando transformações de dados e o modelo em um único fluxo.

**Comandos Fundamentais de Preparação:**
* **Amostragem Estratificada:** Evita que a separação de treino/teste fique desbalanceada (muito comum em provas).
    ```python
    from sklearn.model_selection import train_test_split
    # stratify força o split a manter a mesma proporção da coluna informada
    train_set, test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    ```
* **Tratando Nulos e Categorias:**
    * Numéricos: `SimpleImputer(strategy="median")` preenche buracos com a mediana.
    * Texto/Categorias: `OneHotEncoder(sparse_output=False)` converte nomes em colunas binárias.

**⚠️ Pegadinhas Práticas & Pipelines:**
* **Orquestração com `ColumnTransformer`:** Aplica transformações numéricas apenas nas colunas de número, e transformações de texto apenas nas colunas de texto, unindo tudo no final.
    ```python
    from sklearn.compose import make_column_transformer
    from sklearn.compose import make_column_selector
    
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object))
    )
    ```
* **Busca pelos Melhores Hiperparâmetros:** O modelo final não é testado na base do "chute".
    * `GridSearchCV`: Testa *todas* as combinações de uma lista que você passar.
    * `RandomizedSearchCV`: Testa combinações aleatórias. É o método recomendado quando a lista de hiperparâmetros é muito vasta, utilizando funções da biblioteca `scipy.stats` (como `randint` e `expon`).
* **Persistência (Salvar o modelo para uso):** A etapa final de um projeto é exportar o modelo treinado com a biblioteca `joblib`.
    ```python
    import joblib
    joblib.dump(final_model, "meu_modelo.pkl") # Salva
    modelo_carregado = joblib.load("meu_modelo.pkl") # Carrega em produção
    ```
Eles são muito cobrados em avaliações práticas. Podes adicionar este "Anexo" ao teu documento do Word:

---

### 📌 Anexo: Métricas, Multiclasse e Modelos de Árvore

**1. Códigos de Métricas de Classificação (Aula 05)**
A teoria abordou as métricas, mas é assim que as calculas na prática após fazeres as tuas predições:
```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Matriz de Confusão (Verdadeiros/Falsos Positivos e Negativos)
cm = confusion_matrix(y_train, y_train_pred)

# Métricas de Desempenho
precisao = precision_score(y_train, y_train_pred)
revocacao = recall_score(y_train, y_train_pred) # Também chamado de Recall
f1 = f1_score(y_train, y_train_pred)

# Área sob a curva ROC (mede a eficiência geral do modelo)
auc = roc_auc_score(y_train, y_scores)
```

**2. Regressão Softmax / Classificação Multiclasse (Aula 05)**
Quando o teu problema tem mais de duas categorias (ex: não é apenas "Fraude ou Não Fraude", mas sim "Íris Setosa, Versicolor ou Virginica"), usas a Regressão Softmax. No Scikit-Learn, a própria `LogisticRegression` lida com isso automaticamente:
```python
from sklearn.linear_model import LogisticRegression

# O modelo adapta-se automaticamente para múltiplas classes
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

# Devolve a classe final escolhida
softmax_reg.predict([[5, 2]]) 

# Devolve a probabilidade para CADA UMA das classes (ex: 0% Setosa, 4% Versicolor, 96% Virginica)
softmax_reg.predict_proba([[5, 2]]) 
```

**3. Modelos de Árvore e Validação Cruzada (Aula 06)**
Além da Regressão Linear, o teu projeto de ponta a ponta testa Modelos de Árvore e avalia-os com a Validação Cruzada (`cross_val_score`) para garantir que não estão em *overfitting*:
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# 1. Árvore de Decisão Simples
tree_reg = DecisionTreeRegressor(random_state=42)
# Testando com Validação Cruzada (10 dobras/folds)
tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, 
                              scoring="neg_root_mean_squared_error", cv=10)

# 2. Floresta Aleatória (Random Forest - Combina várias árvores)
forest_reg = RandomForestRegressor(random_state=42)
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, 
                                scoring="neg_root_mean_squared_error", cv=10)
```

---