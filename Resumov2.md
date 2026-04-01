# 📚 Guia Definitivo: Ciência de Dados para Negócios

## 1. Introdução ao Aprendizado de Máquina (Aula 02)
**Conceitos-Chave:** Em vez de programar regras fixas, o modelo aprende padrões a partir dos dados (instâncias e características) para gerar previsões.

* **Explicação Simples:** É como ensinar uma criança a reconhecer frutas. Você não explica a botânica da maçã; você mostra várias maçãs (dados) e diz "isso é uma maçã" (rótulo). Com o tempo, ela aprende a identificar uma maçã que nunca viu antes.
* **Explicação Intermediária:** Você fornece um *Dataset* com características (*Features*, ex: tamanho, cor) e rótulos (*Labels*, ex: nome da fruta). O algoritmo encontra a relação matemática entre as características e o rótulo.
* **Técnica / Profissional:** Um modelo ajusta iterativamente seus hiperparâmetros sobre dados de Treino para minimizar uma função de custo, garantindo capacidade de generalização para dados não vistos (Validação/Teste).

⚠️ **Pegadinhas:**
* **Supervisionado vs Não Supervisionado:** Supervisionado tem rótulo (ex: prever preço de casa, classificar spam). Não Supervisionado não tem rótulo (ex: agrupar clientes por perfil de compra).

---

## 2. Pré-Processamento e Engenharia de Características (Aula 03)
**Conceitos-Chave:** Algoritmos não lidam bem com dados ausentes, escalas diferentes ou texto. A limpeza é a etapa mais demorada de um projeto.

* **Explicação Simples:** É como preparar os ingredientes antes de cozinhar. Você precisa lavar, descascar, tirar as partes estragadas (dados nulos) e picar tudo do mesmo tamanho (escalonamento) para que o prato (modelo) não dê errado.
* **Explicação Intermediária:** Se a idade vai de 0 a 100 e a renda de 0 a 150.000, o modelo vai achar que a renda é mais importante só porque o número é maior. Por isso, "esmagamos" tudo para a mesma escala. Além disso, transformamos categorias (como "Sim" ou "Não") em números (1 e 0), pois a matemática não lê letras.
* **Técnica / Profissional:** O tratamento de dados envolve imputação (preenchimento de valores nulos), codificação de variáveis categóricas (One-Hot Encoding) e padronização (`StandardScaler` com média 0 e variância 1) ou normalização (`MinMaxScaler`).

⚠️ **Pegadinhas:**
* Nunca remover linhas com dados nulos se isso significar perder muita informação útil. A imputação (ex: preencher com a mediana) é geralmente preferível.

---

## 3. Regressão Linear, Gradiente Descendente e Regularização (Aula 04)
**Conceitos-Chave:** Encontrar a reta que melhor se ajusta aos dados numéricos contínuos, minimizando o Erro Quadrático Médio (MSE).

* **Explicação Simples:** Imagine vários pontos desenhados em um papel. A Regressão Linear tenta passar uma reta reta exatamente no meio deles.
* **Explicação Intermediária:** O "Gradiente Descendente" é o motor que ajusta essa reta. Pense num alpinista com os olhos vendados tentando chegar ao fundo de um vale. Ele sente a inclinação do chão e dá pequenos passos (Taxa de Aprendizado) até chegar ao ponto mais baixo (menor erro).
* **Técnica / Profissional:** O modelo pode sofrer de *Overfitting* (decorar os dados de treino, gerando alta variância) ou *Underfitting* (não aprender nada, alto viés). A Regularização é aplicada à função de custo para penalizar coeficientes muito grandes. A penalidade L2 (Ridge) encolhe os coeficientes, enquanto a L1 (Lasso) pode zerar coeficientes inúteis.

### 💻 Prática em Código (Aula 04)

```python
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

# 1. Regressão Linear Simples
modelo_linear = LinearRegression()
modelo_linear.fit(X_treino, y_treino)
print("Ponto de corte:", modelo_linear.intercept_)
print("Inclinação da reta:", modelo_linear.coef_)

# 2. Gradiente Descendente Estocástico (Para Datasets gigantes)
# Nota: y.ravel() é usado para transformar a matriz em um vetor 1D
modelo_sgd = SGDRegressor(max_iter=1000, tol=1e-5, eta0=0.01, random_state=42)
modelo_sgd.fit(X_treino, y_treino.ravel())

# 3. Regressão Polinomial (Quando os dados fazem uma curva)
# Cria características elevadas ao quadrado (grau 2)
caracteristicas_polinomiais = PolynomialFeatures(degree=2, include_bias=False)
X_treino_polinomial = caracteristicas_polinomiais.fit_transform(X_treino)
modelo_linear.fit(X_treino_polinomial, y_treino)

# 4. Regularização (Evitando Overfitting)
modelo_ridge = Ridge(alpha=0.1, solver="cholesky") # Penaliza pesos altos (L2)
modelo_lasso = Lasso(alpha=0.1) # Pode zerar pesos irrelevantes (L1)
```

---

## 4. Regressão Logística e Métricas (Aula 05)
**Conceitos-Chave:** Apesar do nome "Regressão", é um algoritmo de **Classificação**. Usa a função Sigmoide para calcular a probabilidade de uma instância pertencer a uma classe (ex: Spam vs. Não Spam).

* **Explicação Simples:** Ele responde perguntas de "Sim ou Não". Calcula a porcentagem de chance de algo ser "Sim". Se der mais de 50%, ele classifica como "Sim".
* **Explicação Intermediária:** A função de custo aqui não é o erro da distância (MSE), mas sim o *Log Loss* (Entropia Cruzada), que dá uma punição enorme ao modelo se ele disser que tem 99% de certeza que um e-mail é normal, mas na verdade era vírus.
* **Técnica / Profissional:** As métricas de avaliação são críticas. *Acurácia* (taxa de acerto geral); *Precisão* (dos que classifiquei como positivo, quantos eram? - Evita falsos positivos); *Recall/Revocação* (dos positivos reais, quantos encontrei? - Evita falsos negativos); *F1-Score* (Média harmônica entre Precisão e Recall).

### 💻 Prática em Código (Aula 05)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

# 1. Treinamento (O parâmetro C é o inverso do alpha. C maior = menos regularização)
modelo_logistico = LogisticRegression(C=2, random_state=42)
modelo_logistico.fit(X_treino, y_treino)

# 2. Pegadinha: predict vs predict_proba
classe_prevista = modelo_logistico.predict(X_teste) # Retorna True/False ou 1/0
probabilidade = modelo_logistico.predict_proba(X_teste) # Retorna as % exatas

# 3. Métricas
# Usamos Validação Cruzada para não testar o modelo nos mesmos dados que ele treinou
previsoes_cruzadas = cross_val_predict(modelo_logistico, X_treino, y_treino, cv=3)

matriz_confusao = confusion_matrix(y_treino, previsoes_cruzadas)
precisao = precision_score(y_treino, previsoes_cruzadas)
revocacao = recall_score(y_treino, previsoes_cruzadas) # Recall

# 4. Ajuste Manual de Fronteira de Decisão (Threshold)
# Obtendo a pontuação bruta da função de decisão
pontuacoes_decisao = cross_val_predict(modelo_logistico, X_treino, y_treino, cv=3, method="decision_function")

limite_personalizado = 0.3 # Abaixando de 0.0 para 0.3 para focar no Recall
previsoes_personalizadas = (pontuacoes_decisao > limite_personalizado)
```

---

## 5. Projeto End-to-End e Pipelines Automáticos (Aula 06)
**Conceitos-Chave:** Organizar todo o fluxo de trabalho (limpeza, transformação, modelagem e exportação) de forma contínua e imune ao Vazamento de Dados (*Data Leakage*).

* **Explicação Simples:** É a esteira da fábrica. O ingrediente cru entra de um lado, é lavado, cortado, temperado e o prato sai pronto do outro lado, tudo automatizado.
* **Explicação Intermediária:** O Scikit-Learn usa `Pipelines` para empilhar passos. O modelo só vê os dados depois que eles passam por todas as transformações (imputação de nulos, One-Hot Encoding, padronização). 
* **Técnica / Profissional:** O `ColumnTransformer` permite ramificar a esteira: dados numéricos vão para um pipeline matemático, dados em texto vão para um pipeline categórico. Para achar os melhores parâmetros do modelo automaticamente sem chutar, usamos o `GridSearchCV` (testa combinações fixas) ou o `RandomizedSearchCV` (testa aleatoriamente dentro de um intervalo estatístico).

### 💻 Prática em Código (Aula 06)

```python
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Amostragem Estratificada (Mantém a proporção das categorias da vida real)
# Supondo que "categoria_renda" seja uma coluna importante no dataset "dados_habitacao"
separador = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for indice_treino, indice_teste in separador.split(dados_habitacao, dados_habitacao["categoria_renda"]):
    conjunto_treino = dados_habitacao.iloc[indice_treino]
    conjunto_teste = dados_habitacao.iloc[indice_teste]

# 2. Criando as esteiras (Pipelines) de tratamento
pipeline_numerico = make_pipeline(
    SimpleImputer(strategy="median"), # Preenche nulos numéricos com a mediana
    StandardScaler()                  # Padroniza escalas
)

pipeline_categorico = make_pipeline(
    SimpleImputer(strategy="most_frequent"), # Preenche nulos de texto com a moda
    OneHotEncoder(handle_unknown="ignore", sparse_output=False) # Converte texto para colunas 0 e 1
)

# 3. Unindo as esteiras (Column Transformer)
preprocessamento_total = make_column_transformer(
    (pipeline_numerico, make_column_selector(dtype_include=np.number)),
    (pipeline_categorico, make_column_selector(dtype_include=object))
)

# 4. Pipeline Completo: Tratamento + Modelo Final
pipeline_final = make_pipeline(
    preprocessamento_total,
    RandomForestRegressor(random_state=42)
)

# 5. Otimização de Hiperparâmetros (Grid Search)
parametros_para_testar = {
    'randomforestregressor__max_features': [4, 6, 8]
}

busca_em_grade = GridSearchCV(pipeline_final, parametros_para_testar, cv=3, scoring='neg_root_mean_squared_error')
busca_em_grade.fit(X_treino, y_treino)

# Ver qual foi o melhor modelo encontrado
melhor_modelo = busca_em_grade.best_estimator_

# 6. Salvar o modelo para colocar em produção (Exportação)
joblib.dump(melhor_modelo, "meu_modelo_otimizado.pkl")

# (No outro dia, o servidor de produção carregaria assim:)
# modelo_producao = joblib.load("meu_modelo_otimizado.pkl")
# predições = modelo_producao.predict(novos_dados_do_cliente)
```

Aqui tens o guião de estudo completo e definitivo, restruturado num formato Markdown (`.md`) altamente visual, perfeito para leres no teu editor de código (como o VS Code), Obsidian ou Notion. 

Conforme pediste, todo o material das aulas 02 a 06 foi unificado. O código foi reescrito com variáveis e funções em português para que a lógica e a intuição por trás de cada linha fiquem cristalinas. Lembra-te apenas de que, na prática, os métodos oficiais da biblioteca Scikit-Learn (`fit`, `predict`, etc.) mantêm-se em inglês.

O domínio destes ficheiros e destas ferramentas no Scikit-Learn garante a preparação ideal não só para a prova teórica e prática, mas também para os desafios técnicos exigidos nas vagas de nível júnior em Análise e Ciência de Dados na região de Sorocaba.

---

# 📘 Guião de Estudo: Ciência de Dados para Negócios

## 1. Introdução ao Aprendizado de Máquina (Aula 02)
Em vez de programar regras fixas, o modelo matemático aprende os padrões a partir dos dados (instâncias e características) e dos seus resultados, conseguindo generalizar e fazer previsões para novos dados.

* **Supervisionado:** Os dados têm rótulos (a resposta certa já é conhecida). Usado para **Regressão** (prever valores numéricos, como preços) ou **Classificação** (prever categorias, como spam/não spam).
* **Não Supervisionado:** Os dados não têm rótulos. O algoritmo procura padrões ocultos por si mesmo, como no agrupamento (Clustering) de perfis de clientes.

---

## 2. Pré-processamento e Engenharia de Características (Aula 03)
Algoritmos não conseguem processar valores ausentes, textos puros ou escalas muito desproporcionais. A limpeza é a fundação de qualquer modelo.

* **Tratamento de Nulos:** Evita eliminar linhas inteiras para não perderes informação valiosa. É preferível fazer a **imputação** (preencher os vazios com a média ou mediana).
* **Tratamento de Escalas:** Se uma coluna varia de 0 a 100 e a outra de 0 a 10.000, o modelo dará mais peso à segunda. A Padronização (`StandardScaler`) ou Normalização (`MinMaxScaler`) nivela as grandezas.

---

## 3. Regressão Linear, Gradiente Descendente e Regularização (Aula 04)
A regressão procura traçar a melhor reta (ou curva) possível que passe perto de todos os pontos de dados, minimizando o Erro Quadrático Médio (MSE). O **Gradiente Descendente** é o otimizador que vai ajustando o ângulo dessa reta passo a passo até encontrar o ponto de erro mínimo.

### 💻 Prática em Código: Regressão Linear
```python
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

# --- 1. REGRESSÃO LINEAR SIMPLES ---
modelo_linear = LinearRegression()
modelo_linear.fit(caracteristicas_treino, alvo_treino)

print("Ponto de Interceção (Y):", modelo_linear.intercept_)
print("Inclinação da Reta:", modelo_linear.coef_)

# --- 2. GRADIENTE DESCENDENTE ESTOCÁSTICO (SGD) ---
# Usado quando o dataset é demasiado grande para a memória
# O ravel() converte a matriz alvo num formato de lista 1D, exigido pelo SGD
modelo_gradiente = SGDRegressor(max_iter=1000, tol=1e-5, eta0=0.01, random_state=42)
modelo_gradiente.fit(caracteristicas_treino, alvo_treino.ravel())

# --- 3. REGRESSÃO POLINOMIAL ---
# Usado quando os dados não formam uma linha reta, mas sim uma curva
gerador_polinomial = PolynomialFeatures(degree=2, include_bias=False)
caracteristicas_curvas = gerador_polinomial.fit_transform(caracteristicas_treino)
modelo_linear.fit(caracteristicas_curvas, alvo_treino)

# --- 4. REGULARIZAÇÃO (PENALIZAÇÃO) ---
# Evita o Overfitting (quando o modelo "decora" os dados e perde a flexibilidade)
modelo_ridge = Ridge(alpha=0.1, solver="cholesky") # Penaliza coeficientes muito altos (L2)
modelo_lasso = Lasso(alpha=0.1)                    # Força coeficientes inúteis a zero (L1)
```

---

## 4. Regressão Logística e Métricas de Classificação (Aula 05)
Apesar do nome, a Regressão Logística é usada para **Classificação**. Utiliza a matemática da função Sigmoide em forma de "S" para comprimir qualquer resultado entre 0 e 1, devolvendo uma **probabilidade** (percentagem de chance).

Avaliar um modelo de classificação com a Curva ROC e a métrica de Revocação (Recall) segue rigorosamente o mesmo princípio de avaliar a Sensibilidade e a Especificidade de um teste de diagnóstico laboratorial na biomedicina clínica. Não queremos falsos positivos que disparem alarmes errados, mas sobretudo não podemos tolerar falsos negativos numa classificação crítica.

### 💻 Prática em Código: Regressão Logística
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

# --- 1. TREINAMENTO DO MODELO LOGÍSTICO ---
# O parâmetro C é o inverso da regularização (C maior = Menos regularização)
modelo_classificador = LogisticRegression(C=2, random_state=42)
modelo_classificador.fit(caracteristicas_treino, categorias_treino)

# --- 2. CLASSE PREVISTA VS PROBABILIDADE ---
decisao_final = modelo_classificador.predict(novos_dados) # Retorna a categoria (ex: 1 ou 0)
percentagens = modelo_classificador.predict_proba(novos_dados) # Retorna as % [4%, 96%]

# --- 3. AVALIAÇÃO COM VALIDAÇÃO CRUZADA ---
# Fazemos previsões em fatias ocultas dos dados para não testar com o que já foi treinado
previsoes_cruzadas = cross_val_predict(modelo_classificador, caracteristicas_treino, categorias_treino, cv=3)

matriz_erros = confusion_matrix(categorias_treino, previsoes_cruzadas)
precisao = precision_score(categorias_treino, previsoes_cruzadas) # Dos que achei que eram X, quantos eram de facto?
revocacao = recall_score(categorias_treino, previsoes_cruzadas)   # De todos os X que existem, quantos consegui apanhar?

# --- 4. AJUSTE DO LIMITE (THRESHOLD) DA FRONTEIRA DE DECISÃO ---
# Queremos focar em aumentar a Revocação, logo baixamos a exigência do limite de 50% para 30%
pontuacoes_brutas = cross_val_predict(modelo_classificador, caracteristicas_treino, categorias_treino, cv=3, method="decision_function")
limite_permissivo = 0.3
novas_previsoes = (pontuacoes_brutas > limite_permissivo)
```

---

## 5. Projeto de Ponta a Ponta e Automação (Aula 06)
O projeto final ensina a não executar os códigos de pré-processamento um a um. Num ambiente corporativo, crias um `Pipeline` completo (uma linha de montagem) que recebe os dados em bruto de um lado e entrega o modelo ajustado do outro.

### 💻 Prática em Código: Pipeline e GridSearchCV
```python
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

# --- 1. AMOSTRAGEM ESTRATIFICADA ---
# Garante que as divisões de treino e teste mantêm a mesma proporção de uma categoria vital
separador_estratificado = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for indice_treino, indice_teste in separador_estratificado.split(dados_brutos, dados_brutos["categoria_de_renda"]):
    dados_treino = dados_brutos.iloc[indice_treino]
    dados_teste = dados_brutos.iloc[indice_teste]

# --- 2. TRANSFORMADOR MATEMÁTICO PERSONALIZADO ---
# Útil para aplicar operações que não vêm de fábrica, como calcular logaritmos
transformador_logaritmo = FunctionTransformer(np.log, inverse_func=np.exp)

# --- 3. CRIAÇÃO DAS LINHAS DE MONTAGEM (PIPELINES) ---
esteira_numerica = make_pipeline(
    SimpleImputer(strategy="median"), # Tapa os buracos dos números com a mediana
    StandardScaler()                  # Equilibra a escala (Padronização)
)

esteira_texto = make_pipeline(
    SimpleImputer(strategy="most_frequent"),                    # Tapa buracos com a palavra mais comum
    OneHotEncoder(handle_unknown="ignore", sparse_output=False) # Converte palavras em colunas binárias 0 e 1
)

# --- 4. ORQUESTRADOR CENTRAL (COLUMN TRANSFORMER) ---
# Diz qual esteira deve processar que tipo de dados
processador_global = make_column_transformer(
    (esteira_numerica, make_column_selector(dtype_include=np.number)),
    (esteira_texto, make_column_selector(dtype_include=object))
)

# --- 5. PIPELINE FINAL COM O MODELO INCLUÍDO ---
modelo_de_ponta_a_ponta = make_pipeline(
    processador_global,
    RandomForestRegressor(random_state=42) # Floresta Aleatória
)

# --- 6. OTIMIZAÇÃO AUTOMÁTICA DE PARÂMETROS (GRID SEARCH) ---
# Em vez de tentar "chutar" qual é a melhor configuração, a máquina testa todas
parametros_para_testar = {
    'randomforestregressor__max_features': [4, 6, 8]
}

buscador_de_parametros = GridSearchCV(modelo_de_ponta_a_ponta, parametros_para_testar, cv=3, scoring='neg_root_mean_squared_error')
buscador_de_parametros.fit(caracteristicas_treino, alvo_treino)

# --- 7. EXPORTAÇÃO E PERSISTÊNCIA ---
melhor_modelo_encontrado = buscador_de_parametros.best_estimator_
joblib.dump(melhor_modelo_encontrado, "modelo_mercado_imobiliario.pkl")

# Num servidor de produção na nuvem, importaríamos assim:
# modelo_em_producao = joblib.load("modelo_mercado_imobiliario.pkl")
```

***

```markdown
# 📘 Guião Completo: Aprendizado de Máquina (Machine Learning)

---

## 📌 Módulo 1: Introdução ao Aprendizado de Máquina (Aula 02)

**Conceitos-Chave:** Em vez de programar regras lógicas estáticas (programação tradicional), o modelo matemático aprende padrões a partir dos dados fornecidos para gerar previsões.

* **Nível 1 (Simples e Intuitivo):** É como ensinar uma criança a distinguir cães de gatos. Não lhe explicas a biologia da espécie; mostras-lhe 100 fotos de cães e 100 fotos de gatos. O cérebro dela (o modelo) identifica os padrões sozinha.
* **Nível 2 (Intermediário com Exemplos):** Tens um *Dataset* (conjunto de dados) com **Características** (peso, altura, cor) e **Rótulos** (se é cão ou gato). O algoritmo encontra a equação matemática que liga as características ao rótulo.
* **Nível 3 (Técnica e Profissional):** Um modelo supervisionado mapeia vetores de entrada (variáveis independentes) para uma variável alvo. O objetivo é ajustar iterativamente os hiperparâmetros sobre os dados de **Treino** minimizando a função de custo, garantindo a capacidade de generalização para dados nunca vistos (dados de **Teste**).

⚠️ **Pegadinhas Comuns:**
* **Supervisionado vs. Não Supervisionado:** Se os dados têm a resposta esperada (rótulo), é Supervisionado (ex: prever preço de casas). Se não têm resposta e o algoritmo apenas agrupa dados semelhantes, é Não Supervisionado (ex: segmentação de clientes por perfil).
* **Overfitting vs. Underfitting:** *Overfitting* (Sobreajuste) ocorre quando o modelo "decora" os dados de treino e falha no mundo real. *Underfitting* (Subajuste) ocorre quando o modelo é demasiado simples para captar os padrões dos dados.

---

## 📌 Módulo 2: Pré-processamento e Engenharia de Características (Aula 03)

**Conceitos-Chave:** A qualidade dos dados dita a qualidade do modelo ("Garbage in, garbage out"). Algoritmos não processam valores ausentes, textos puros ou escalas díspares.

* **Nível 1 (Simples e Intuitivo):** É o equivalente a preparar os ingredientes antes de cozinhar. Tens de lavar os legumes, retirar as partes estragadas (dados nulos) e cortar tudo do mesmo tamanho (escalonamento) para que o prato não fique arruinado.
* **Nível 2 (Intermediário com Exemplos):** Se a variável "Idade" varia de 0 a 100 e a "Renda" varia de 0 a 150.000, o modelo matemático vai achar erradamente que a renda é mais importante só porque o número é maior. Além disso, a matemática não lê as palavras "Sim" e "Não", pelo que temos de as converter em "1" e "0".
* **Nível 3 (Técnica e Profissional):** A fase de pré-processamento consome cerca de 80% do tempo do projeto. Envolve a **Imputação** (substituição de valores *NaN* pela média ou mediana), **Codificação Categórica** (usando matrizes esparsas como o *One-Hot Encoding*) e **Escalonamento de Características** (através de *MinMaxScaler* ou *StandardScaler* para normalização e padronização geométrica).

⚠️ **Pegadinhas Comuns:**
* Nunca elimines linhas com dados ausentes (`dropna`) de forma impulsiva, pois perdes informação valiosa. Utiliza sempre estratégias de imputação, a menos que o atributo tenha quase 100% de valores nulos.

---

## 📌 Módulo 3: Regressão Linear e Regularização (Aula 04)

**Conceitos-Chave:** O objetivo é encontrar a melhor reta (hiperplano) que se ajuste a dados numéricos contínuos, minimizando o Erro Quadrático Médio (MSE).

* **Nível 1 (Simples e Intuitivo):** Imagina dezenas de pontos desenhados numa folha de papel. A regressão linear procura desenhar uma linha reta que passe exatamente no meio de todos os pontos, com a menor distância possível para cada um deles.
* **Nível 2 (Intermediário com Exemplos):** Para ajustar essa reta, usamos o **Gradiente Descendente**. Imagina uma pessoa de olhos vendados a tentar descer uma montanha. Ela sente a inclinação do terreno com o pé e dá pequenos passos (a Taxa de Aprendizado) até chegar ao fundo do vale (onde o erro é zero).
* **Nível 3 (Técnica e Profissional):** O modelo otimiza os coeficientes $	heta$ calculando as derivadas parciais da função de custo MSE. Para evitar o *overfitting* em modelos complexos (como a Regressão Polinomial), aplicamos **Regularização**: restrições matemáticas na função de custo. A norma L2 (Ridge) encolhe os pesos numéricos; a norma L1 (Lasso) atua como seletor de características, forçando pesos irrelevantes a zero absoluto.

### 💻 Prática em Código (Scikit-Learn)

```python
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures

# 1. Regressão Linear Clássica (Equação Normal)
modelo_linear = LinearRegression()
modelo_linear.fit(caracteristicas_treino, alvo_treino)
print("Ponto de Interceção:", modelo_linear.intercept_)
print("Pesos/Coeficientes:", modelo_linear.coef_)

# 2. Gradiente Descendente Estocástico (SGD) - Para Datasets Gigantes
# O .ravel() converte a matriz alvo num vetor 1D, exigido pelo algoritmo
modelo_sgd = SGDRegressor(max_iter=1000, tol=1e-5, eta0=0.01, random_state=42)
modelo_sgd.fit(caracteristicas_treino, alvo_treino.ravel())

# 3. Regressão Polinomial (Para dados que formam curvas)
# Cria características elevadas ao quadrado (grau 2)
gerador_polinomial = PolynomialFeatures(degree=2, include_bias=False)
caracteristicas_curvas = gerador_polinomial.fit_transform(caracteristicas_treino)
modelo_linear.fit(caracteristicas_curvas, alvo_treino)

# 4. Regularização (Penalização para evitar Overfitting)
modelo_ridge = Ridge(alpha=0.1, solver="cholesky") # Penalidade L2
modelo_lasso = Lasso(alpha=0.1)                    # Penalidade L1 (Pode zerar coeficientes)
modelo_elastic = ElasticNet(alpha=0.1, l1_ratio=0.5) # Mistura 50/50 de Ridge e Lasso
```

---

## 📌 Módulo 4: Regressão Logística e Métricas (Aula 05)

**Conceitos-Chave:** Apesar de ter "Regressão" no nome, é um modelo de **Classificação**. Calcula a probabilidade de uma instância pertencer a uma determinada classe.

* **Nível 1 (Simples e Intuitivo):** O modelo serve para responder com "Sim" ou "Não" (ex: É fraude?). Em vez de dar um número solto, dá uma percentagem de probabilidade. Se for maior que 50%, classifica como "Sim".
* **Nível 2 (Intermediário com Exemplos):** A matemática baseia-se na função **Sigmoide**, que tem a forma de um "S" e esmaga qualquer número de entrada para um limite estrito entre 0 e 1. A função de custo não é o MSE, mas sim o **Log Loss** (Entropia Cruzada), que pune severamente o modelo se este der 99% de certeza a uma resposta errada.
* **Nível 3 (Técnica e Profissional):** A avaliação do desempenho é crítica e vai além da *Acurácia* global. Analisamos a **Precisão** (rácios de falsos positivos), a **Revocação / Recall** (rácios de falsos negativos) e a **F1-Score** (média harmónica). Num cenário multiclasse, a Regressão Logística generaliza-se através da função **Softmax**, convertendo pontuações brutas em probabilidades para múltiplas categorias (ex: Baixo, Médio, Alto).

### 💻 Prática em Código (Scikit-Learn)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict

# 1. Treinamento Base (C é o inverso do Alpha. Um C maior = menos regularização)
modelo_classificador = LogisticRegression(C=2, random_state=42)
modelo_classificador.fit(caracteristicas_treino, categorias_treino)

# 2. Previsão Direta vs Probabilidade
categoria_final = modelo_classificador.predict(novos_dados)       # Devolve a classe final (Ex: 1)
probabilidade_exata = modelo_classificador.predict_proba(novos_dados) # Devolve as % (Ex: [0.15, 0.85])

# 3. Avaliação com Validação Cruzada e Métricas
previsoes_cruzadas = cross_val_predict(modelo_classificador, caracteristicas_treino, categorias_treino, cv=3)

matriz_erros = confusion_matrix(categorias_treino, previsoes_cruzadas)
precisao = precision_score(categorias_treino, previsoes_cruzadas)
revocacao = recall_score(categorias_treino, previsoes_cruzadas) # Fundamental para não deixar escapar casos positivos
f1 = f1_score(categorias_treino, previsoes_cruzadas)

# 4. Ajuste de Limiar (Threshold) da Fronteira de Decisão
# Se quisermos focar em não deixar escapar NENHUM positivo (Recall alto), baixamos o limite
pontuacoes_brutas = cross_val_predict(modelo_classificador, caracteristicas_treino, categorias_treino, cv=3, method="decision_function")
limite_permissivo = 0.3 # O padrão é 0.0 (50%). Baixamos para 0.3.
previsoes_ajustadas = (pontuacoes_brutas > limite_permissivo)

# 5. Regressão Softmax (Múltiplas Categorias em vez de apenas Sim/Não)
modelo_multiclasse = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
modelo_multiclasse.fit(caracteristicas_treino, categorias_multiplas)
```

---

## 📌 Módulo 5: Projeto de Ponta a Ponta (Aula 06)

**Conceitos-Chave:** Num cenário profissional, o modelo de Machine Learning é apenas a ponta do iceberg. O projeto exige automação integral através de *Pipelines* para evitar o Vazamento de Dados (*Data Leakage*).

* **Nível 1 (Simples e Intuitivo):** Em vez de tratar os dados "à mão" um a um, constróis uma linha de montagem de fábrica. Os dados brutos entram de um lado, são limpos, transformados, previstos e empacotados do outro lado de forma 100% automática.
* **Nível 2 (Intermediário com Exemplos):** A Análise Exploratória (EDA) é vital antes do modelo. Visualizamos mapas (latitude/longitude), procuramos correlações de Pearson (quais colunas impactam mais o preço) e criamos novas colunas (ex: dividir o número de quartos pelo número de famílias).
* **Nível 3 (Técnica e Profissional):** Utiliza-se o `ColumnTransformer` para bifurcar as transformações: variáveis contínuas passam por Imputação e Padronização; variáveis categóricas passam por *One-Hot Encoding*. O modelo é encapsulado no *Pipeline* e avaliado via *Validação Cruzada*. Por fim, não "chutamos" os hiperparâmetros: automatizamos a pesquisa profunda com o `GridSearchCV` e persistimos o objeto binário para produção com a biblioteca `joblib`.

### 💻 Prática em Código (Scikit-Learn)

```python
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Separação Estratificada (Garante que Treino/Teste têm a mesma proporção da vida real)
separador = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for indice_treino, indice_teste in separador.split(dados_brutos, dados_brutos["categoria_renda"]):
    dados_treino = dados_brutos.iloc[indice_treino]
    dados_teste = dados_brutos.iloc[indice_teste]

# 2. Análise Exploratória Rápida (EDA)
# Matriz de correlação linear para perceber o que afeta o alvo
matriz_correlacao = dados_treino.corr(numeric_only=True)
print(matriz_correlacao["alvo_preco"].sort_values(ascending=False))

# 3. Construção das Linhas de Montagem (Pipelines)
esteira_numerica = make_pipeline(
    SimpleImputer(strategy="median"), # Tapa buracos com a mediana
    StandardScaler()                  # Padroniza escalas
)

esteira_texto = make_pipeline(
    SimpleImputer(strategy="most_frequent"),                    # Tapa buracos com o mais frequente
    OneHotEncoder(handle_unknown="ignore", sparse_output=False) # Texto para colunas binárias densas
)

# 4. Orquestrador Central (Column Transformer)
processador_global = make_column_transformer(
    (esteira_numerica, make_column_selector(dtype_include=np.number)),
    (esteira_texto, make_column_selector(dtype_include=object))
)

# 5. Pipeline Completo: Tratamento + Modelo de Floresta Aleatória
pipeline_final = make_pipeline(
    processador_global,
    RandomForestRegressor(random_state=42)
)

# 6. Otimização de Hiperparâmetros (Grid Search vs Randomized Search)
# O algoritmo vai testar todas as combinações destas listas automaticamente
grelha_de_parametros = {
    'randomforestregressor__max_features': [4, 6, 8],
    'randomforestregressor__n_estimators': [30, 50, 100]
}

buscador_otimizado = GridSearchCV(pipeline_final, grelha_de_parametros, cv=3, scoring='neg_root_mean_squared_error')
buscador_otimizado.fit(caracteristicas_treino, alvo_treino) # Executa o treino massivo

# Extrai o modelo vencedor
melhor_modelo_pronto = buscador_otimizado.best_estimator_

# 7. Persistência do Modelo (Salvar o trabalho final)
joblib.dump(melhor_modelo_pronto, "modelo_final_producao.pkl")

# (No momento da entrada em produção no servidor:)
# modelo_em_producao = joblib.load("modelo_final_producao.pkl")
# previsao_cliente = modelo_em_producao.predict(dados_novos_cliente)
```
```

Sendo 100% rigoroso, o guião anterior cobriu a **espinha dorsal** de toda a teoria e os códigos mais importantes para o funcionamento dos modelos. No entanto, para que o material seja didático e não vire uma cópia exata de dezenas de páginas de PDF, alguns **tópicos avançados e códigos de otimização específicos** foram condensados ou deixados de fora.

Se você quiser ter a garantia de que **absolutamente tudo** o que estava nos cadernos está no seu `.md` (especialmente para consultas futuras no mercado de trabalho), adicione este bloco final ao seu arquivo. Ele contém as técnicas avançadas que fecham a lacuna dos arquivos 04, 05 e 06:

***

```markdown
## 📌 Módulo Bônus: Técnicas Avançadas e Otimização

### 1. Parada Antecipada (Early Stopping) e Curvas de Aprendizado (Aula 04)
**Conceito:** O *Early Stopping* interrompe o treinamento assim que o erro de validação atinge o seu mínimo, evitando que o modelo continue a treinar e entre em *Overfitting* (sobreajuste). As *Curvas de Aprendizado* permitem visualizar graficamente esse fenômeno.

```python
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

# 1. Parada Antecipada (Early Stopping) com SGD
# warm_start=True permite que o modelo continue o treinamento de onde parou em vez de reiniciar
modelo_sgd = SGDRegressor(max_iter=1, tol=None, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

erro_minimo_validacao = float("inf")
melhor_modelo = None

for epoca in range(1000):
    modelo_sgd.fit(caracteristicas_treino, alvo_treino) # Treina uma época
    previsao_validacao = modelo_sgd.predict(caracteristicas_validacao)
    erro_validacao = mean_squared_error(alvo_validacao, previsao_validacao)
    
    if erro_validacao < erro_minimo_validacao:
        erro_minimo_validacao = erro_validacao
        melhor_modelo = clone(modelo_sgd) # Salva uma cópia do modelo no seu melhor momento

# 2. Curvas de Aprendizado (Learning Curves)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

tamanhos_treino, erros_treino, erros_validacao = learning_curve(
    modelo_linear, caracteristicas_treino, alvo_treino, cv=5, scoring="neg_mean_squared_error"
)
# (A plotagem do gráfico com plt.plot ajuda a diagnosticar Overfitting vs Underfitting)
```

### 2. Avaliação Visual com Curva ROC (Aula 05)
**Conceito:** A Curva ROC plota a Taxa de Verdadeiros Positivos (Recall) contra a Taxa de Falsos Positivos para todos os limiares de classificação possíveis. Quanto mais próxima a curva estiver do canto superior esquerdo, melhor é o classificador.

```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Calcula as taxas para a curva ROC
fpr, tpr, limiares = roc_curve(categorias_treino, pontuacoes_brutas)

# Plotagem básica da Curva ROC
plt.plot(fpr, tpr, linewidth=2, label="Curva ROC")
plt.plot([0, 1], [0, 1], 'k--') # Reta tracejada representando um classificador aleatório
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR) - Recall")
plt.title("Curva ROC do Classificador Logístico")
plt.show()
```

### 3. Busca Aleatória e Transformadores Customizados em Classe (Aula 06)
**Conceito:** Enquanto o `GridSearchCV` testa todas as combinações exatas, o `RandomizedSearchCV` sorteia valores aleatórios dentro de uma distribuição estatística, sendo muito mais eficiente quando o espaço de busca é vasto. Além disso, para transformações complexas que precisam "aprender" algo dos dados de treino (como `fit`), precisamos criar classes baseadas no Scikit-Learn.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, expon
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Busca Aleatória (Randomized Search)
# Usamos distribuições estatísticas como randint (inteiros) e expon (exponencial)
distribuicao_parametros = {
    'randomforestregressor__max_features': randint(low=2, high=20),
    'randomforestregressor__n_estimators': randint(low=10, high=150)
}

busca_aleatoria = RandomizedSearchCV(
    pipeline_final, 
    param_distributions=distribuicao_parametros, 
    n_iter=10, # Testa 10 combinações aleatórias diferentes
    cv=3, 
    scoring='neg_root_mean_squared_error', 
    random_state=42
)
busca_aleatoria.fit(caracteristicas_treino, alvo_treino)

# 2. Transformador Customizado (BaseEstimator, TransformerMixin)
# Permite integração perfeita com Pipelines para comportamentos não nativos do Scikit-Learn
class MeuTransformador(BaseEstimator, TransformerMixin):
    def __init__(self, adicionar_coluna_extra=True):
        self.adicionar_coluna_extra = adicionar_coluna_extra
        
    def fit(self, X, y=None):
        return self # Nada a aprender dos dados neste caso específico
        
    def transform(self, X):
        if self.adicionar_coluna_extra:
            coluna_nova = X[:, 0] / X[:, 1]
            return np.c_[X, coluna_nova]
        else:
            return X
```
```
***