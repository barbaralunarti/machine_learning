# 📌 Projeto de Machine Learning

Este repositório contém códigos que exploram algoritmos de aprendizado de máquina, utilizando bibliotecas do ecossistema Python. O projeto aborda diferentes técnicas de modelagem, pré-processamento de dados e avaliação de modelos, com foco em classificação e regressão.

## 📦 Bibliotecas Utilizadas

Os códigos fazem uso das seguintes bibliotecas:

- **pandas**: Manipulação e análise de dados estruturados.  
- **scikit-learn**: Implementação de algoritmos de aprendizado de máquina.  
- **matplotlib**: Visualização de dados e métricas dos modelos.  
- **feature_engine**: Ferramentas avançadas para pré-processamento de dados.  
- **scikit-plot**: Visualização de curvas ROC, lift, KS e outros gráficos de avaliação.  
- **scipy**: Cálculos matemáticos avançados.

## ⚙️ Principais Funções e Métodos Utilizados

### 📌 Modelagem

#### **tree.DecisionTreeClassifier**  
Implementa um classificador baseado em árvores de decisão.

#### **tree.DecisionTreeRegressor**  
Versão para problemas de regressão.

#### **ensemble.RandomForestClassifier**  
Algoritmo de florestas aleatórias que combina diversas árvores de decisão para melhorar a precisão.

#### **naive_bayes.GaussianNB**  
Classificador Naïve Bayes para dados que seguem distribuição Gaussiana.

#### **linear_model.LogisticRegression**  
Modelo de regressão logística para classificação binária e multiclasse.

### 📊 Predição

#### **arvore.predict(train)** / **arvore.predict(test)**  
Realiza previsões do modelo treinado nos conjuntos de treino e teste.

#### **arvore.predict_proba(train)** / **arvore.predict_proba(test)**  
Retorna as probabilidades das classes previstas pelo modelo.

### 📏 Avaliação de Modelos

#### **metrics.accuracy_score(train)** / **metrics.accuracy_score(test)**  
Calcula a acurácia do modelo nos dados de treino e teste.

#### **metrics.precision_score** / **metrics.recall_score**  
Métricas de avaliação que medem a precisão e a abrangência das previsões.

#### **metrics.confusion_matrix**  
Gera a matriz de confusão para análise de classificação.

#### **metrics.roc_auc_score(train)** / **metrics.roc_auc_score(test)**  
Calcula a área sob a curva ROC para medir a capacidade discriminativa do modelo.

### 📊 Visualização de Métricas

#### **skplt.metrics.plot_roc**  
Plota a curva ROC do modelo para análise do desempenho.

#### **skplt.metrics.plot_cumulative_gain**  
Gera a curva de ganho acumulado, útil para classificação.

#### **skplt.metrics.plot_lift_curve**  
Exibe a curva Lift para medir o desempenho do modelo.

#### **skplt.metrics.plot_ks_statistic**  
Mostra a estatística KS para avaliar separação entre classes.

### 🔄 Engenharia de Atributos e Pipelines

#### **imputation.ArbitraryNumberImputer**  
Preenche valores ausentes com um número arbitrário.

#### **encoding.OneHotEncoder**  
Transforma variáveis categóricas em um formato binário (one-hot encoding).

#### **model_selection.GridSearchCV**  
Busca os melhores hiperparâmetros do modelo utilizando validação cruzada.

#### **pipeline.Pipeline**  
Cria uma sequência estruturada de transformações e modelos.

#### **meu_pipeline.fit**  
Treina todo o pipeline nos dados de treino.

#### **meu_pipeline[-1].best_estimator_.feature_importances_**  
Acessa a importância dos atributos no melhor modelo encontrado pelo GridSearchCV.

#### **(pd.Series(f_importance, index=features).sort_values(ascending=False))**  
Cria uma série ordenada que mostra a importância de cada atributo no modelo.
