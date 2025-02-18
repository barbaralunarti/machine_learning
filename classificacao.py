# %%
# Importação de Bibliotecas
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import naive_bayes
import matplotlib.pyplot as plt

# %%
# Carregamento dos Dados
df = pd.read_excel("dados_cerveja_nota.xlsx")

# %%
# Criação da Variável Target
# Uma nova coluna binária (Aprovado) é criada para representar a variável target
df["Aprovado"] = df["nota"] >= 5

# %%
# Definição de Features e Target
features = ["cerveja"] # variáveis independentes
target = ["Aprovado"] # variável dependente

# %%
# Regressão Logística
# Treinamento de um modelo de Regressão Logística para prever a aprovação das cervejas
# A penalidade é desativada (`penalty=None`), e o modelo inclui um intercepto (`fit_intercept=True`)
reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)
reg.fit(df[features], df[target])

# Realiza previsões com o modelo treinado
reg_predict = reg.predict(df[features])

# Avaliação da Regressão Logística com métricas de desempenho
reg_acc = metrics.accuracy_score(df[target], reg_predict)
print("Acurácia Regressão Logística: ", reg_acc)

reg_prec = metrics.precision_score(df[target], reg_predict)
print("Precisão Regressão Logística: ", reg_prec)

reg_recall = metrics.recall_score(df[target], reg_predict)
print("Recall Regressão Logística: ", reg_recall)

# Obtém as probabilidades preditas pelo modelo
reg_proba = reg.predict_proba(df[features])
print("Probabilidade Regressão Logística: ", reg_proba)

# Criação da matriz de confusão para melhor visualização dos erros e acertos do modelo
reg_conf = metrics.confusion_matrix(df[target], reg_predict)
reg_conf = pd.DataFrame(reg_conf, index=['False', 'True'], columns=['False', 'True'])
print(reg_conf)

# %%
# Ajuste de um limiar de decisão para 80% de probabilidade na Regressão Logística
# Apenas previsões com probabilidade superior a 0.8 serão classificadas como aprovadas
reg_proba = reg.predict_proba(df[features])[:,1]
reg_predict_proba = reg_proba > 0.8 # ponto de corte
print("Probabilidade para previsões de classe Regressão Logística: ", reg_predict_proba)

# Avaliação do modelo após a alteração do ponto de corte
reg_acc = metrics.accuracy_score(df[target], reg_predict_proba)
print("Acurácia Regressão Logística: ", reg_acc)

reg_prec = metrics.precision_score(df[target], reg_predict_proba)
print("Precisão Regressão Logística: ", reg_prec)

reg_recall = metrics.recall_score(df[target], reg_predict_proba)
print("Recall Regressão Logística: ", reg_recall)

# Matriz de confusão após ajuste do ponto de corte.
reg_conf = metrics.confusion_matrix(df[target], reg_predict_proba)
reg_conf = pd.DataFrame(reg_conf,
                          index=['False', 'True'],
                          columns=['False', 'True'])

# %%
# Treinamento de um modelo de Árvore de Decisão com profundidade máxima de 2
# Modelos mais rasos evitam overfitting e tornam a interpretação mais simples
arvore = tree.DecisionTreeClassifier(max_depth=2)
arvore.fit(df[features], df[target])
arvore_predict = arvore.predict(df[features])

# Avaliação do modelo de Árvore de Decisão
arvore_acc = metrics.accuracy_score(df[target], arvore_predict)
print("Acurácia Árvore: ", arvore_acc)

arvore_prec = metrics.precision_score(df[target], arvore_predict)
print("Precisão Árvore: ", arvore_prec)

arvore_recall = metrics.recall_score(df[target], arvore_predict)
print("Recall Árvore: ", arvore_recall)

# Probabilidades preditas pelo modelo de Árvore de Decisão
arvore_proba = arvore.predict_proba(df[features])
print("Probabilidade Árvore: ", arvore_proba)

# # Matriz de confusão para avaliação detalhada dos resultados do modelo
arvore_conf = metrics.confusion_matrix(df[target], arvore_predict)
arvore_conf = pd.DataFrame(arvore_conf,
                           index=['False', 'True'],
                           columns=['False', 'True'])
print(arvore_conf)

# %% 
# Ajuste do limiar de decisão para 80% na Árvore de Decisão
arvore_proba = arvore.predict_proba(df[features])[:,1]
arvore_predict_proba = arvore_proba > 0.8  
print("Probabilidade para previsões de classe Árvore: ", arvore_predict_proba)

# Avaliação do modelo com o novo ponto de corte
arvore_acc = metrics.accuracy_score(df[target], arvore_predict_proba)
print("Acurácia Árvore: ", arvore_acc)

arvore_prec = metrics.precision_score(df[target], arvore_predict_proba)
print("Precisão Árvore: ", arvore_prec)

arvore_recall = metrics.recall_score(df[target], arvore_predict_proba)
print("Recall Árvore: ", arvore_recall)

# Matriz de confusão após ajuste do ponto de corte
arvore_conf = metrics.confusion_matrix(df[target], arvore_predict_proba)
arvore_conf = pd.DataFrame(arvore_conf, index=['False', 'True'], columns=['False', 'True'])

# %% 
# Treinamento de um modelo Naive Bayes Gaussiano
# Esse modelo assume que as features seguem uma distribuição normal
nb = naive_bayes.GaussianNB()
nb.fit(df[features], df[target])

# Realiza previsões com o modelo Naive Bayes treinado
nb_predict = nb.predict(df[features])

# Avaliação do desempenho do modelo Naive Bayes
nb_acc = metrics.accuracy_score(df[target], nb_predict)
print("Acurácia Naive Bayes: ", nb_acc)

nb_prec = metrics.precision_score(df[target], nb_predict)
print("Precisão Naive Bayes: ", nb_prec)

nb_recall = metrics.recall_score(df[target], nb_predict)
print("Recall Naive Bayes: ", nb_recall)

# Probabilidades preditas pelo modelo
nb_proba = nb.predict_proba(df[features])
print("Probabilidade Naive Bayes: ", nb_proba)

# Matriz de confusão para avaliação dos resultados
nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf = pd.DataFrame(nb_conf, index=['False', 'True'], columns=['False', 'True'])
print(nb_conf)

# %% 
# Ajuste do limiar de decisão para 80% no modelo Naive Bayes
nb_proba = nb.predict_proba(df[features])[:,1]
nb_predict_proba = nb_proba > 0.8  
print("Probabilidade para previsões de classe Naive Bayes: ", nb_predict_proba)

# Avaliação do modelo com o novo ponto de corte
nb_acc = metrics.accuracy_score(df[target], nb_predict_proba)
print("Acurácia Naive Bayes: ", nb_acc)

nb_prec = metrics.precision_score(df[target], nb_predict_proba)
print("Precisão Naive Bayes: ", nb_prec)

nb_recall = metrics.recall_score(df[target], nb_predict_proba)
print("Recall Naive Bayes: ", nb_recall)

# Matriz de confusão após ajuste do ponto de corte
nb_conf = metrics.confusion_matrix(df[target], nb_predict_proba)
nb_conf = pd.DataFrame(nb_conf, index=['False', 'True'], columns=['False', 'True'])

# %% 
# Salvando as probabilidades preditas pelo modelo Naive Bayes no DataFrame e exportando para Excel
df['prob_nb'] = nb_proba
df.to_excel("dados_cerveja_nota_predict.xlsx", index=False)

# %% 
# Construção da curva ROC para o modelo Naive Bayes
roc_curve_nb = metrics.roc_curve(df[target], nb_proba)
plt.plot(roc_curve_nb[0], roc_curve_nb[1])
plt.grid(True)
plt.plot([0,1], [0,1], '--')  # Linha de referência para um modelo aleatório
plt.show()

# %% 
# Cálculo da AUC (Área sob a Curva ROC), indicando a performance do modelo
roc_auc = metrics.roc_auc_score(df[target], nb_proba)
print(roc_auc)