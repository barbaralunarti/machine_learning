# %% 
# Importação das bibliotecas essenciais para análise de dados, aprendizado de máquina e visualização
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

# %% 
# Carregamento dos dados a partir de um arquivo Excel
# O DataFrame resultante contém características de diferentes frutas
df = pd.read_excel("dados_frutas.xlsx")
df

# %% 
# Aplicação de filtros para identificar frutas que possuem TODAS as seguintes características:
# - Arredondada (1 para sim, 0 para não)
# - Suculenta (1 para sim, 0 para não)
# - Vermelha (1 para sim, 0 para não)
# - Doce (1 para sim, 0 para não)
filtro_redonda = df["Arredondada"] == 1
filtro_suculenta = df["Suculenta"] == 1
filtro_vermelha = df["Vermelha"] == 1
filtro_doce = df["Doce"] == 1

# Exibe apenas as frutas que atendem a todos os critérios definidos
df[filtro_redonda & filtro_suculenta & filtro_vermelha & filtro_doce]

# %% 
# Definição das variáveis preditoras (features) e do alvo (target)
# O modelo usa as características da fruta para prever sua classificação
features = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
target = "Fruta"

# X contém os atributos das frutas e y contém suas classificações
X = df[features]
y = df[target]

# %% 
# Criação e treinamento de um modelo de árvore de decisão
# O hiperparâmetro `random_state=42` garante reprodutibilidade nos resultados
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)  # Ajusta o modelo aos dados

# %% 
# Visualização da árvore de decisão gerada
# O gráfico mostra como o modelo toma decisões com base nos atributos fornecidos
plt.figure(dpi=600)  # Aumenta a resolução para melhor qualidade na visualização
tree.plot_tree(arvore,
               class_names=arvore.classes_,  # Nome das classes (frutas)
               feature_names=features,  # Nome das características utilizadas
               filled=True)  # Preenchimento das caixas para facilitar a interpretação

# %% 
# Predição da probabilidade de uma fruta ser classificada como cada uma das possíveis categorias.
# O modelo recebe um conjunto de atributos [1,1,1,1], indicando uma fruta
# Retorna as probabilidades de pertencimento a cada classe
probas = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(probas, index=arvore.classes_)  # Exibe as probabilidades em formato de série Pandas
