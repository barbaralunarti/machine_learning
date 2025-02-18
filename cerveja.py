# %%
# Importação das bibliotecas essenciais para manipulação de dados, modelagem e visualização
import pandas as pd
from sklearn import tree  # Modelo de Árvore de Decisão
import matplotlib.pyplot as plt  # Biblioteca para visualização de dados

# %%
# Carregamento do conjunto de dados a partir de um arquivo Excel
df = pd.read_excel("dados_cerveja.xlsx")
# Exibição do DataFrame para conferência dos dados
df

# %%
# Definição das variáveis preditoras (features) e da variável alvo (target)
features = ["temperatura", "copo", "espuma", "cor"]
target = "classe"

X = df[features]  # Matriz de variáveis preditoras
y = df[target]    # Vetor da variável resposta

# %%
# Substituição de valores categóricos por valores numéricos para facilitar o treinamento do modelo
X = X.replace({
    "mud": 1, "pint": 0,  # Tipo de copo: 1 para "mud", 0 para "pint"
    "sim": 1, "não": 0,   # Presença de espuma: 1 para "sim", 0 para "não"
    "escura": 1, "clara": 0,  # Cor da cerveja: 1 para "escura", 0 para "clara"
})

# Exibição do DataFrame após a substituição dos valores categóricos
X

# %%
# Instanciação e treinamento do modelo de Árvore de Decisão
# O parâmetro `random_state=42` garante a reprodutibilidade dos resultados
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)

# %%
# Visualização gráfica da Árvore de Decisão treinada
plt.figure(dpi=600)
tree.plot_tree(arvore,
               class_names=arvore.classes_,  # Nome das classes alvo
               feature_names=features,  # Nome das variáveis preditoras
               filled=True)
plt.show()

# %%
# Realização de uma previsão de probabilidades para um novo conjunto de dados
# Exemplo: [-5, 1, 0, 1] representa:
# Temperatura = -5,
# Copo = 1 (mud),
# Espuma = 0 (não),
# Cor = 1 (escura)
probas = arvore.predict_proba([[-5, 1, 0, 1]])[0]
# Exibição das probabilidades preditas para cada classe do modelo
pd.Series(probas, index=arvore.classes_)
