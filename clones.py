# %% 
# Importação das bibliotecas essenciais para análise de dados, engenharia de features e aprendizado de máquina
import pandas as pd
from feature_engine import encoding  # Biblioteca para encoding de variáveis categóricas
from sklearn import tree  # Algoritmo de árvore de decisão
import matplotlib.pyplot as plt  # Biblioteca para visualização de dados

# %% 
# Carregamento do conjunto de dados a partir de um arquivo Parquet
# O dataset contém informações sobre clones e suas características físicas
df = pd.read_parquet("dados_clones.parquet")
df  # Exibe o DataFrame para uma inspeção inicial

# %% 
# Cálculo da média das variáveis 'Estatura(cm)' e 'Massa(em kilos)', agrupadas pelo status do clone
# Essa análise permite entender diferenças estatísticas entre os clones aptos e inaptos
df.groupby(["Status "])[['Estatura(cm)', 'Massa(em kilos)']].mean()

# %% 
# Criação de uma variável booleana para representar o status do clone:
# - `True` para clones aptos
# - `False` para clones inaptos
df['Status_bool'] = df['Status '] == 'Apto'

# %% 
# Análise da relação entre a distância ombro a ombro e a taxa de aptidão dos clones
# O objetivo é verificar se essa característica tem influência sobre a aprovação dos clones
df.groupby(["Distância Ombro a ombro"])['Status_bool'].mean()

# %% 
# Avaliação do impacto do tamanho do crânio na aptidão dos clones
# Essa métrica pode indicar uma possível correlação entre essa característica e o status final
df.groupby(["Tamanho do crânio"])['Status_bool'].mean()

# %% 
# Verificação da relação entre o tamanho dos pés e a aptidão
# Esse tipo de análise pode revelar padrões inesperados no dataset
df.groupby(["Tamanho dos pés"])['Status_bool'].mean()

# %% 
# Análise da taxa de aptidão dos clones sob diferentes Generais Jedi encarregados
# Esse fator pode indicar variações na exigência ou critérios de aprovação entre os Jedi
df.groupby(["General Jedi encarregado"])['Status_bool'].mean()

# %% 
# Definição das variáveis preditoras (features) e categóricas
# As características físicas dos clones serão usadas para prever sua aptidão
features = [
    "Estatura(cm)",
    "Massa(em kilos)",
    "Distância Ombro a ombro",
    "Tamanho do crânio",
    "Tamanho dos pés",
]

# Identificação das variáveis categóricas que precisam ser codificadas
cat_features = ["Distância Ombro a ombro",
                "Tamanho do crânio",
                "Tamanho dos pés"]

# Seleção do subconjunto de variáveis do DataFrame para modelagem
X = df[features]

# %% 
# Aplicação de codificação one-hot nas variáveis categóricas
# Essa transformação é necessária para que o modelo de árvore de decisão consiga processar os dados corretamente
onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)
X = onehot.transform(X)

# %% 
# Criação e treinamento de um modelo de árvore de decisão para classificar os clones
# O hiperparâmetro `max_depth=4` limita a profundidade da árvore para evitar overfitting
arvore = tree.DecisionTreeClassifier(max_depth=4)
arvore.fit(X, df["Status "])

# %% 
# Visualização da árvore de decisão gerada
# O gráfico mostra como o modelo toma decisões com base nos atributos dos clones
plt.figure(dpi=600)  # Aumenta a resolução para melhor qualidade na visualização
tree.plot_tree(arvore,
               class_names=arvore.classes_,  # Nome das classes ('Apto' e 'Inapto')
               feature_names=X.columns,  # Nome das características utilizadas
               filled=True)  # Preenchimento das caixas para facilitar a interpretação
