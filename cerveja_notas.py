# %%

# Importação das bibliotecas essenciais
import pandas as pd  # Para manipulação e análise de dados
import matplotlib.pyplot as plt  # Para visualização de dados
from sklearn import linear_model  # Para modelos de regressão linear
from sklearn import tree  # Para árvores de decisão

# %%

# Carregamento dos dados a partir de um arquivo Excel
df = pd.read_excel("dados_cerveja_nota.xlsx")  # Carrega os dados em um DataFrame

# %%

# Visualização da relação entre as variáveis 'cerveja' e 'nota'
plt.plot(df["cerveja"], df["nota"], 'o')  # Gráfico de dispersão com marcadores circulares
plt.grid(True)  # Adiciona uma grade para melhorar a legibilidade
plt.title("Relação Nota vs Cerveja")  # Título do gráfico
plt.ylim(0, 11)  # Define os limites do eixo Y (0 a 11)
plt.xlim(0, 11)  # Define os limites do eixo X (0 a 11)
plt.xlabel("Cerveja")  # Rótulo do eixo X
plt.ylabel("Nota")  # Rótulo do eixo Y
plt.show()  # Exibe o gráfico

# %%

# Criação e treinamento do modelo de regressão linear
reg = linear_model.LinearRegression()  # Instância o modelo de regressão linear
reg.fit(df[["cerveja"]], df["nota"])  # Treina o modelo com os dados

# %%

# Extração e exibição dos coeficientes da regressão linear
a, b = [reg.intercept_], reg.coef_[0]  # Coeficientes: intercepto (a) e inclinação (b)
print(f"a={a}; b={b}")  # Exibe os coeficientes de forma formatada

# %%

# Visualização da regressão linear sobreposta aos dados reais
X = df[["cerveja"]].drop_duplicates()  # Valores únicos de 'cerveja' para previsão
y_estimado = reg.predict(X)  # Previsões do modelo de regressão linear

plt.plot(df["cerveja"], df["nota"], 'o')  # Plota os dados reais
plt.plot(X, y_estimado, '-')  # Plota a reta de regressão linear
plt.grid(True)  # Adiciona uma grade
plt.title("Relação Nota vs Cerveja")  # Título do gráfico
plt.ylim(0, 11)  # Limites do eixo Y
plt.xlim(0, 11)  # Limites do eixo X
plt.xlabel("Cerveja")  # Rótulo do eixo X
plt.ylabel("Nota")  # Rótulo do eixo Y
plt.show()  # Exibe o gráfico

# %%

# Criação e treinamento de uma árvore de decisão para regressão
arvore = tree.DecisionTreeRegressor(max_depth=2)  # Instância a árvore com profundidade máxima 2
arvore.fit(df[["cerveja"]], df["nota"])  # Treina a árvore com os dados

# Previsões da árvore de decisão
y_estimado_arvore = arvore.predict(X)  # Gera previsões para os valores únicos de 'cerveja'

# Visualização comparativa: dados reais, regressão linear e árvore de decisão
plt.plot(df["cerveja"], df["nota"], 'o')  # Plota os dados reais
plt.plot(X, y_estimado, '-')  # Plota a regressão linear
plt.plot(X, y_estimado_arvore, '-')  # Plota a árvore de decisão
plt.grid(True)  # Adiciona uma grade
plt.title("Relação Nota vs Cerveja")  # Título do gráfico
plt.ylim(0, 11)  # Limites do eixo Y
plt.xlim(0, 11)  # Limites do eixo X
plt.xlabel("Cerveja")  # Rótulo do eixo X
plt.ylabel("Nota")  # Rótulo do eixo Y
plt.legend(["Pontos", "Regressão Linear", "Árvore"])  # Legenda para identificar as curvas
plt.show()  # Exibe o gráfico

# %%