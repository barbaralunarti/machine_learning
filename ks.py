# %%

# Importação das bibliotecas essenciais
import pandas as pd  # Para manipulação e análise de dados

from sklearn import model_selection  # Para divisão de dados e validação cruzada
from sklearn import ensemble  # Para modelos de ensemble, como RandomForest
from sklearn import pipeline  # Para criar pipelines de processamento de dados

from feature_engine import imputation  # Para técnicas avançadas de imputação de dados

import scikitplot as skplt  # Para visualizações de métricas de modelos de machine learning

# %%
# Carregamento dos dados a partir de um arquivo CSV
df = pd.read_csv("../data/dados_pontos.csv", sep=";")  # Carrega os dados em um DataFrame

# %%
# Definição das variáveis independentes (features) e dependente (target)
features = df.columns[3:-1]  # Seleciona as colunas de features (da 4ª à penúltima)
target = "flActive"  # Define a coluna target (variável dependente)

# %%

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df[features],  # Variáveis independentes
    df[target],  # Variável dependente
    random_state=42,  # Semente para reprodutibilidade
    test_size=0.2,  # 20% dos dados para teste
    stratify=df[target]  # Mantém a proporção da variável target em treino e teste
)

# Exibição das taxas de resposta (média da variável target) nos conjuntos de treino e teste
print("Tx. resposta Train:", y_train.mean())  # Taxa de resposta no conjunto de treino
print("Tx. resposta Test:", y_test.mean())  # Taxa de resposta no conjunto de teste

# %%
# Verificação de valores faltantes (NaN) no conjunto de treino
X_train.isna().sum()  # Conta a quantidade de valores NaN em cada coluna

# %%
# Configuração da imputação de valores faltantes
imput_max = imputation.ArbitraryNumberImputer(
    arbitrary_number=999,  # Valor arbitrário para substituir NaNs
    variables=['avgRecorrencia']  # Coluna específica para imputação
)

# Configuração do modelo RandomForestClassifier
clf = ensemble.RandomForestClassifier(random_state=42)  # Modelo de Random Forest

# Definição dos hiperparâmetros para otimização
params = {
    "n_estimators": [200, 300, 400, 500],  # Número de árvores na floresta
    "min_samples_leaf": [10, 20, 50, 100]  # Número mínimo de amostras nas folhas
}

# Configuração da busca em grade (GridSearchCV) para otimização de hiperparâmetros
grid = model_selection.GridSearchCV(
    clf,  # Modelo a ser otimizado
    param_grid=params,  # Hiperparâmetros a serem testados
    scoring='roc_auc',  # Métrica de avaliação (área sob a curva ROC)
    n_jobs=-1  # Uso de todos os núcleos do processador para paralelização
)

# Criação de um pipeline para integrar a imputação e o modelo
model = pipeline.Pipeline([
    ('imput', imput_max),  # Etapa de imputação de valores faltantes
    ('model', grid)  # Etapa de modelagem com GridSearchCV
])

# Treinamento do modelo
model.fit(X_train, y_train)  # Ajusta o pipeline aos dados de treino

# %%

# Geração de probabilidades para o conjunto de teste
y_test_proba = model.predict_proba(X_test)  # Probabilidades de cada classe
y_test_proba  # Exibe as probabilidades geradas

# %%

# Criação de um DataFrame com as probabilidades e a variável target real
df = pd.DataFrame({
    "flActive": y_test,  # Variável target real
    "proba_modelo": y_test_proba[:, 1]  # Probabilidades da classe positiva
})

# Salvamento dos resultados em um arquivo Excel
df.to_excel("dados_ks.xlsx", index=False)  # Salva sem incluir índices

# %%
# Visualização da estatística KS (Kolmogorov-Smirnov)
skplt.metrics.plot_ks_statistic(y_test, y_test_proba)  # Plota a curva KS