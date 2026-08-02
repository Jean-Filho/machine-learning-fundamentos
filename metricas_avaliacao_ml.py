# Métricas de Avaliação e Validação de Modelos

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


# ============================
# CLASSIFICADOR DUMMY
# ============================

print("Classificador Dummy - Iris")

# Carrega dados Iris
X, y = load_iris(return_X_y=True)

# Cria modelo Dummy
dc = DummyClassifier(strategy="stratified")

# Treina
dc.fit(X, y)

# Avaliação
print("Acurácia:", dc.score(X, y))


# ============================
# REGRESSOR DUMMY
# ============================

print("\nRegressor Dummy")

# Criando dados de exemplo para regressão
# load_boston foi removido das versões atuais do sklearn
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)

dr = DummyRegressor(strategy="mean")

dr.fit(X, y)

print("Resultado:", dr.score(X, y))


# ============================
# MATRIZ DE CONFUSÃO
# ============================

print("\nMatriz de Confusão")

X, y = load_breast_cancer(return_X_y=True)

dc = DummyClassifier(strategy="stratified")

dc.fit(X, y)

previsoes = dc.predict(X)

print(confusion_matrix(y, previsoes))


# Relatório de classificação
print("\nRelatório de Classificação")

print(classification_report(y, previsoes))


# ============================
# VALIDAÇÃO CRUZADA
# ============================

print("\nValidação Cruzada")

X, y = load_iris(return_X_y=True)

dc = DummyClassifier(strategy="stratified")

resultados = cross_val_score(dc, X, y, cv=3)

print("Resultados:", resultados)

print("Média:", resultados.mean())