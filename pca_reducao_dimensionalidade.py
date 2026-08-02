# Principal Component Analysis (PCA)

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


# =====================================
# CARREGANDO OS DADOS
# =====================================

X, y = fetch_california_housing(return_X_y=True)

print("Dimensão original dos dados:")
print(X.shape)


# =====================================
# APLICAÇÃO DO PCA
# =====================================

# Reduzindo para 2 componentes principais
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print("\nDimensão após PCA:")
print(X_pca.shape)


# =====================================
# VISUALIZAÇÃO DOS DADOS
# =====================================

plt.scatter(X_pca[:, 0], X_pca[:, 1])

plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")

plt.title("Dados após PCA")

plt.show()


# =====================================
# TREINAMENTO DO MODELO
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)


modelo = LinearRegression()

modelo.fit(X_train, y_train)


# Avaliação do modelo
resultado = modelo.score(X_test, y_test)

print("\nResultado do modelo R²:")
print(resultado)