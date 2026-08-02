# ==============================
# Introdução às Redes Neurais
# ==============================

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing


# ==============================
# Carregando os dados
# ==============================

housing = fetch_california_housing()

X = housing.data
y = housing.target


# ==============================
# Separando dados de treino e teste
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ==============================
# Normalização dos dados
# ==============================

mm = MinMaxScaler()

X_train = mm.fit_transform(X_train)

X_test = mm.transform(X_test)


# ==============================
# Criando a Rede Neural
# ==============================

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 100, 50, 50),
    max_iter=1000,
    random_state=42
)


# ==============================
# Treinando o modelo
# ==============================

mlp.fit(X_train, y_train)


# ==============================
# Avaliando o modelo
# ==============================

resultado = mlp.score(X_test, y_test)

print("R² da Rede Neural:", resultado)