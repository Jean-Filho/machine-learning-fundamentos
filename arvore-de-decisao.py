from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    fetch_california_housing
)
from sklearn.model_selection import train_test_split

# ==========================
# Classificação - Iris
# ==========================

dt = DecisionTreeClassifier(random_state=42)

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

dt.fit(X_train, y_train)

print("Acurácia (Iris):", dt.score(X_test, y_test))

# ==========================
# Regressão - California Housing
# ==========================

dt = DecisionTreeRegressor(random_state=42)

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

dt.fit(X_train, y_train)

print("R² (California Housing):", dt.score(X_test, y_test))

# ==========================
# Classificação com profundidade máxima
# ==========================

dt = DecisionTreeClassifier(max_depth=3, random_state=42)

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

dt.fit(X_train, y_train)

print("Acurácia (max_depth=3):", dt.score(X_test, y_test))

# ==========================
# Importância das características
# ==========================

X, y = load_breast_cancer(return_X_y=True)

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X, y)

print("\nFormato do conjunto de dados:")
print(X.shape)

print("\nImportância das características:")
print(dt.feature_importances_)