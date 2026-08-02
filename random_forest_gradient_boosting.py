# Random Forest e Gradient Boosting

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split


# =====================================
# RANDOM FOREST - CLASSIFICAÇÃO
# =====================================

print("Random Forest - Classificação")

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(X_train, y_train)

print("Acurácia:", rf_classifier.score(X_test, y_test))


# =====================================
# RANDOM FOREST - REGRESSÃO
# =====================================

print("\nRandom Forest - Regressão")

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_regressor.fit(X_train, y_train)

print("R²:", rf_regressor.score(X_test, y_test))


# =====================================
# GRADIENT BOOSTING - CLASSIFICAÇÃO
# =====================================

print("\nGradient Boosting - Classificação")

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gb_classifier = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    random_state=42
)

gb_classifier.fit(X_train, y_train)

print("Acurácia:", gb_classifier.score(X_test, y_test))


# =====================================
# GRADIENT BOOSTING - REGRESSÃO
# =====================================

print("\nGradient Boosting - Regressão")

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gb_regressor = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=3,
    random_state=42
)

gb_regressor.fit(X_train, y_train)

print("R²:", gb_regressor.score(X_test, y_test))