# Classificação usando SVM (Support Vector Machine)

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# ==============================
# MODELO COM DADOS DE CÂNCER
# ==============================

print("Modelo - Câncer de Mama")

# Carrega os dados
X, y = load_breast_cancer(return_X_y=True)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cria o modelo SVM
svm = SVC(kernel="linear", C=1.0)

# Treina o modelo
svm.fit(X_train, y_train)

# Mostra a precisão
resultado = svm.score(X_test, y_test)

print("Precisão:", resultado)


# ==============================
# MODELO COM DADOS IRIS
# ==============================

print("\nModelo - Flores Iris")

# Carrega os dados
X, y = load_iris(return_X_y=True)

# Divide os dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# SVM Linear
svm_linear = SVC(kernel="linear")

svm_linear.fit(X_train, y_train)

print("Kernel Linear:", svm_linear.score(X_test, y_test))


# SVM Polinomial
svm_poly = SVC(kernel="poly", degree=3)

svm_poly.fit(X_train, y_train)

print("Kernel Polinomial:", svm_poly.score(X_test, y_test))


# SVM RBF
svm_rbf = SVC(kernel="rbf")

svm_rbf.fit(X_train, y_train)

print("Kernel RBF:", svm_rbf.score(X_test, y_test))