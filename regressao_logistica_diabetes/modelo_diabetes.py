# Modelo de previsão de diabetes usando Regressão Logística

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Carregando o dataset
df = pd.read_csv('diabetes.csv')


# Visualizando os dados
print("Primeiras linhas do dataset:")
print(df.head())


print("\nResumo estatístico:")
print(df.describe())


# Gráfico Idade x Número de gravidezes
plt.plot(df['Age'], df['Pregnancies'], 'o')

plt.xlabel('Idade')
plt.ylabel('Gravidezes')

plt.title('Idade x Gravidez')

plt.show()


# Separando variável alvo (resultado)
y = df['Outcome']


# Separando atributos de entrada
x = df.drop('Outcome', axis=1)


print("\nDados de entrada:")
print(x.head())


# Separando treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# Criando o modelo
modelo = LogisticRegression(
    max_iter=5000
)


# Treinando o modelo
modelo.fit(
    x_treino,
    y_treino
)


# Fazendo previsões
y_previsto = modelo.predict(x_teste)


print("\nPrimeiras previsões:")
print(y_previsto[:10])


# Avaliando o modelo
acuracia = accuracy_score(
    y_teste,
    y_previsto
)


print("\nAcurácia do modelo:")
print(f"{acuracia * 100:.2f}%")