# Análise Exploratória de Dados - Frutas

import pandas as pd
import matplotlib.pyplot as plt


# Carregando o conjunto de dados
frutas = pd.read_table(
    'dados_frutas.txt',
    sep='\t'
)


# Visualizando os dados
print(frutas)

print("\nQuantidade de linhas e colunas:")
print(frutas.shape)


print("\nPrimeiras 5 linhas:")
print(frutas.head(5))


# -----------------------------
# Análise Estatística
# -----------------------------

print("\nResumo estatístico:")
print(frutas.describe())


print("\nEstatísticas da massa:")
print(frutas.describe()['mass'])


print("\nMenor massa encontrada:")
print(frutas.describe()['mass']['min'])


print("\nColuna massa:")
print(frutas['mass'])


print("\nMassa e pontuação de cor:")
print(frutas[['mass', 'color_score']])


print("\nLinhas 10 até 14:")
print(frutas[10:15])


i = 15

print("\nCinco linhas antes do índice 15:")
print(frutas[i-5:i])


print("\nCinco linhas depois do índice 15:")
print(frutas[i:i+5])


print("\nMassa e cor das linhas 15 até 19:")
print(frutas[['mass', 'color_score']][i:i+5])


# -----------------------------
# Análise e Visualização
# -----------------------------


# Frequência das frutas
freq = frutas['fruit_name'].value_counts()

print("\nQuantidade de cada fruta:")
print(freq)


# Gráfico de barras
freq.plot(kind='bar')

plt.title("Quantidade de frutas")
plt.xlabel("Fruta")
plt.ylabel("Quantidade")

plt.show()



# -----------------------------
# Filtrando maçãs
# -----------------------------


macas = frutas['fruit_name'] == 'apple'


print("\nTodas as maçãs:")
print(frutas[macas])


# Maçãs pesadas
pesadas = frutas['mass'] > 175


print("\nMaçãs com massa maior que 175:")
print(frutas[macas & pesadas])



# -----------------------------
# Gráfico de maçãs pesadas
# -----------------------------


X1 = frutas[macas & pesadas]['width']
X2 = frutas[macas & pesadas]['height']


plt.scatter(X1, X2)

plt.xlabel("Comprimento")
plt.ylabel("Altura")

plt.title("Maçãs pesadas")

plt.show()



# -----------------------------
# Gráfico de todas as frutas
# -----------------------------


X1 = frutas['width']
X2 = frutas['height']


plt.scatter(X1, X2)

plt.xlabel("Comprimento")
plt.ylabel("Altura")

plt.title("Altura x Comprimento das frutas")

plt.show()



# -----------------------------
# Gráfico colorido pelo tipo de fruta
# -----------------------------


y = frutas['fruit_label']


plt.scatter(
    X1,
    X2,
    c=y
)


plt.xlabel("Comprimento")
plt.ylabel("Altura")

plt.title("Classificação das frutas")

plt.show()



