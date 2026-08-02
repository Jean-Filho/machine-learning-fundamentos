# Tratamento e análise de dados de frutas

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Carregando os dados
data = pd.read_table(
    'dataset_frutas.txt',
    na_values=['.', '?']
)

print("Dados iniciais:")
print(data)


# Verificando valores ausentes
print("\nQuantidade de valores ausentes:")
print(data.isnull().sum())


print("\nPercentual de valores ausentes:")
print((data.isnull().sum() / data.shape[0]) * 100)


# Preenchendo valores ausentes das colunas numéricas pela média
colunas_numericas = data.select_dtypes(include=['number']).columns

data[colunas_numericas] = data[colunas_numericas].fillna(
    data[colunas_numericas].mean()
)


# Preenchendo valores ausentes da coluna categórica
if 'fruit_subtype' in data.columns:
    data['fruit_subtype'] = data['fruit_subtype'].fillna(
        data['fruit_subtype'].mode()[0]
    )


print("\nDados após tratamento:")
print(data)


# Selecionando algumas colunas
dados = data[
    ['mass', 'width', 'height', 'color_score']
]


print("\nResumo estatístico:")
print(dados.describe())


# Normalização dos dados
scaler = MinMaxScaler()

dados_normalizados = scaler.fit_transform(dados)


print("\nDados normalizados:")
print(dados_normalizados)


# -----------------------------
# Análise de maçãs
# -----------------------------

macas = data[data['fruit_name'] == 'apple']


print("\nEstatística das maçãs:")
print(macas.describe())


# Calculando média e desvio padrão da massa
estatistica = macas['mass'].describe()


# Maçãs acima de 2 desvios padrões
print("\nMaçãs com massa muito alta:")
print(
    macas[
        macas['mass'] > estatistica['mean'] + (estatistica['std'] * 2)
    ]
)


# Maçãs abaixo de 2 desvios padrões
print("\nMaçãs com massa muito baixa:")
print(
    macas[
        macas['mass'] < estatistica['mean'] - (estatistica['std'] * 2)
    ]
)


print("\nDesvio padrão da massa das maçãs:")
print(estatistica['std'])