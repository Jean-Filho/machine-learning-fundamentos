#Processamento de Linguagem Natural

import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================
# PARTE 1 - NLTK
# ==========================

# Baixar recursos do NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# Lista de palavras sem importância
stopwords = nltk.corpus.stopwords.words('portuguese')

print("Stopwords:")
print(stopwords)


# Frase para análise

frase = "Eu dirijo devagar porque nós queremos ver os animais."


# Separando a frase em palavras

tokens = word_tokenize(frase, language="portuguese")

print("\nTokens:")
print(tokens)


# Removendo stopwords

print("\nPalavras importantes:")

for palavra in tokens:
    if palavra.lower() not in stopwords:
        print(palavra)



# ==========================
# PARTE 2 - TF-IDF
# ==========================


texto1 = "A matemática é muito importante para compreendermos como a natureza funciona"


# Criando o modelo TF-IDF

tf_idf = TfidfVectorizer()


# Transformando texto em números

vetor = tf_idf.fit_transform([texto1])


print("\nVetor TF-IDF:")
print(vetor)


# Transformando em matriz normal

vetor = vetor.todense()


# Pegando nomes das palavras

nomes = tf_idf.get_feature_names_out()


# Criando tabela

df = pd.DataFrame(
    vetor,
    columns=nomes
)


print("\nTabela TF-IDF texto 1:")
print(df)



# ==========================
# Segundo texto
# ==========================


texto2 = (
    "A matemática é incrível, quanto mais estudo matemática, "
    "mais eu consigo aprender matemática"
)


tf_idf = TfidfVectorizer()


vetor2 = tf_idf.fit_transform([texto2])


vetor2 = vetor2.todense()


nomes = tf_idf.get_feature_names_out()


df2 = pd.DataFrame(
    vetor2,
    columns=nomes
)


print("\nTabela TF-IDF texto 2:")
print(df2)