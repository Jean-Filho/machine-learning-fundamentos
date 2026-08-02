import tensorflow as tf
from tensorflow import keras
import numpy as np


# ==========================
# Carregando o dataset Fashion MNIST
# ==========================

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# ==========================
# Verificando os dados
# ==========================

print("Formato das imagens de treino:", train_images.shape)
print("Quantidade de rótulos de treino:", len(train_labels))
print("Classes existentes:", np.unique(train_labels))

print("Formato das imagens de teste:", test_images.shape)
print("Quantidade de rótulos de teste:", len(test_labels))


# ==========================
# Normalização dos pixels
# ==========================

train_images = train_images / 255.0
test_images = test_images / 255.0


# ==========================
# Criando a rede neural
# ==========================

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# Mostra a arquitetura da rede

model.summary()


# ==========================
# Configurando o treinamento
# ==========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================
# Treinando o modelo
# ==========================

model.fit(
    train_images,
    train_labels,
    epochs=5
)


# ==========================
# Avaliando o modelo
# ==========================

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels
)

print("\nAcurácia no teste:", test_acc)


# ==========================
# Fazendo previsões
# ==========================

predictions = model.predict(test_images)


# Primeira previsão

print("\nProbabilidades da primeira imagem:")
print(predictions[0])


print("\nClasse prevista:")
print(np.argmax(predictions[0]))


print("\nClasse real:")
print(test_labels[0])


# ==========================
# Testando uma única imagem
# ==========================

img = test_images[0]

print("\nFormato da imagem:")
print(img.shape)


# Adiciona uma dimensão para a rede receber

img = np.expand_dims(img, 0)

print("\nNovo formato:")
print(img.shape)


prediction_single = model.predict(img)


print("\nResultado da previsão:")
print(prediction_single)


print("\nClasse prevista da imagem:")
print(np.argmax(prediction_single[0]))