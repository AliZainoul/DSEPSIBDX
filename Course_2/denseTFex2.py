# Objectif de l'exemple :
# L'objectif de cet exemple est de construire, entraîner et évaluer un réseau neuronal dense à l'aide de TensorFlow pour la classification d'images du jeu de données MNIST.

# Importation des bibliothèques
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Documentation :
# - TensorFlow : https://www.tensorflow.org/
# - scikit-learn : https://scikit-learn.org/stable/documentation.html
# - MNIST Dataset : https://www.tensorflow.org/datasets/catalog/mnist
# - TensorFlow Keras Layers : https://www.tensorflow.org/api_docs/python/tf/keras/layers
# - TensorFlow Keras Models : https://www.tensorflow.org/api_docs/python/tf/keras/models

# Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation des images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Conversion des étiquettes en catégories
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Division des données en ensembles d'entraînement et de test
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Construction du modèle TensorFlow
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Aplatit l'image en un vecteur
    layers.Dense(128, activation='relu'),   # Couche dense avec 128 neurones et fonction d'activation ReLU
    layers.Dense(10, activation='softmax')  # Couche dense de sortie avec 10 neurones (classes) et fonction d'activation softmax
])

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Documentation :
# - tf.keras.Sequential : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# - tf.keras.layers.Flatten : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
# - tf.keras.layers.Dense : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
# - Optimizers in TensorFlow : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# - Losses in TensorFlow : https://www.tensorflow.org/api_docs/python/tf/keras/losses

# Entraînement du modèle
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=2)

# Documentation :
# - Model training in TensorFlow : https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
#   - x (array-like) : Données d'entraînement.
#   - y (array-like) : Étiquettes d'entraînement.
#   - epochs (int) : Nombre d'époques d'entraînement.

# Évaluation du modèle
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy (TensorFlow Neural Network): {test_accuracy}")

# Documentation :
# - Model evaluation in TensorFlow : https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
#   - x (array-like) : Données de test.
#   - y (array-like) : Étiquettes de test.
