# Objectif de l'exemple :
# L'objectif de cet exemple est de construire, entraîner et évaluer un réseau neuronal dense à l'aide de TensorFlow
# pour effectuer la classification des données Iris en trois classes.

# Importation des bibliothèques
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Documentation :
# - TensorFlow : https://www.tensorflow.org/
# - scikit-learn : https://scikit-learn.org/stable/documentation.html
# - NumPy : https://numpy.org/doc/stable/
# - accuracy_score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

# Chargement du jeu de données Iris
iris = load_iris()
X, y = iris.data, iris.target

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    # Données d'entrée (features)
    X,
    # Étiquettes (classes)
    y,
    # Taille de l'ensemble de test (20% dans ce cas)
    test_size=0.2,
    # Graine pour la reproductibilité des résultats
    random_state=42
)

# Normalisation des données
scaler = StandardScaler()

# Fit et transform sur l'ensemble d'entraînement
X_train_scaled = scaler.fit_transform(
    # Données d'entraînement à normaliser
    X_train
)

# Transform sur l'ensemble de test (utilisation des paramètres appris sur l'ensemble d'entraînement)
X_test_scaled = scaler.transform(
    # Données de test à normaliser
    X_test
)

# Construction du modèle TensorFlow
model = tf.keras.Sequential([
    # Couche dense avec 3 neurones, fonction d'activation softmax et input_shape défini par le nombre de features
    tf.keras.layers.Dense(
        # Dimensionnalité de l'espace de sortie (nombre de classes à prédire)
        units=3,
        # Fonction d'activation pour introduire la non-linéarité
        activation='softmax',
        # Forme de l'input (nombre de features)
        input_shape=(X_train.shape[1],)
    )
])

# Documentation :
# - tf.keras.Sequential : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
#   - layers (list) : Liste des couches du modèle.
# - tf.keras.layers.Dense : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
#   - units (int) : Dimensionnalité de l'espace de sortie.
#   - activation (str) : Fonction d'activation à utiliser.

# Compilation du modèle
model.compile(
    # Optimiseur Adam avec learning rate par défaut
    optimizer='adam',
    # Fonction de perte pour classification multiclasse (entiers)
    loss='sparse_categorical_crossentropy',
    # Métrique pour évaluer la performance du modèle
    metrics=['accuracy']
)

# Documentation :
# - Optimizers in TensorFlow : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
#   - optimizer (str or Optimizer) : Algorithme d'optimisation à utiliser.
# - Losses in TensorFlow : https://www.tensorflow.org/api_docs/python/tf/keras/losses
#   - loss (str or Loss) : Fonction de perte à optimiser.
# - metrics (list) : Liste des métriques à évaluer pendant l'entraînement et le test.

# Entraînement du modèle
model.fit(
    # Données d'entraînement normalisées
    X_train_scaled,
    # Étiquettes d'entraînement
    y_train,
    # Nombre d'époques d'entraînement
    epochs=50,
    # Verbosité (niveau de détails lors de l'entraînement)
    verbose=2
)

# Documentation :
# - Model training in TensorFlow : https://www.tensorflow.org/guide/keras/train_and_evaluate
#   - x (array-like) : Données d'entraînement.
#   - y (array-like) : Étiquettes d'entraînement.
#   - epochs (int) : Nombre d'époques d'entraînement.

# Évaluation du modèle
y_pred = model.predict(
    # Données de test normalisées
    X_test_scaled
)
y_pred_classes = tf.argmax(
    # Probabilités prédites
    y_pred,
    # Axe le long duquel trouver les valeurs maximales
    axis=1
)

# Calcul de la précision
accuracy = accuracy_score(
    # Vraies étiquettes de l'ensemble de test
    y_test,
    # Étiquettes prédites
    y_pred_classes.numpy()
)
print(f"Accuracy (TensorFlow Dense Neural Network): {accuracy}")

# Documentation :
# - tf.argmax : https://www.tensorflow.org/api_docs/python/tf/math.argmax
#   - input (array-like) : Un tableau d'entrée.
#   - axis (int) : Dimension le long de laquelle trouver les valeurs maximales.
