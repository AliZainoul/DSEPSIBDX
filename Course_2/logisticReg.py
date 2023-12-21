# Ce script Python illustre l'application de la régression logistique à un jeu de données en utilisant scikit-learn.
# Le jeu de données utilisé est le jeu de données Iris, un ensemble classique en apprentissage automatique.
# Les principales étapes du script sont les suivantes :
# 1. Chargement du jeu de données Iris à l'aide de la fonction load_iris() de scikit-learn.
# 2. Séparation du jeu de données en features (X) et variable cible (y).
# 3. Division du jeu de données en ensembles d'entraînement et de test à l'aide de train_test_split.
# 4. Initialisation d'un modèle de régression logistique à l'aide de LogisticRegression.
# 5. Normalisation des données à l'aide de StandardScaler.
# 6. Entraînement du modèle sur les données d'entraînement à l'aide de la méthode fit.
# 7. Prédiction des classes pour les données de test à l'aide de la méthode predict.
# 8. Évaluation de la précision du modèle en utilisant la fonction accuracy_score.
# 9. Affichage de la précision du modèle.

# Importation des bibliothèques nécessaires
from sklearn.datasets import load_iris                
# Importation de la fonction load_iris pour charger le jeu de données Iris
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

from sklearn.model_selection import train_test_split  
# Importation de la fonction train_test_split pour diviser le jeu de données
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.linear_model import LogisticRegression   
# Importation de la classe LogisticRegression pour créer un modèle de régression logistique
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.metrics import accuracy_score            
# Importation de la fonction accuracy_score pour évaluer la précision du modèle
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

from sklearn.preprocessing import StandardScaler
# Importation de la classe StandardScaler pour normaliser les données
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

# Chargement du jeu de données Iris
iris = load_iris()  
# Chargement des données du jeu de données Iris
# Membre: iris.data, iris.target
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

X, y = iris.data, iris.target  
# Séparation des features et de la variable cible

# Division du jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
# Création d'ensembles d'entraînement et de test
# Méthode: train_test_split
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# Application d'une régression logistique
logistic_model = LogisticRegression()  
# Initialisation du modèle de régression logistique
# Constructeur: LogisticRegression
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model.fit(X_train_scaled, y_train)  
# Entraînement du modèle sur les données d'entraînement
# Méthode: fit
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit

logistic_predictions = logistic_model.predict(X_test_scaled)  
# Prédiction sur les données de test
# Méthode: predict
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict

# Évaluation de la régression logistique
accuracy_logistic = accuracy_score(y_test, logistic_predictions)  
# Calcul de la précision du modèle
# Méthode: accuracy_score
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

# Affichage de la précision du modèle
print(f"Accuracy (Logistic Regression): {accuracy_logistic}")  

# Matrice de confusion
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, logistic_predictions)
print("Matrice de confusion :\n", confusion_mat)
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

# Rapport de classification
from sklearn.metrics import classification_report
classification_rep = classification_report(y_test, logistic_predictions)
print("Rapport de classification :\n", classification_rep)
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html


# LOGIQUE COURBE ROC
# Courbe ROC pour la classification multiclasse (modification ajoutée)
from sklearn.metrics import roc_curve, auc
# Importation des fonctions roc_curve et auc pour générer la courbe ROC et calculer l'aire sous la courbe (AUC)
# Documentation roc_curve: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# Documentation auc: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

import matplotlib.pyplot as plt
# Importation de la bibliothèque matplotlib.pyplot pour la visualisation graphique
# Documentation: https://matplotlib.org/stable/api/pyplot_summary.html

from sklearn.preprocessing import label_binarize
# Importation de la fonction label_binarize pour binariser les étiquettes multiclasse
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html

from itertools import cycle
# Importation de la fonction cycle pour itérer en boucle sur les couleurs lors de l'affichage des courbes ROC
# Documentation: https://docs.python.org/3/library/itertools.html#itertools.cycle

# Binariser les étiquettes
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
# Utilisation de label_binarize pour transformer les étiquettes multiclasse en un format binaire
# Méthode: label_binarize
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html

# Prédictions pour chaque classe
logistic_predictions_proba = logistic_model.predict_proba(X_test_scaled)
# Utilisation de la méthode predict_proba pour obtenir les probabilités de prédiction pour chaque classe
# Méthode: predict_proba
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba

# Calcul des courbes ROC et de l'aire sous la courbe (AUC) pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(iris.target_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], logistic_predictions_proba[:, i])
    # Calcul des taux de faux positifs (fpr) et des taux de vrais positifs (tpr) pour chaque classe
    # Méthode: roc_curve
    # Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    roc_auc[i] = auc(fpr[i], tpr[i])
    # Calcul de l'aire sous la courbe (AUC) pour chaque classe
    # Méthode: auc
    # Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

# Affichage des courbes ROC pour chaque classe
plt.figure(figsize=(8, 6))
colors = cycle(['orange', 'red', 'aqua'])
lw_multiplier = 1.3
alpha_multiplier = 0.8  # Ajustez la transparence selon vos préférences

for i, color in zip(range(len(iris.target_names)), colors):
    lw = 1.3 * (i + 1)  # Chaque courbe a une épaisseur différente
    alpha = alpha_multiplier / (i + 1)  # Ajustez la transparence pour chaque courbe
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, alpha=alpha, label='Courbe ROC (AUC = {:.2f})'.format(roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
# Ligne en pointillés représentant la ligne de référence aléatoire

plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC pour chaque classe')
plt.legend(loc="lower right")
# Ajout de légendes pour chaque courbe ROC
# Test pour vérifier les données des courbes ROC pour chaque classe

plt.show()
# Affichage graphique de toutes les courbes ROC


# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
