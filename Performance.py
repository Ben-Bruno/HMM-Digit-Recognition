import numpy as np
from imtools import imageLocalizationExtraction, imagePreprocessing, imageFeaturesExtraction, imageRecognition

mrow, ncol=16, 16
# Extraction des chiffres à partir de l'image
filename = 'datasets/test.tif'
Images = imageLocalizationExtraction(filename) # Retourne un dictionnaire avec clés "chiffre_index"

# Chargement du modèle entraîné
hmmModel_data = np.load('NPZFile/hmmModel.npz', allow_pickle=True)
hmmModel = hmmModel_data['hmmModel'].item()  # Récupération du dictionnaire

# Initialisation des matrices de comptage
Nbre = np.zeros(10)  # Nombre d'images par chiffre
A = np.zeros((10, 10))  # Matrice de confusion

# Boucle sur chaque chiffre (de 0 à 9)
for chiffre in range(10):
    indices = [n for n in range(len(Images)) if f"{chiffre}_{n}" in Images]  # Vérifier les clés existantes
    ind = len(indices)  # Nombre d'images pour ce chiffre
    
    
    if ind == 0:
        continue # Passer si aucune image trouvée
    
    temp = np.zeros(ind)  # Temporaire pour stocker les prédictions

    for n in range(ind):
        # Lecture de chaque chiffre à reconnaître
        imdata = Images[f"{chiffre}_{n}"]

        # Image Preprocessing
        imBW, imdata = imagePreprocessing(imdata, mrow, ncol)
        # TODO : à compléter par l'appel de la fonction

        # Features Extraction
        features = imageFeaturesExtraction(imBW, "hist")  
        # TODO : à compléter par l'appel de la fonction

        # Reconnaissance de l'image avec le HMM
        probabilities = imageRecognition(features, hmmModel)
        # TODO : à compléter par l'appel de la fonction
        ind_maxi = np.argmax(probabilities)
        # TODO : Compléter avec np.argmax()

        temp[n] = ind_maxi  # Stocker la prédiction

    # Mise à jour de la matrice de confusion
    for j in range(10):
        A[chiffre, j] = np.sum(temp == j)

    Nbre[chiffre] = ind  # Stocker le nombre d'images pour ce chiffre

# Calcul de la matrice de confusion normalisée
AA = A / (Nbre[:, np.newaxis] + 1e-10)  # Ajout d'un epsilon pour éviter division par zéro

# 1. Nombre d'images bien classées (la diagonale de la matrice A)
nbre_reconnues = np.sum(np.diag(A))

# 2. Nombre total d'images testées
total_images = np.sum(A)

# 3. Nombre d'images mal classées
nbre_non_recon = total_images - nbre_reconnues

# 4. Taux de reconnaissance (Accuracy)
# On multiplie par 100 pour l'avoir en pourcentage
taux_reconnaissance = (nbre_reconnues / total_images) * 100

# 5. Taux d'erreur
taux_erreur = 100 - taux_reconnaissance

# Affichage des résultats
print("\n" + "."*30)
print("RÉSULTATS DE LA CLASSIFICATION")
print("."*10)
print("Matrice de confusion: ")
print(AA)
print(f"Images bien classées : {int(nbre_reconnues)}")
print(f"Images mal classées  : {int(nbre_non_recon)}")
print(f"Taux de reconnaissance : {taux_reconnaissance:.2f}%")
print(f"Taux d'erreur         : {taux_erreur:.2f}%")
print("."*10)

# Optionnel : Affichage de la matrice de confusion normalisée
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(AA, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Chiffre Prédit')
plt.ylabel('Chiffre Réel (Vrai)')
plt.title('Matrice de Confusion Normalisée')
plt.show()