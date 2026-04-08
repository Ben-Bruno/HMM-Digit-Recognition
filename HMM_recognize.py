import numpy as np
import matplotlib.pyplot as plt

from imtools import imageLocalizationExtraction, imagePreprocessing, imageFeaturesExtraction, imageRecognition

mrows, ncols = 16, 16
# Définition du chiffre à reconnaître
chiffre = 0

# Extraction de l'image du chiffre
filename = 'datasets/test.tif'
Images = imageLocalizationExtraction(filename)

# Récupération du chiffre à reconnaître
imdata = Images[f"{chiffre}_{0}"]

# Affichage de l'image (commenté comme en MATLAB)
plt.imshow(imdata, cmap='gray')
plt.axis('off')
plt.show()

# Prétraitement de l'image
imBW, imdata = imagePreprocessing(imdata, mrows, ncols) 
 # TODO : à compléter par l'appel de la fonction

# Extraction des caractéristiques
features = imageFeaturesExtraction(imBW, "hist")
# TODO : à compléter par l'appel de la fonction


# Chargement du modèle HMM entraîné
hmmModel_data = np.load('NPZFile/hmmModel.npz', allow_pickle=True)
hmmModel = hmmModel_data['hmmModel'].item()  # Récupération du dictionnaire

# Calcul des probabilités de reconnaissance
probabilities = imageRecognition(features, hmmModel)
# TODO : à compléter par l'appel de la fonction

# Détection du chiffre reconnu
ind_maxi = np.argmax(probabilities)# TODO : Compléter avec np.argmax() pour obtenir le chiffre prédit

# Affichage du résultat
print(f"Le chiffre recherché est : {ind_maxi}")

# Affichage de l'image et des probabilités
plt.figure(figsize=(8, 6))

# Affichage de l'image du chiffre reconnu
plt.subplot(2, 1, 1)
plt.imshow(imdata, cmap='gray')
plt.axis('off')
plt.title("Chiffre extrait")

# Affichage des probabilités de reconnaissance
plt.subplot(2, 1, 2)
numbers = np.arange(10)
plt.stem(numbers, probabilities / np.sum(probabilities), linefmt='b', markerfmt='bo', basefmt="r")
plt.xlabel("Chiffres")
plt.ylabel("Probabilités normalisées")
plt.title("Probabilité de reconnaissance")

plt.tight_layout()
plt.show()
