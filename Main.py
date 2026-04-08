# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:30:57 2025

@author: mouhamadou
"""

import cv2 #Importation de la librairie OpenCv 
import numpy as np
import matplotlib.pyplot as plt
from imtools import imageLocalizationExtraction, histogrammeProjection, imageLocalization

from  imtools import image2Double, imageBinarisation, imageCropping, imageResize, imageFeaturesExtraction


# Définir les chemins des fichiers d'entraînement et de test
filename_train = 'datasets/train.tif'
filename_test = 'datasets/test.tif'

# Charger les images en niveaux de gris avec OpenCV
train = cv2.imread(filename_train, cv2.IMREAD_GRAYSCALE)
test = cv2.imread(filename_test, cv2.IMREAD_GRAYSCALE)

# Afficher l'image d'entraînement
plt.figure(1)
plt.imshow(train, cmap='gray')
plt.title('Image d\'entraînement')
plt.axis('off')
plt.show()

# Afficher l'image de test
plt.figure(2)
plt.imshow(test, cmap='gray')
plt.title('Image de test')
plt.axis('off')
plt.show()

#imgG= imageLocalizationExtraction(filename_train)
train_img= 1.0-image2Double(train) # 1.0- ... Car dans mon afichage j'avait des histogrammes inversés
histV=histogrammeProjection(train_img, dim='v')
histH=histogrammeProjection(train_img, dim='h')

plt.figure()
plt.plot(histH)
plt.title("Histogramme de projection horizontale (Détection des lignes)")
plt.show()

[ind1, ind2, l_horiz] = imageLocalization(train_img, dim='h')

premiere_ligne = train_img[ind1[0]:ind2[0], :]

#Afficher l'histogramme vertical de cette ligne
v_hist_ligne = histogrammeProjection(premiere_ligne, dim='v')
plt.figure()
plt.plot(v_hist_ligne)
plt.title("Histogramme vertical de la 1ère ligne (Détection des chiffres)")
plt.show()

# Extraire le premier chiffre de cette ligne
[c1, c2, l_vert] = imageLocalization(premiere_ligne, dim='v')
premier_chiffre = premiere_ligne[:, c1[0]:c2[0]]

# Affichage du résultat
plt.figure()
plt.imshow(premier_chiffre, cmap='gray')
plt.title("Premier chiffre extrait")
plt.show()

#Étape 3 : Sauvegarde des chiffres extraits de la base de données

img_bin_train= imageLocalizationExtraction(filename_train)

img_bin_test= imageLocalizationExtraction(filename_test)
np.savez('NPZFile/train.npz', **img_bin_train)
np.savez('NPZFile/test.npz', **img_bin_test)

imgtr= img_bin_train[f"{7}_{4}"]
plt.figure()
plt.imshow(imgtr, cmap='gray')
plt.title("Chiffre de la ligne 7 colonne 4")
plt.show()

imgRiceNormale= cv2.imread('rice.png', cv2.IMREAD_GRAYSCALE)
imgPrinted= cv2.imread('printedtext.png', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(imgRiceNormale, cmap='gray')
plt.title("Image du riz avant binarisation")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(imgPrinted, cmap='gray')
plt.title("Image de printedText avant binarisation")
plt.axis('off')
plt.show()

rice_double=image2Double(imgRiceNormale)
rice_binaire=imageBinarisation(rice_double, 0.5)

printed_double=image2Double(imgPrinted)
printed_binaire=imageBinarisation(printed_double, 0.5)

plt.figure()
plt.imshow(rice_binaire, cmap='gray')
plt.title("Image du riz après binarisation")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(printed_binaire, cmap='gray')
plt.title("Image de printedText après binarisation")
plt.axis('off')
plt.show()

#Recadrage
imgtr2= img_bin_train[f"{8}_{4}"]
plt.figure()
plt.imshow(imgtr2, cmap='gray')
plt.title("image de 1 avant recadrage")
plt.axis('off')
plt.show()


bw_crop=imageCropping(imgtr2)
plt.figure()
plt.imshow(bw_crop, cmap='gray')
plt.title('Image de 1 après recadrage')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imgtr2, cmap='gray')
plt.title("image de 1 avant recadrage")
plt.subplot(1, 2, 2)
plt.imshow(bw_crop, cmap='gray')
plt.title('Image de 1 après recadrage')
plt.show()


imBW_Resize= imageResize(bw_crop, 16, 16)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(bw_crop, cmap='gray')
plt.title('Image recadrée avant dimensionnement')

plt.subplot(1, 2, 2)
plt.imshow(imBW_Resize, cmap='gray')
plt.title('Image recadrée après dimensionnement')
plt.show

# page 9

observation=imageFeaturesExtraction(imBW_Resize)
import subprocess
import sys
import os

# 1. S'assurer que le dossier de destination existe
os.makedirs('NPZFile', exist_ok=True)

print("--- ÉTAPE 1 : Extraction des caractéristiques ---")
try:
    # On appelle d'abord le script qui génère features.npz
    # Remplace 'featureExtraction.py' par le nom exact de TON script d'extraction
    subprocess.run([sys.executable, "featureExtraction.py"], check=True)
    print("Fichier features.npz généré avec succès.")
except Exception as e:
    print(f"Erreur lors de l'extraction : {e}")
    sys.exit(1) # On arrête tout si l'extraction rate

print("\n--- ÉTAPE 2 : Lancement de l'apprentissage ---")
try:
    # Maintenant que features.npz existe, on peut lancer l'entraînement
    subprocess.run([sys.executable, "trainHMM.py"], check=True)
    print("Modèles HMM entraînés et sauvegardés.")
except Exception as e:
    print(f"Erreur lors de l'apprentissage : {e}")
# TODO : Partie de script à compléter ...
