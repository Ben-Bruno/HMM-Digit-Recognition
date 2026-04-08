import numpy as np

from imtools import imageLocalizationExtraction, imagePreprocessing, imageFeaturesExtraction
import os

nrow, ncol=16, 16
# Définir les chemins de fichiers et dossiers
filename = 'datasets/train.tif'

# Lecture des chiffres de la base de données
Images = imageLocalizationExtraction(filename)

# Initialisation d'un dictionnaire pour stocker les caractéristiques
features = {f"{n}": [] for n in range(10)}

 # TODO : Extraction des caractéristiques pour chaque chiffre de chaque classe (0 à 9)
# à compléter ...
#BOUCLE D'EXTRACTION
for key, im_raw in Images.items():
    # 1. Identifier la classe (le premier caractère de la clé "0_5" -> "0")
    label = key.split('_')[0]
    
    # 2. Prétraitement complet (Binarisation, Cropping, Resize)
    # On récupère l'image binaire imBW (16x16)
    imBW, _ = imagePreprocessing(im_raw, nrow, ncol)
    
    # 3. Extraction des caractéristiques (Vecteur de taille 32)
    obs = imageFeaturesExtraction(imBW, choix="hist")
    
    # 4. Stockage dans la classe correspondante
    features[label].append(obs)

# Conversion des listes en tableaux numpy pour la sauvegarde .npz
for n in range(10):
    features[f"{n}"] = np.array(features[f"{n}"])

# Sauvegarde des résultats
output_dir = 'NPZFile'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'features.npz')
np.savez(output_path, **features)

print(f"Extraction terminée. Caractéristiques sauvegardées dans {output_path}")