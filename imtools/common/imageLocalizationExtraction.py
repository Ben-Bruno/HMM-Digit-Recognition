import cv2

from imtools import image2Double, imageLocalization

def imageLocalizationExtraction(filename):
    """
    Localisation et extraction des chiffres de l'image spécifiée.

    Paramètre :
    - filename : str, chemin vers le fichier image.

    Retour :
    - Images : dictionnaire contenant les images des chiffres extraits.
    """

    # Lecture du fichier image
    A = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Convertir une image (8bits ou 16bits) en double
    AA = image2Double(A)
    AA_bw = 1.0 - (AA >= 0.005).astype(float)

    # Initialiser un dictionnaire pour stocker les images extraites
    Images = {}

    # Segmentation et extraction des chiffres
    for n in range(10):
        # Histogramme des projections horizontales --> extraction des Lignes
        ind1, ind2, _ = imageLocalization(AA_bw, 'h')
        tmp = AA_bw[ind1[n]:ind2[n], :]

        # Histogramme des projections verticales --> extraction des Colonnes
        ind11, ind22, _ = imageLocalization(tmp, 'v')

        # Extraction de chaque chiffre
        for m in range(len(ind11)):
            Images[f"{n}_{m}"] = tmp[:, ind11[m]:ind22[m]] # Stockage de chaque chiffre extrait dans le dictionnaire avec une clé formatée
    
    return Images
