import numpy as np
from imtools import image2Double

def imageBinarisation(imdata, seuil=0.5):
    """
    Binarise une image en niveaux de gris en utilisant un seuil donné.

    Paramètres :
    - imdata : ndarray, image en niveaux de gris (8 bits, 16 bits, float ou bool)
    - seuil : float, seuil de binarisation (valeur entre 0 et 1)

    Retour :
    - imgBW : ndarray, image binaire avec des valeurs 0.0 ou 1.0
    """

    # Convertir l'image en type double (valeurs entre 0 et 1)
    imdata = image2Double(imdata)

    # TODO : Appliquer le seuillage binaire (0 si < seuil, 1 sinon)
    # Remplacer par la condition de binarisation

    #imgBW = 1 if imdata>seuil else 0 
    imgBW = (imdata > seuil).astype(np.float64)
     
    # TODO : Convertir en type float pour obtenir des valeurs 0.0 et 1.0
    # Remplacer par l'usage de astype(np.float64)
    #imgBW = np.astype(np.float64())

    return imgBW



    
