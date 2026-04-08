import numpy as np

from imtools import histogrammeProjection

def imageLocalization(imdata, dim='h'):
    """
    Localise les indices de début et de fin des régions non nulles dans l'image
    en utilisant un histogramme de projection.

    Paramètres :
    - imdata : ndarray, image en niveaux de gris (matrice 2D)
    - dim : str, 'v' pour une projection verticale, 'h' pour une projection horizontale

    Retour :
    - ind1 : ndarray, indices de début des régions non nulles
    - ind2 : ndarray, indices de fin des régions non nulles
    - l : ndarray, histogramme de projection
    """

    # Calcul de l'histogramme de projection de l'image selon la dimension spécifiée
    l = histogrammeProjection(imdata, dim)

    # Détection des indices de début et de fin des régions non nulles
    ind1 = np.flatnonzero((l[:-1] == 0) & (l[1:] != 0)) + 1
    ind2 = np.flatnonzero((l[:-1] != 0) & (l[1:] == 0))
    
    return ind1, ind2, l
