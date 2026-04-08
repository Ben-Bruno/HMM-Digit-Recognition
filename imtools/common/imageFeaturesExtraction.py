import numpy as np
from imtools import histogrammeProjection

def imageFeaturesExtraction(imBW, choix="hist"):
    """
    Extrait les caractéristiques d'une image binaire en utilisant les histogrammes de projection.

    Paramètres :
    - imBW : ndarray, image binaire (0 pour le fond, 1 pour les pixels du chiffre)
    - choix : str, type de caractéristique à extraire (par défaut "hist" pour les histogrammes de projection)

    Retour :
    - observation : ndarray, vecteur de caractéristiques combinant les projections verticales et horizontales
    """

    if choix.lower() == "hist":
        # TODO : Calcul des histogrammes de projections verticales et horizontales
        Hist_V = histogrammeProjection(imBW, dim='v')  
        # Compléter avec l'appel à l'histogrammes de projection verticale
        Hist_H = histogrammeProjection(imBW, dim='h')  

        # Compléter avec l'appel à l'histogrammes de projection horizontale

         # TODO : Combinaison des caractéristiques verticales et horizontales
        observation = np.concatenate([Hist_V, Hist_H])  
        # Compléter avec np.concatenate([Hist_V, Hist_H])
    else:
        raise ValueError("Choix non valide : utilisez 'hist' pour les histogrammes de projection")

    return observation
