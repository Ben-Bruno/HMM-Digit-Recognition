import numpy as np
from imtools import histogrammeProjection

def imageCropping(bw):
    """
    Recadre une image binaire en utilisant les histogrammes de projection horizontale et verticale.

    Paramètres :
    - bw : ndarray, image binaire avec 0 pour le fond et 1 pour les pixels du chiffre

    Retour :
    - bw2 : ndarray, image recadrée avec 0 pour le fond et 1 pour les pixels du chiffre
    """
    if len(bw.shape)==3: 
        raise ValueError ('Image non binaire non acceptée!')
    # TODO : Calcul de l'histogramme de projection horizontal
    hist_h = histogrammeProjection(bw, dim='h') 
    
    # Déterminer les lignes contenant des pixels du chiffre
    rows = np.where(hist_h > 0)[0]
    
    # TODO : Calcul de l'histogramme de projection vertical
    hist_v = histogrammeProjection(bw, dim='v')  # Remplacer par le calcul de projection verticale
    
    # Déterminer les colonnes contenant des pixels du chiffre
    cols = np.where(hist_v > 0)[0]
    
    # Si aucun pixel n'est détecté, retourner une image vide
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros_like(bw, dtype=np.float64)
    
    # Recadrer l'image aux limites détectées et convertir en float64
    bw2 = bw[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1].astype(np.float64)
    
    return bw2
