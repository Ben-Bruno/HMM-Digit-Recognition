import numpy as np

def histogrammeProjection(imdata, dim='h'):
    """
    Calcule l'histogramme de projection de l'image selon une dimension spécifiée.

    Paramètres :
    - imdata : ndarray, l'image en niveaux de gris (matrice 2D)
    - dim : str, 'v' pour une projection verticale, 'h' pour une projection horizontale

    Retour :
    - l : ndarray, le vecteur de projection
    """
    if len(imdata.shape) == 3:
        raise ValueError("L'image doit être en niveaux de gris !")
        
    # Histogramme projection     
    if dim.lower() == 'v':  
        # TODO : Implémenter ici l'histogramme verticale
        l = np.sum(imdata, axis=0)
    elif dim.lower() == 'h':  
        # TODO : Implémenter ici l'histogramme horizontales 
        l = np.sum(imdata, axis=1)
    else:
        raise ValueError('Dimension spécifiée incorrecte : utilisez "v" pour vertical ou "h" pour horizontal')
    
    return l
