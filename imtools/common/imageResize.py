import numpy as np
from scipy.interpolate import RectBivariateSpline

def imageResize(imBW_crop, mrows, ncols):
    """
    Redimensionne une image binaire en utilisant l'interpolation bicubique (spline).
    """

    # 1. Récupérer les dimensions (yy: lignes, xx: colonnes)
    yy, xx = imBW_crop.shape
    print(f'Dimension avant redimensionnement : ({yy}, {xx})')

    # 2. Créer les vecteurs de coordonnées originales
    # Attention à l'ordre : (lignes, colonnes)
    lignes_orig = np.arange(yy)
    cols_orig = np.arange(xx)

    # 3. Créer la fonction d'interpolation
    # RectBivariateSpline(y, x, z) où z.shape est (len(y), len(x))
    interp_func = RectBivariateSpline(lignes_orig, cols_orig, imBW_crop)

    # 4. Générer les nouvelles coordonnées cibles
    # On veut mrows lignes et ncols colonnes
    nouvelles_lignes = np.linspace(0, yy - 1, mrows)
    nouvelles_cols = np.linspace(0, xx - 1, ncols)

    # 5. Évaluer la spline sur la nouvelle grille
    # Le résultat aura la taille (len(nouvelles_lignes), len(nouvelles_cols))
    imBW_resize = interp_func(nouvelles_lignes, nouvelles_cols)

    # 6. Conversion en binaire (seuillage à 0.5 est plus standard pour du bicubique)
    imBW_resize = (imBW_resize > 0.5).astype(np.float64)
    
    print(f'Dimension après redimensionnement : {imBW_resize.shape}')

    return imBW_resize











"""import numpy as np
from scipy.interpolate import RectBivariateSpline

def imageResize(imBW_crop, mrows, ncols):"""
"""
    Redimensionne une image binaire en utilisant l'interpolation bicubique (spline).

    Paramètres :
    - imBW_crop : ndarray, image binaire recadrée à redimensionner
    - mrows : int, nombre de lignes souhaité pour l'image redimensionnée
    - ncols : int, nombre de colonnes souhaité pour l'image redimensionnée

    Retour :
    - imBW_resize : ndarray, image binaire redimensionnée avec 0 pour le fond et 1 pour les pixels du chiffre
    """
"""
    # Dimensions de l'image originale
    yy, xx = imBW_crop.shape
    print('Dimansion avant le recadrage: ', imBW_crop.shape)

    # Générer les nouvelles coordonnées cibles
    x = np.linspace(0, xx - 1, ncols)
    y = np.linspace(0, yy - 1, mrows)
    xi, yi = np.meshgrid(x, y)

    # Interpolation bicubique (spline)
    interp_func = RectBivariateSpline( np.arange(yy), np.arange(xx), imBW_crop)
    imBW_resize = interp_func(x, y)

    # Conversion en image binaire avec seuillage
    imBW_resize = (imBW_resize > 0.05).astype(np.float64)
    print('Dimansion après le recadrage: ', imBW_resize)


    return imBW_resize
"""