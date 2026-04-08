
import imtools

def imagePreprocessing(imdata, mrows, ncols):
    """
    Prétraite une image en niveaux de gris :
    - Binarisation
    - Recadrage de l'image
    - Redimensionnement

    Paramètres :
    - imdata : ndarray, image en niveaux de gris
    - mrows : int, nombre de lignes souhaité pour l'image redimensionnée
    - ncols : int, nombre de colonnes souhaité pour l'image redimensionnée
    
    Retour :
    - imBW : ndarray, image binaire prétraitée
    - imData : ndarray, image en niveaux de gris prétraitée
    """

    # Binarisation de l'image
    imBW = imtools.imageBinarisation(imdata, seuil=0.005)

    # Recadrage de l'image binaire
    imBW_crop = imtools.imageCropping(imBW)
    imData = imtools.imageResize(imBW_crop, mrows, ncols)
    # Redimensionnement direct avec binarisation finale intégrée
    imBW = imtools.imageBinarisation(imtools.imageResize(imBW_crop, mrows, ncols), seuil=0.001)

    return imBW, imData
