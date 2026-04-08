import numpy as np

def image2Double(image):
    """
    Convertit une image de différents types (uint8, uint16, int16, float32, bool) 
    en une image de type float64 normalisée entre 0 et 1.

    Paramètre :
    - image : ndarray, image en niveaux de gris avec un type de données supporté

    Retour :
    - image_double : ndarray, image normalisée en type float64
    """

    # Conversion des images en fonction de leur type de données
    if image.dtype == np.uint8:  # Image 8 bits (0 à 255)
        image_double = image.astype(np.float64) / 255.0

    elif image.dtype == np.uint16:  # Image 16 bits non signé (0 à 65535)
        image_double = image.astype(np.float64) / 65535.0

    elif image.dtype == np.int16:  # Image 16 bits signé (-32768 à 32767)
        image_double = (image.astype(np.float64) + 32768) / 65535.0

    elif image.dtype == np.float32:  # Image de type 'single' (float32)
        # Si l'image est déjà normalisée entre 0 et 1, la conversion est directe
        image_double = np.clip(image.astype(np.float64), 0.0, 1.0)

    elif image.dtype == bool:  # Image logique (booléen)
        image_double = image.astype(np.float64)
        
    elif image.dtype == np.float64:  # Image logique (booléen)
        image_double = image

    else:
        raise ValueError("Le format de l'image doit être uint8, uint16, int16, float32 ou bool")

    return image_double
