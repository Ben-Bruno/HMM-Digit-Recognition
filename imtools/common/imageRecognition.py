import numpy as np
import hmmPy as hmm

def imageRecognition(obs, hmmModel):
    """
    Calcule les probabilités de reconnaissance en utilisant un HMM.
    
    Paramètres :
    - obs : séquence d'observations
    - hmmModel : dictionnaire contenant les modèles HMM pour chaque chiffre
    
    Retour :
    - probabilities : vecteur des probabilités normalisées pour chaque chiffre (0-9)
    """
    
    probabilities = np.zeros(10)  # Initialisation du vecteur de probabilité

    # Parcourir tous les modèles HMM (chiffres de 0 à 9)
    for nn in range(10):
        model = hmmModel[str(nn)]  # Récupérer le modèle HMM du chiffre nn
        prior = model['prior']      # Vecteur de probabilité initiale
        transmat = model['transmat']  # Matrice de transition
        obsmat = model['obsmat']    # Matrice d’observation
        
        obs = obs.astype(int)
        
        # TODO : Calcul de la probabilité d'observation avec Forward
        P, _ = hmm.forward(obs, transmat, obsmat, prior)
        probabilities[nn] = np.exp(P) if P < 0 else P

    # Normalisation des probabilités
    total_prob = np.sum(probabilities)
    if total_prob > 0:
        probabilities /= np.sum(probabilities)


    return probabilities
