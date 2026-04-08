import numpy as np

def mk_dhmm_obs_lik(data, obsmat, obsmat1=None):
    """
    Crée la matrice de vraisemblance des observations pour un HMM discret.

    Args:
        data (list or numpy.ndarray): Séquence d'observations y(t).
        obsmat (numpy.ndarray): Matrice des probabilités conditionnelles P(Y(t)=o | Q(t)=i).
        obsmat1 (numpy.ndarray, optional): Matrice spécifique pour P(Y(1)=o | Q(1)=i).
                                           Si None, utilise `obsmat`.

    Returns:
        numpy.ndarray: Matrice B(i,t) = P(y(t) | Q(t)=i).
    """
    if obsmat1 is None:
        obsmat1 = obsmat

    Q, O = obsmat.shape  # Q : Nombre d'états, O : Nombre d'observations possibles
    T = len(data)  # Nombre de pas de temps

    B = np.zeros((Q, T))

    # Initialisation
    B[:, 0] = obsmat1[:, data[0]]

    # Remplissage de la matrice pour t > 1
    for t in range(1, T):
        B[:, t] = obsmat[:, data[t]]

    return B
