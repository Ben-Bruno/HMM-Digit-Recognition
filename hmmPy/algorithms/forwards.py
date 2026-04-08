
import numpy as np
from hmmPy.utils_hmm import normalise

def forwards(prior, transmat, obslik):
    """
    Calcule les probabilités filtrées dans un HMM en utilisant l'algorithme Forward.

    Args:
        prior (numpy.ndarray): Vecteur de probabilités initiales P(Q(1)=i).
        transmat (numpy.ndarray): Matrice de transition P(Q(t+1)=j | Q(t)=i).
        obslik (numpy.ndarray): Matrice des vraisemblances P(y(t) | Q(t)=i).

    Returns:
        tuple: (alpha, xi, loglik) où :
            - alpha (numpy.ndarray): Matrice des probabilités filtrées.
            - xi (numpy.ndarray): Matrice des probabilités conjointes d'état.
            - loglik (float): Log-vraisemblance de la séquence d'observations.
    """
    T = obslik.shape[1]  # Nombre de temps
    Q = len(prior)  # Nombre d'états cachés

    scaled = True
    scale = np.ones(T)
    loglik = 0

    prior = prior.reshape(-1, 1)  # Assurez-vous que prior est un vecteur colonne
    alpha = np.zeros((Q, T))
    xi = np.zeros((Q, Q, T-1))

    # Initialisation
    alpha[:, 0] = (prior.flatten() * obslik[:, 0])
    if scaled:
        alpha[:, 0], n = normalise(alpha[:, 0])
        scale[0] = 1 / n  # Équivalent à 1 / (somme des alpha[:,0])

    # Boucle Forward
    transmat_T = transmat.T  # Transposé pour simplifier les calculs
    for t in range(1, T):
        alpha[:, t] = (transmat_T @ alpha[:, t-1]) * obslik[:, t]
        if scaled:
            alpha[:, t], n = normalise(alpha[:, t])
            scale[t] = 1 / n
        xi[:, :, t-1] = normalise((alpha[:, t-1][:, None] * obslik[:, t]) * transmat)[0]

    # Calcul de la log-vraisemblance
    if scaled:
        loglik = -np.sum(np.log(scale))
    else:
        _, lik = normalise(alpha[:, -1])
        loglik = np.log(lik)

    return alpha, xi, loglik
