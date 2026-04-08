import numpy as np

from hmmPy.algorithms.mk_dhmm_obs_lik import mk_dhmm_obs_lik
from hmmPy.algorithms.forwards import forwards

def log_lik_dhmm(data, prior, transmat, obsmat):
    """
    Calcule la log-vraisemblance d'un ensemble de données à l'aide d'un HMM discret.

    Args:
        data (numpy.ndarray or list): Matrice des séquences d'observations (ncases, T).
        prior (numpy.ndarray): Distribution initiale P(Q(1) = i).
        transmat (numpy.ndarray): Matrice de transition P(Q(t+1) = j | Q(t) = i).
        obsmat (numpy.ndarray): Matrice d'observation P(Y(t) = o | Q(t) = i).

    Returns:
        float: Log-vraisemblance totale des séquences.
    """
    if isinstance(data, list):
        data = np.array(data)  # Convertir la liste en array numpy si nécessaire

    ncases, T = data.shape  # Nombre de séquences et longueur de chaque séquence
    loglik = 0

    for m in range(ncases):
        obslik = mk_dhmm_obs_lik(data[m], obsmat)
        _, LL = forwards(prior, transmat, obslik)
        loglik += LL

    return loglik
