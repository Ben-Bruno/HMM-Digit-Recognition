import numpy as np
from hmmPy.utils_hmm import normalise

def viterbi_path(prior, transmat, obslik):
    """
    Trouve le chemin le plus probable (Viterbi) à travers le treillis d'états du HMM.

    Args:
        prior (numpy.ndarray): Distribution initiale P(Q(1) = i).
        transmat (numpy.ndarray): Matrice de transition P(Q(t+1) = j | Q(t) = i).
        obslik (numpy.ndarray): Matrice des vraisemblances P(y(t) | Q(t) = i).

    Returns:
        numpy.ndarray: Chemin le plus probable des états cachés.
    """
    T = obslik.shape[1]  # Nombre de pas de temps
    Q = len(prior)  # Nombre d'états cachés

    delta = np.zeros((Q, T))
    psi = np.zeros((Q, T), dtype=int)
    path = np.zeros(T, dtype=int)
    scale = np.ones(T)

    # Initialisation
    delta[:, 0] = prior * obslik[:, 0]
    delta[:, 0], scale[0] = normalise(delta[:, 0])
    psi[:, 0] = 0  # Valeur arbitraire pour t=1 car pas de prédécesseur

    # Boucle de récursion de Viterbi
    for t in range(1, T):
        for j in range(Q):
            max_val = delta[:, t - 1] * transmat[:, j]
            delta[j, t] = np.max(max_val)
            psi[j, t] = np.argmax(max_val)
            delta[j, t] *= obslik[j, t]
        delta[:, t], scale[t] = normalise(delta[:, t])

    # Terminaison
    path[T - 1] = np.argmax(delta[:, T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]

    return path
