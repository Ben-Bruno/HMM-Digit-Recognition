import numpy as np
from hmmPy.utils_hmm import normalise

def forwards_backwards(prior, transmat, obslik, filter_only=False):
    """
    Calcule les probabilités a posteriori dans un HMM en utilisant l'algorithme Forward-Backward.

    Args:
        prior (numpy.ndarray): Vecteur des probabilités initiales P(Q(1) = i).
        transmat (numpy.ndarray): Matrice de transition P(Q(t+1) = j | Q(t) = i).
        obslik (numpy.ndarray): Matrice des vraisemblances P(Y(t) | Q(t) = i).
        filter_only (bool, optional): Si True, effectue uniquement le filtrage. Sinon, effectue aussi le lissage.

    Returns:
        tuple: (gamma, xi, loglik) où :
            - gamma (numpy.ndarray): Probabilité a posteriori P(X(t) = i | O(1:T)).
            - xi (numpy.ndarray): Probabilité conjointe P(X(t) = i, X(t+1) = j | O(1:T)) pour t ≤ T-1.
            - loglik (float): Log-vraisemblance des observations.
    """
    T = obslik.shape[1]  # Nombre d'observations
    Q = len(prior)  # Nombre d'états cachés

    scale = np.ones(T)
    loglik = 0
    alpha = np.zeros((Q, T))
    gamma = np.zeros((Q, T))
    xi = np.zeros((Q, Q, T-1))

    # Forward pass (Filtrage)
    alpha[:, 0] = prior.flatten() * obslik[:, 0]
    alpha[:, 0], scale[0] = normalise(alpha[:, 0])
    transmat_T = transmat.T

    for t in range(1, T):
        alpha[:, t], scale[t] = normalise((transmat_T @ alpha[:, t-1]) * obslik[:, t])
        if filter_only:
            xi[:, :, t-1] = normalise((np.outer(alpha[:, t-1], obslik[:, t])) * transmat)[0]

    loglik = np.sum(np.log(scale))

    if filter_only:
        return alpha, xi, loglik

    # Backward pass (Lissage)
    beta = np.ones((Q, T))
    gamma[:, T-1] = normalise(alpha[:, T-1] * beta[:, T-1])[0]

    for t in range(T-2, -1, -1):
        b = beta[:, t+1] * obslik[:, t+1]
        beta[:, t] = normalise(transmat @ b)[0]
        gamma[:, t] = normalise(alpha[:, t] * beta[:, t])[0]
        xi[:, :, t] = normalise(transmat * np.outer(alpha[:, t], b))[0]

    return gamma, xi, loglik
