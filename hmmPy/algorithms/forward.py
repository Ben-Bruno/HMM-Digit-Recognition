import numpy as np

def forward(obs, A, B, Pi0):
    """
    Algorithme de propagation vers l'avant (Forward) pour un modèle de Markov caché.

    Args:
        obs (list or array): Séquence d'observations (indices des observations).
        A (numpy.ndarray): Matrice de transition d'état (NxN).
        B (numpy.ndarray): Matrice d'émission (NxM).
        Pi0 (numpy.ndarray): Probabilités initiales des états (N,).

    Returns:
        tuple: (P, alfa) où :
            - P (float): Probabilité de la séquence d'observation.
            - alfa (numpy.ndarray): Matrice des probabilités vers l'avant.
    """
    T = len(obs)  # Nombre d'observations
    N = A.shape[0]  # Nombre d'états cachés

    # Assurez-vous que Pi0 est un vecteur colonne
    Pi0 = np.array(Pi0).reshape(-1, 1)

    # Initialisation
    alfa = np.zeros((T, N))
    alfa[0, :] = (Pi0.flatten() * B[:, obs[0]])

    # Boucle de calcul
    for t in range(1, T):
        somme = np.dot(alfa[t - 1, :], A)
        alfa[t, :] = B[:, obs[t]] * somme

    # Terminaison
    P = np.sum(alfa[T - 1, :])

    return P, alfa
