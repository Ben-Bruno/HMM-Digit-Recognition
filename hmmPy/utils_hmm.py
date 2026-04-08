# utils_hmm.py
import numpy as np

def normalise(M):
    """
    Normalise un tableau pour que sa somme soit égale à 1.
    
    Args:
        M (numpy.ndarray): Tableau à normaliser.

    Returns:
        tuple: (M_normalisé, somme_originale)
    """
    c = np.sum(M)
    d = c if c != 0 else 1  # Éviter la division par zéro
    return M / d, c

def mk_stochastic(T):
    """
    Assure que l'argument est une matrice stochastique, c'est-à-dire que la somme sur la dernière dimension est égale à 1.
    
    Args:
        T (numpy.ndarray): Tableau à rendre stochastique.

    Returns:
        numpy.ndarray: Matrice ou tenseur stochastique.
    """
    T = np.array(T, dtype=float)  # Convertir en tableau numpy pour assurer la compatibilité

    if T.ndim == 1:  # Vecteur
        T, _ = normalise(T)
    elif T.ndim == 2:  # Matrice
        S = np.sum(T, axis=1, keepdims=True)
        S[S == 0] = 1  # Éviter la division par zéro
        T = T / S
    else:  # Tableau multidimensionnel
        ns = T.shape
        T = T.reshape(-1, ns[-1])
        S = np.sum(T, axis=1, keepdims=True)
        S[S == 0] = 1  # Éviter la division par zéro
        T = T / S
        T = T.reshape(ns)

    return T

def em_converged(loglik, previous_loglik, threshold=1e-4):
    """
    Vérifie si l'algorithme EM a convergé.
    
    Args:
        loglik (float): Log-vraisemblance actuelle.
        previous_loglik (float): Log-vraisemblance précédente.
        threshold (float, optional): Seuil de convergence. Par défaut à 1e-4.
    
    Returns:
        tuple: (converged, decrease)
            - converged (bool): True si la convergence est atteinte.
            - decrease (bool): True si la log-vraisemblance a diminué.
    """
    converged = False
    decrease = False

    if loglik - previous_loglik < -1e-3:  # Autorise une légère imprécision
        print(f"****** La log-vraisemblance a diminué de {previous_loglik:.4f} à {loglik:.4f} !")
        decrease = True

    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2

    if (delta_loglik / avg_loglik) < threshold:
        converged = True

    return converged, decrease
