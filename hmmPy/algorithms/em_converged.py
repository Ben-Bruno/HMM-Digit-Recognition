
import numpy as np

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