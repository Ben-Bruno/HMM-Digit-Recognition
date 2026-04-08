import numpy as np
from hmmPy.utils_hmm import mk_stochastic
from hmmPy.algorithms.learn_dhmm import learn_dhmm

def train_HMM(data, states, count, it_sp, it_bw, model_type=1):
    """
    Entraîne un HMM avec l'algorithme Baum-Welch et sélectionne le modèle avec la meilleure log-vraisemblance.

    Args:
        data (list of lists or numpy.ndarray): Séquences d'observations.
        states (int): Nombre d'états cachés.
        count (int): Nombre d'observations possibles.
        it_sp (int): Nombre d'itérations pour essayer différentes initialisations.
        it_bw (int): Nombre d'itérations de Baum-Welch (EM).
        model_type (int): 1 pour un modèle "left-right", sinon modèle standard.

    Returns:
        tuple: (ll, prior_matrix, transition_matrix, observation_matrix)
            - ll (float): Meilleure log-vraisemblance obtenue.
            - prior_matrix (numpy.ndarray): Meilleure distribution initiale des états.
            - transition_matrix (numpy.ndarray): Meilleure matrice de transition.
            - observation_matrix (numpy.ndarray): Meilleure matrice d'observation.
    """
    best_ll = -np.inf  # Log-vraisemblance initiale très faible

    for k in range(it_sp):
        # Initialisation aléatoire des matrices
        prior_matrix_0 = np.random.rand(states)
        transition_matrix_0 = np.random.rand(states, states)
        observation_matrix_0 = np.random.rand(states, count)
        
        # if model_type == 1:  # Modèle "left-right"
        #     transition_matrix_0 = np.zeros((states, states))
        #     for i in range(states - 1):
        #         transition_matrix_0[i, i]   = 0.6
        #         transition_matrix_0[i, i+1] = 0.4
        #     transition_matrix_0[states - 1, states - 1] = 1.0
        #     prior_matrix_0[0] = 1  # Premier état toujours choisi

        if model_type == 1:  # Modèle "left-right"
            for i in range(states):
                transition_matrix_0[i, :i] = 0  # Zéro avant l'état actuel
            # prior_matrix_0 = np.zeros((states,1))
            prior_matrix_0[0] = 1  # Premier état toujours choisi

        # Normalisation des matrices
        prior_matrix_0 = mk_stochastic(prior_matrix_0)
        transition_matrix_0 = mk_stochastic(transition_matrix_0)
        observation_matrix_0 = mk_stochastic(observation_matrix_0)

        #  Entraînement du HMM avec Baum-Welch
        LL, prior_matrix_0, transition_matrix_0, observation_matrix_0, _ = learn_dhmm(
            data, prior_matrix_0, transition_matrix_0, observation_matrix_0, it_bw, 1e-4, verbose=0
        )

        #  Vérifier si la log-vraisemblance obtenue est meilleure
        if LL[-1] > best_ll:
            best_ll = LL[-1]
            best_prior_matrix = prior_matrix_0
            best_transition_matrix = transition_matrix_0
            best_observation_matrix = observation_matrix_0

    return best_ll, best_prior_matrix, best_transition_matrix, best_observation_matrix
