import numpy as np

from hmmPy.utils_hmm import normalise, mk_stochastic, em_converged
from hmmPy.algorithms.forwards_backwards import forwards_backwards
from hmmPy.algorithms.mk_dhmm_obs_lik import mk_dhmm_obs_lik

def learn_dhmm(data, prior, transmat, obsmat, max_iter=10, thresh=1e-4, verbose=1, 
               act=None, adj_prior=True, adj_trans=True, adj_obs=True, dirichlet=0):
    """
    Apprend les paramètres d'un HMM discret en utilisant l'algorithme EM.

    Args:
        data (list of lists or arrays): Liste des séquences d'observations.
        prior (numpy.ndarray): Distribution initiale P(Q(1)=i).
        transmat (numpy.ndarray or list): Matrice de transition P(Q(t+1)=j | Q(t)=i).
        obsmat (numpy.ndarray): Matrice d'observation P(Y(t)=o | Q(t)=i).
        max_iter (int, optional): Nombre maximal d'itérations de l'algorithme EM.
        thresh (float, optional): Seuil de convergence pour EM.
        verbose (int, optional): Affichage des logs (0 = non, 1 = oui).
        act (list, optional): Actions (pour POMDPs).
        adj_prior (bool, optional): Ajuster `prior` pendant l'EM.
        adj_trans (bool, optional): Ajuster `transmat` pendant l'EM.
        adj_obs (bool, optional): Ajuster `obsmat` pendant l'EM.
        dirichlet (float, optional): Régularisation Dirichlet pour éviter les zéros.

    Returns:
        tuple: (LL, prior, transmat, obsmat, gamma)
    """
    if not isinstance(data, list):
        data = [row for row in data]  # Convertir chaque ligne en liste si nécessaire
    if act is not None and not isinstance(act, list):
        act = [row for row in act]

    previous_loglik = -np.finfo(float).max #-np.inf
    loglik = 0
    converged = False
    num_iter = 1
    LL = []

    while num_iter <= max_iter and not converged:
        # E-step
        loglik, exp_num_trans, exp_num_visits1, exp_num_emit, gamma = compute_ess(
            prior, transmat, obsmat, data, act, dirichlet)

        if verbose:
            print(f"Iteration {num_iter}, Log-Likelihood = {loglik}")

        num_iter += 1

        # M-step
        if adj_prior:
            prior, _ = normalise(exp_num_visits1)
        if adj_trans and exp_num_trans is not None:
            if act is None:
                transmat = mk_stochastic(exp_num_trans)
            else:
                for a in range(len(transmat)):
                    transmat[a] = mk_stochastic(exp_num_trans[a])
        if adj_obs:
            obsmat = mk_stochastic(exp_num_emit)

        # Vérifier la convergence
        converged, _ = em_converged(loglik, previous_loglik, thresh)
        previous_loglik = loglik
        LL.append(loglik)

    return LL, prior, transmat, obsmat, gamma

def compute_ess(prior, transmat, obsmat, data, act=None, dirichlet=1.0):
    """
    Compute the Expected Sufficient Statistics for a discrete Hidden Markov Model.
    
    Outputs:
    - exp_num_trans[i,j] = sum_l sum_{t=2}^T P(X(t-1) = i, X(t) = j | Obs(l))
    - exp_num_visits1[i] = sum_l P(X(1) = i | Obs(l))
    - exp_num_emit[i,o] = sum_l sum_{t=1}^T P(X(t) = i, O(t) = o | Obs(l))
    where Obs(l) = O_1 .. O_T for sequence l.
    """
    
    numex = len(data)
    S, O = obsmat.shape
    
    if act is None:
        exp_num_trans = np.zeros((S, S))
        A = 0
    else:
        A = len(transmat)
        exp_num_trans = [np.zeros((S, S)) for _ in range(A)]
    
    exp_num_visits1 = np.zeros(S)
    exp_num_emit = dirichlet * np.ones((S, O))
    loglik = 0
    estimated_trans = False

    for ex in range(numex):
        obs = data[ex]
        T = len(obs)
        olikseq = mk_dhmm_obs_lik(obs, obsmat)
        
        if act is None:
            gamma, xi, current_ll = forwards_backwards(prior, transmat, olikseq)
        else:
            gamma, xi, current_ll = forwards_backwards_pomdp(prior, transmat, olikseq, act[ex])

        loglik += current_ll

        if T > 1:
            estimated_trans = True
            if act is None:
                exp_num_trans += np.sum(xi, axis=2)
            else:
                for a in range(A):
                    ndx = np.where(np.array(act[ex][1:]) == a+1)[0]  # Act[2:end] en MATLAB
                    if ndx.size > 0:
                        exp_num_trans[a] += np.sum(xi[:, :, ndx], axis=2)

        exp_num_visits1 += gamma[:, 0]

        if T < O:
            for t in range(T):
                o = obs[t]
                exp_num_emit[:, o] += gamma[:, t]
        else:
            for o in range(O):
                ndx = np.where(np.array(obs) == o)[0]
                if ndx.size > 0:
                    exp_num_emit[:, o] += np.sum(gamma[:, ndx], axis=1)

    if not estimated_trans:
        exp_num_trans = None

    return loglik, exp_num_trans, exp_num_visits1, exp_num_emit, gamma
