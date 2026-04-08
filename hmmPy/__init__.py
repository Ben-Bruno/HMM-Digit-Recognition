# hmm/__init__.py

# Importation des utilitaires
from .utils_hmm                     import normalise, mk_stochastic, em_converged

# Importation des algorithmes depuis le dossier algorithms/
from .algorithms.forward            import forward
from .algorithms.forwards           import forwards
from .algorithms.forwards_backwards import forwards_backwards
from .algorithms.learn_dhmm         import learn_dhmm
from .algorithms.log_lik_dhmm       import log_lik_dhmm
from .algorithms.mk_dhmm_obs_lik    import mk_dhmm_obs_lik
from .algorithms.viterbi_path       import viterbi_path
from .algorithms.train_HMM          import train_HMM

# Liste des modules disponibles
__all__ = [
    "normalise", "mk_stochastic", "em_converged", "log_lik_dhmm",
    "forward", "forwards", "forwards_backwards", "learn_dhmm",
    "expectation_maximization", "mk_dhmm_obs_lik", "viterbi_path", "train_HMM"
]
