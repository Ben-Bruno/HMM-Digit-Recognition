import os
import numpy as np
import hmmPy as hmm

# Définir le chemin pour sauvegarder les modèles HMM
nom_resultat = 'NPZFile/hmmModel.npz'
os.makedirs(os.path.dirname(nom_resultat), exist_ok=True)

# Chargement des caractéristiques extraites de la base d'apprentissage
features_data = np.load('NPZFile/features.npz', allow_pickle=True)

    
# PARAMÈTRES DE CLASSIFICATION
# Initialisation des paramètres de HMM
typeHMM = 1  # 0 pour pour modèle ergodique, 1 pour modèle gauche-droite
Nbre_etats = 5  # Nombre d'états cachés

# TODO : Définir le nombre d'observations
# Puisque nos sommes de pixels vont de 0 à 16, nous avons 17 symboles possibles.
Nbre_Obs = 17 # Compléter avec la taille du vecteur de caractéristiques

# Nombre d'itérations
iter_sp = 4    # Nombre d'itérations pour le meilleur modèle initial
max_iter = 20  # Nombre maximal d'itérations pour l'algorithme de Baum-Welch

# === Apprentissage des modèles HMM pour chaque chiffre (0 à 9) ===
# Structure pour stocker les modèles HMM
hmmModel = {}

# Apprentissage des HMM pour chaque chiffre (0 à 9)
for Chiffre in range(10):
    obs = features_data[str(Chiffre)].astype(int) # Extraction des observations pour chaque chiffre

    # Empiler toutes les séquences d'observations pour le modèle HMM
    lengths = [len(sequence) for sequence in obs]
    X = np.concatenate(obs, axis=0)
    
    X = X.reshape(-1, 1)
    
    # Entraînement du modèle HMM avec l'algorithme de Baum-Welch
    ll, prior, transmat, obsmat = hmm.train_HMM(obs, Nbre_etats, Nbre_Obs, iter_sp, max_iter, typeHMM)
   
    # Stockage des paramètres du modèle HMM entraîner dans un dictionnaire `hmmModel`
    hmmModel[str(Chiffre)] = {
        'prior': prior,
        'transmat': transmat,
        'obsmat': obsmat
    }

    print(f"Modèle HMM entraîné pour la classe {Chiffre}")

# Sauvegarde des résultats dans un fichier .npz
np.savez(nom_resultat, hmmModel=hmmModel, typeHMM=typeHMM, iter_sp=iter_sp, max_iter=max_iter)
# np.savez_compressed(nom_resultat, hmmModel=hmmModel, typeHMM=typeHMM, iter_sp=iter_sp, max_iter=max_iter)
print(f"Les modèles HMM ont été sauvegardés avec succès dans {nom_resultat}")
