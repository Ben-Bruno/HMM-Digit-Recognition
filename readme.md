# Reconnaissance de Chiffres Manuscrits par Modeles de Markov Caches (HMM)

Ce projet implemente un systeme complet de reconnaissance de formes applique aux chiffres manuscrits (0 a 9). Il repose sur l'utilisation de Modeles de Markov Caches discrets pour modeliser la structure sequentielle des traces de chiffres a partir de caracteristiques geometriques simples.

---

## Description du Projet

L'objectif est de classer des images de chiffres issues de bases de donnees (train.tif et test.tif). Le pipeline de traitement se decompose en quatre phases principales :

1. Pretraitement des images (segmentation, binarisation, redimensionnement).
2. Extraction de caracteristiques par histogrammes de projection.
3. Apprentissage de 10 modeles HMM (un par classe) via l'algorithme de Baum-Welch.
4. Reconnaissance par calcul de vraisemblance via l'algorithme Forward.

---

## Structure du Repertoire

```
projet/
|-- datasets/             # Fichiers images de base (train.tif, test.tif)
|-- hmmPy/                # Bibliotheque HMM (Forward, Viterbi, Baum-Welch)
|-- imtools/              # Utilitaires de traitement d'image et extraction
|-- NPZFile/              # Fichiers .npz (caracteristiques et modeles entraines)
|-- Main.py               # Script principal orchestrant le pipeline complet
|-- featureExtraction.py  # Extraction des primitives de la base d'entrainement
|-- trainHMM.py           # Entrainement des modeles statistiques
|-- HMM_recognize.py      # Test unitaire pour la reconnaissance d'un chiffre isole
`-- Performance.py        # Evaluation globale (precision, matrice de confusion)
```

---

## Pretraitement et Caracteristiques

Chaque chiffre est normalise dans une grille de 16x16 pixels. Les caracteristiques extraites sont des histogrammes de projection :

- Projections verticales : somme des pixels par colonne (16 valeurs).
- Projections horizontales : somme des pixels par ligne (16 valeurs).

Le vecteur d'observation resultant est une sequence de 32 symboles discrets compris entre 0 et 16.

---

## Configuration du Modele HMM

| Parametre | Valeur |
|---|---|
| Topologie | Gauche-Droite (Bakis) |
| Nombre d'etats caches | 5 |
| Alphabet d'observation | 17 symboles (0 a 16) |
| Algorithme d'apprentissage | Baum-Welch (Expectation-Maximization) |
| Algorithme de decision | Forward (log-vraisemblance) |

La topologie Gauche-Droite est choisie pour respecter l'ordre temporel de l'ecriture.

---

## Installation et Utilisation

### Prerequis

- Python 3.x
- NumPy
- Matplotlib

### Execution du projet

Extraire les caracteristiques de la base d'entrainement :

```bash
python featureExtraction.py
```

Entrainer les 10 modeles HMM :

```bash
python trainHMM.py
```

Evaluer les performances sur la base de test :

```bash
python Performance.py
```

---

## Resultats et Evaluation

Le systeme genere les metriques suivantes :

- Matrice de confusion : visualisation des correspondances entre classes reelles et predites.
- Taux de reconnaissance : pourcentage global de precision (Accuracy).
- Analyse des erreurs : identification des chiffres presentant des similitudes morphologiques (ex : 4 et 9, 1 et 7).

---

## Limites Identifiees

- Sensibilite a la rotation et a l'inclinaison des chiffres.
- Dependance a la qualite de la binarisation.
- Hypothese de stationnarite inherente aux modeles de Markov classiques.
 
 
 Structure du projet

.		# 📂 Répertoire racine du projet
├── datasets	 		# 📂 Contient les données d'entraînement et de test (images .tif)             
|   ├── test.tif			# Image utilisée pour entraîner le modèle
|   └── train.tif 			# Image utilisée pour tester le modèle       
|
├── hmmPy/			# 📂 Module principal pour l'implémentation des HMM
|   ├── __init__.py			# Permet d'utiliser "hmmPy" comme un package Python
|   ├── utils_hmm.py			# Fonctions utilitaires: normalise, em_converged et mk_stochastic. 
|   └── algorithms/		# 📂 Dossier contenant tous les algorithmes HMM
|       ├── em_algorithm.py		# Algorithme EM (Expectation-Maximization) pour HMM
|       ├── forwards_backwards.py   	# Algorithme Forward-Backward pour pour le calcul de la probabilité de vraisemblance 
|       ├── forward.py   	   	# Algorithme Forward pour pour le calculde la probabilité de vraisemblance  
|       ├── forwards.py   	    	# Version alternative de l'algorithme Forward (si besoin)
|       ├── viterbi_path.py  		# Algorithme de Viterbi pour trouver la séquence d'états cachés
|       ├── mk_dhmm_obs_lik.py		# Génération des vraisemblances pour un HMM discret
|       ├── learn_dhmm.py		# Entraînement du HMM discret avec l'algorithme EM
|       ├── log_lik_dhmm.py		# Calcul de la log-vraisemblance d'un HMM discret  
|       └── train_HMM.py 		# Script d'entraînement du HMM à partir des données
|  
├── imtools/			# 📂 Module pour le traitement d'images et pour la reconnaissance
|   ├── __init__.py			# Permet d'utiliser "imtools" comme un package Python
|   └── common/			# 📂 Dossier contenant des outils de traitement d'images
|       ├── histogrammeProjection.py	# Projection des histogrammes d'images
|       ├── image2Double.py		# Conversion d'images en format "double"
|       ├── imageBinarisation.py	# Algorithme de binarisation d'image (seuillage)
|       ├── imageCropping.py		# Recadrage des images
|       ├── imageResize.py		# Redimensionnement des images
|       ├── imageFeaturesExtraction.py	# Extraction des caractéristiques d'une image
|       ├── imageLocalization.py	# Localisation des éléments dans une image
|       ├── imageLocalizationExtraction.py # Extraction des zones localisées dans une image
|       ├── imagePreprocessing.py	# Prétraitement général des images
|       └── imageRecognition.py		# Reconnaissance d'image (utilisé avec le HMM)
| 
├── NPZFile/			# 📂 Dossier contenant les fichiers NPZ pour stocker les features, les modèles, etc.
|   ├── features.npz			# Contient les features extraites des images
|   ├── hmmModel.npz			# Contient un modèle HMM pré-entraîné
|   ├── test.npz			# Contient les données de test pour évaluer le HMM                
|   └── train.npz			# Contient les données d'entraînement pour le HMM
| 
├── FeatureExtraction.py		# Script permettant d'extraire des caractéristiques d'images
├── HMM_recognize.py			# Script qui utilise un HMM pour reconnaître du texte dans une image
├── Main.py				# Script principal
├── Performance.py			# Script permettant d'évaluer la performance du HMM
├── trainHMM.py				# Script pour entraîner un HMM sur les données disponibles
├── printedtext.png			# Image de texte imprimé pour tester la binarisation 
├── rice.png				# Image de grains de riz pour tester la binarisation
└── readme.md				# Documentation générale du projet
       



