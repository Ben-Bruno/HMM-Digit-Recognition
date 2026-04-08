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
       



