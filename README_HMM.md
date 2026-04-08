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
