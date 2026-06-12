# =============================================================================
# config.py — Fichier de configuration partagé
# À importer dans TOUS les fichiers du projet :
#   from config import BETA, TEMPERATURE, N_ITERATIONS, N_CLASSES, IMAGE_SIZE
# =============================================================================
#
# CE FICHIER EST LE SEUL ENDROIT OÙ ON CHANGE LES PARAMÈTRES.
# Si le prof veut tester avec un beta différent → on change ici,
# ça se répercute automatiquement partout.
#
# =============================================================================

# Force de cohésion entre pixels voisins
# Plus beta est grand → les zones sont plus "solides" et homogènes
# Valeur faible (0.5) = peu de cohésion, résultat bruité
# Valeur forte (3.0) = très cohésif, grandes zones uniformes
BETA = 1.5

# Température de relaxation de l'algorithme de Gibbs
# Haute température → le modèle accepte plus d'erreurs (exploration)
# Basse température → le modèle est strict (exploitation)
# En pratique : garder entre 0.5 et 2.0
TEMPERATURE = 1.0

# Nombre de classes (segments/couleurs) dans l'image
# Ex: 2 = noir/blanc, 3 = fond + objet + contour
N_CLASSES = 3

# Nombre d'itérations de l'algorithme de Gibbs
# 30 est suffisant pour voir la convergence visuellement
# Augmenter pour un meilleur résultat (mais plus lent)
N_ITERATIONS = 30

# Taille à laquelle toutes les images sont redimensionnées
# Garder petit (128x128) pour que le Gibbs soit rapide
# 256x256 est possible mais plus lent
IMAGE_SIZE = (128, 128)
