"""
Fonctions exportées :
    - get_neighbors()      : voisins d'un pixel (4-connexité)
    - potts_energy()       : énergie locale d'un pixel (modèle de Potts)
    - gibbs_probabilities(): distribution de probabilité sur les classes
    - gibbs_step()         : une passe complète de l'algorithme de Gibbs
    - run_gibbs()          : boucle N itérations + historique (pour main.py)
"""

import numpy as np
from config import BETA, TEMPERATURE, N_CLASSES

# 1
def get_neighbors(labels, i, j):
#    labels : ndarray 2D (H x W)
#    list[int] — entre 2 et 4 valeurs selon la position dans l'image
    H, W = labels.shape
    neighbors = []
    # Voisin du haut
    if i > 0:
        neighbors.append(labels[i - 1, j])

    # Voisin du bas
    if i < H - 1:
        neighbors.append(labels[i + 1, j])

    # Voisin gauche
    if j > 0:
        neighbors.append(labels[i, j - 1])

    # Voisin droite
    if j < W - 1:
        neighbors.append(labels[i, j + 1])

    return neighbors

# 2
def potts_energy(labels, i, j, candidate, beta):
    neighbors = get_neighbors(labels, i, j)
    matching = sum(1 for n in neighbors if n == candidate)
    return -beta * matching

# 3
#   P(k) = exp(-E_k / T) / Σ_k' exp(-E_k' / T)
def gibbs_probabilities(labels, i, j, beta, temperature, n_classes):
    energies = np.array([
        potts_energy(labels, i, j, k, beta)
        for k in range(n_classes)
    ])

    shifted = energies - energies.min()
    exp_vals = np.exp(-shifted / temperature)
    probs = exp_vals / exp_vals.sum()

    return probs

# 4
def gibbs_step(labels, image, beta, temperature):
    H, W = labels.shape
    n_classes = labels.max() + 1

    # Copie pour ne pas modifier le tableau pendant qu'on le lit
    # (mise à jour "synchrone" — plus stable que la mise à jour en place)
    new_labels = labels.copy()

    # Générer un ordre de parcours aléatoire des pixels
    # pixels_order : liste de (i, j) mélangée aléatoirement
    indices = [(i, j) for i in range(H) for j in range(W)]
    np.random.shuffle(indices)

    for (i, j) in indices:
        # Calculer la distribution de probabilité sur les classes
        probs = gibbs_probabilities(new_labels, i, j, beta, temperature, n_classes)

        # Tirer une nouvelle étiquette selon cette distribution
        new_labels[i, j] = np.random.choice(n_classes, p=probs)

    return new_labels

# 5
def run_gibbs(labels, image, beta=BETA, temperature=TEMPERATURE, n_iter=30,
              callback=None):

    history = [labels.copy()]
    current = labels.copy()

    for it in range(n_iter):
        current = gibbs_step(current, image, beta, temperature)
        history.append(current.copy())

        # Appel optionnel pour mise à jour live de l'UI
        if callback is not None:
            callback(current, it + 1)

    return current, history


# TEST INDEPENDANT 
if __name__ == '__main__':
    from config import BETA, TEMPERATURE, N_CLASSES, IMAGE_SIZE, N_ITERATIONS

    print("=" * 55)
    print("  TEST INDEPENDANT — gibbs_sampler.py")
    print("=" * 55)

    H, W = IMAGE_SIZE

    # Simuler une image RGB aléatoire (comme si image_loader.py l'avait chargée)
    fake_image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    # Simuler une carte d'étiquettes initiale aléatoire (comme initializer.py)
    fake_labels = np.random.randint(0, N_CLASSES, (H, W))

    print(f"Image simulée     : {H}x{W} pixels, {N_CLASSES} classes")
    print(f"Beta              : {BETA}")
    print(f"Temperature       : {TEMPERATURE}")
    print(f"Iterations        : {N_ITERATIONS}")
    print()

    # Test 1 : une seule passe
    result_one = gibbs_step(fake_labels, fake_image, BETA, TEMPERATURE)
    assert result_one.shape == (H, W), "ERREUR : mauvaise shape après gibbs_step"
    assert result_one.dtype in [np.int32, np.int64, np.int_], \
        f"ERREUR : dtype inattendu {result_one.dtype}"
    print(f"[OK] gibbs_step()  : shape={result_one.shape}, dtype={result_one.dtype}")

    # Test 2 : boucle complète N itérations
    final, history = run_gibbs(fake_labels, fake_image, n_iter=N_ITERATIONS)
    assert len(history) == N_ITERATIONS + 1, \
        f"ERREUR : history devrait avoir {N_ITERATIONS + 1} entrées"
    print(f"[OK] run_gibbs()   : {len(history)} snapshots dans l'historique")

    # Test 3 : vérifier que les étiquettes restent dans [0, N_CLASSES-1]
    assert final.min() >= 0 and final.max() < N_CLASSES, \
        f"ERREUR : étiquettes hors plage [0, {N_CLASSES-1}]"
    print(f"[OK] Plage labels  : [{final.min()}, {final.max()}] ⊆ [0, {N_CLASSES-1}]")

    # Afficher la distribution des étiquettes finales
    print()
    print("Distribution finale des étiquettes :")
    for k in range(N_CLASSES):
        count = (final == k).sum()
        pct = count / (H * W) * 100
        bar = "█" * int(pct / 2)
        print(f"  Classe {k} : {count:5d} pixels ({pct:5.1f}%)  {bar}")

    print()
    print("Tous les tests passés. gibbs_sampler.py est prêt.")
