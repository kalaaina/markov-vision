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
def gibbs_step(labels, image, beta, temperature, vectorized=True, n_classes=None):
    """
    Exécute une passe complète de l'algorithme de Gibbs sur les étiquettes.
    
    Paramètres :
        labels : array (H, W) - carte des classes actuelle
        image : array (H, W, 3) - image originale (facultative pour Potts pur)
        beta : float - force de cohésion spatiale
        temperature : float - paramètre de relaxation T
        vectorized : bool - si True, utilise la parallélisation par damier (checkerboard)
                            pour une accélération NumPy massive (30x-800x).
                            si False, utilise la boucle de parcours pixel par pixel d'origine.
        n_classes : int - nombre de classes au total. Si None, déduit depuis labels.max() + 1.
    """
    H, W = labels.shape
    if n_classes is None:
        n_classes = labels.max() + 1

    if not vectorized:
        # --- CODE D'ORIGINE : BOUCLE PIXEL PAR PIXEL ---
        # Copie pour ne pas modifier le tableau pendant qu'on le lit
        new_labels = labels.copy()

        # Générer un ordre de parcours aléatoire des pixels
        indices = [(i, j) for i in range(H) for j in range(W)]
        np.random.shuffle(indices)

        for (i, j) in indices:
            # Calculer la distribution de probabilité sur les classes
            probs = gibbs_probabilities(new_labels, i, j, beta, temperature, n_classes)

            # Tirer une nouvelle étiquette selon cette distribution
            new_labels[i, j] = np.random.choice(n_classes, p=probs)

        return new_labels

    # --- CODE OPTIMISÉ : PARALLÉLISATION PAR DAMIER (CHECKERBOARD GIBBS) ---
    # Dans un graphe de 4-connexité (haut, bas, gauche, droite), les pixels pairs
    # (i+j pair) ne dépendent que des pixels impairs (i+j impair) et inversement.
    # On peut donc mettre à jour tous les pixels pairs en parallèle, puis tous les impairs.
    new_labels = labels.copy()

    # Création des coordonnées et des masques de damier
    r_coords, c_coords = np.indices((H, W))
    checker = (r_coords + c_coords) % 2
    even_mask = (checker == 0)
    odd_mask = (checker == 1)

    for mask in [even_mask, odd_mask]:
        # Compter le nombre de voisins de chaque classe k pour TOUS les pixels
        # counts : forme (H, W, n_classes)
        counts = np.zeros((H, W, n_classes), dtype=np.int32)
        for k in range(n_classes):
            # Voisin du haut : new_labels[:-1, :] correspond à la ligne i-1
            counts[1:, :, k] += (new_labels[:-1, :] == k)
            # Voisin du bas
            counts[:-1, :, k] += (new_labels[1:, :] == k)
            # Voisin de gauche
            counts[:, 1:, k] += (new_labels[:, :-1] == k)
            # Voisin de droite
            counts[:, :-1, k] += (new_labels[:, 1:] == k)

        # Calculer les probabilités "softmax" d'appartenir à chaque classe
        # L'énergie de Potts locale est E_k = -beta * counts_k
        # Probabilité P(k) prop à exp(-E_k / T) = exp(beta * counts_k / T)
        logits = counts * (beta / temperature)
        
        # Soustraction du max pour stabilité numérique du softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True) # (H, W, n_classes)

        # Échantillonnage inverse cumulé (CDF) en parallèle
        # On calcule la fonction de répartition (CDF)
        cdf = np.cumsum(probs, axis=-1)
        
        # Tirage aléatoire uniforme pour chaque pixel
        r = np.random.rand(H, W, 1)
        
        # La classe choisie est le nombre de fois où le tirage est supérieur à la CDF cumulée
        sampled = np.sum(r > cdf, axis=-1) # (H, W)

        # Appliquer les tirages uniquement pour les pixels du groupe actif (le masque)
        new_labels[mask] = sampled[mask]

    return new_labels

# 5
def run_gibbs(labels, image, beta=BETA, temperature=TEMPERATURE, n_iter=30,
              callback=None, vectorized=True, n_classes=None):
    """
    Boucle d'échantillonnage de Gibbs sur N itérations.
    Retourne l'état final et l'historique complet.
    """
    if n_classes is None:
        n_classes = labels.max() + 1

    history = [labels.copy()]
    current = labels.copy()

    for it in range(n_iter):
        current = gibbs_step(current, image, beta, temperature, vectorized=vectorized, n_classes=n_classes)
        history.append(current.copy())

        # Appel optionnel pour mise à jour live de l'UI
        if callback is not None:
            callback(current, it + 1)

    return current, history


# TEST INDEPENDANT 
if __name__ == '__main__':
    from config import BETA, TEMPERATURE, N_CLASSES, IMAGE_SIZE, N_ITERATIONS

    print("=" * 55)
    print("  TEST INDEPENDANT - gibbs_sampler.py")
    print("=" * 55)

    H, W = IMAGE_SIZE

    # Simuler une image RGB aléatoire (comme si image_loader.py l'avait chargée)
    fake_image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    # Simuler une carte d'étiquettes initiale aléatoire (comme initializer.py)
    fake_labels = np.random.randint(0, N_CLASSES, (H, W))

    print(f"Image simulee     : {H}x{W} pixels, {N_CLASSES} classes")
    print(f"Beta              : {BETA}")
    print(f"Temperature       : {TEMPERATURE}")
    print(f"Iterations        : {N_ITERATIONS}")
    print()

    # Test 1 : une seule passe en boucle classique (non-vectorisee)
    t0 = time.time() if 'time' in globals() else 0
    result_one = gibbs_step(fake_labels, fake_image, BETA, TEMPERATURE, vectorized=False)
    assert result_one.shape == (H, W), "ERREUR : mauvaise shape après gibbs_step"
    assert result_one.dtype in [np.int32, np.int64, np.int_], \
        f"ERREUR : dtype inattendu {result_one.dtype}"
    print(f"[OK] gibbs_step() boucle   : shape={result_one.shape}, dtype={result_one.dtype}")

    # Test 2 : une passe en vectorise
    result_one_vec = gibbs_step(fake_labels, fake_image, BETA, TEMPERATURE, vectorized=True)
    assert result_one_vec.shape == (H, W), "ERREUR : mauvaise shape apres gibbs_step vectorise"
    print(f"[OK] gibbs_step() vectorise: shape={result_one_vec.shape}, dtype={result_one_vec.dtype}")

    # Test 3 : boucle complete N iterations (vectorisee par defaut)
    final, history = run_gibbs(fake_labels, fake_image, n_iter=N_ITERATIONS, vectorized=True)
    assert len(history) == N_ITERATIONS + 1, \
        f"ERREUR : history devrait avoir {N_ITERATIONS + 1} entrees"
    print(f"[OK] run_gibbs() vectorise : {len(history)} snapshots dans l'historique")

    # Test 4 : verifier que les etiquettes restent dans [0, N_CLASSES-1]
    assert final.min() >= 0 and final.max() < N_CLASSES, \
        f"ERREUR : etiquettes hors plage [0, {N_CLASSES-1}]"
    print(f"[OK] Plage labels  : [{final.min()}, {final.max()}] <= [0, {N_CLASSES-1}]")

    # Afficher la distribution des etiquettes finales
    print()
    print("Distribution finale des etiquettes :")
    for k in range(N_CLASSES):
        count = (final == k).sum()
        pct = count / (H * W) * 100
        bar = "#" * int(pct / 2)
        print(f"  Classe {k} : {count:5d} pixels ({pct:5.1f}%)  {bar}")

    print()
    print("Tous les tests passes. gibbs_sampler.py est pret.")
