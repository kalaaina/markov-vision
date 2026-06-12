import numpy as np

# On importe les constantes partagées
from config import BETA, TEMPERATURE, N_ITERATIONS, IMAGE_SIZE

# On importe le coeur de l'algorithme codé par S3
# ⚠️  Si S3 n'a pas encore fini, commenter cette ligne et utiliser le stub en bas
from gibbs_sampler import gibbs_step


# =============================================================================
# FONCTION PRINCIPALE — c'est celle que S6 appellera dans main.py
# =============================================================================

def synthesize_texture(source_patch, output_size=(128, 128), beta=BETA, T=TEMPERATURE):
    if source_patch.ndim == 3:
        # Formule standard luminosité : 0.299*R + 0.587*G + 0.114*B
        patch_gray = (0.299 * source_patch[:, :, 0] +
                      0.587 * source_patch[:, :, 1] +
                      0.114 * source_patch[:, :, 2]).astype(np.float32)
    else:
        # Le patch est déjà en niveaux de gris ou déjà des labels
        patch_gray = source_patch.astype(np.float32)

    # On discrétise le patch en N classes (comme S2 le fait pour les images)
    # On divise la plage de valeurs en tranches égales
    n_classes = _count_classes(patch_gray)

    # Convertir le patch en labels (0, 1, 2, ..., n_classes-1)
    patch_labels = _quantize_to_labels(patch_gray, n_classes)

    print(f"[Texture] Patch analysé : {patch_labels.shape}, {n_classes} classes détectées")

    # ------------------------------------------------------------------
    # ÉTAPE 2 : Créer un "canevas vierge" de la taille souhaitée
    # On remplit la grande image avec du bruit aléatoire —
    # chaque pixel reçoit une classe au hasard entre 0 et n_classes-1
    # C'est le point de départ, comme le fait S2 (initializer.py)
    # ------------------------------------------------------------------

    H, W = output_size

    # Bruit aléatoire : chaque pixel a une classe random
    # C'est volontairement "moche" — le Gibbs va corriger ça
    canvas_labels = np.random.randint(0, n_classes, size=(H, W))

    print(f"[Texture] Canevas vierge créé : {canvas_labels.shape} (bruit aléatoire)")



    reference_image = _tile_patch_to_size(patch_labels, output_size)

    reference_image_rgb = _labels_to_rgb(reference_image, n_classes)

    print(f"[Texture] Image de référence construite par carrelage du patch")


    labels = canvas_labels.copy()

    for i in range(N_ITERATIONS):
        # Appel à la fonction de S3 — c'est elle qui fait le vrai travail
        labels = gibbs_step(labels, reference_image_rgb, beta, T)

        # Affichage de progression toutes les 5 itérations
        if (i + 1) % 5 == 0:
            energie = _compute_local_energy(labels, beta)
            print(f"[Texture] Itération {i+1}/{N_ITERATIONS} — énergie : {energie:.1f}")

    print(f"[Texture] Synthèse terminée !")

    # On retourne les labels finaux — S6 les affichera dans la fenêtre
    return labels


# =============================================================================
# FONCTIONS UTILITAIRES INTERNES
# (préfixe _ = usage interne, S6 n'a pas besoin de les appeler)
# =============================================================================

def _count_classes(patch_gray):
    """
    Détermine automatiquement le nombre de classes à utiliser
    en fonction de la variance du patch.
    Plus le patch est varié, plus on utilise de classes.
    On reste entre 2 et 5 classes pour garder des calculs rapides.
    """
    variance = np.var(patch_gray)

    if variance < 500:
        return 2   # Texture simple (ex: tissu uni)
    elif variance < 2000:
        return 3   # Texture modérée (ex: brique)
    else:
        return 4   # Texture complexe (ex: herbe, feuilles)


def _quantize_to_labels(gray_image, n_classes):
    """
    Convertit une image en niveaux de gris (valeurs 0-255)
    en une carte de labels discrets (valeurs 0 à n_classes-1).

    Exemple avec n_classes=3 :
        pixels 0-85   → label 0
        pixels 86-170 → label 1
        pixels 171-255→ label 2

    C'est l'équivalent de ce que fait S2 (init_labels) mais sur le patch source.
    """
    # np.linspace crée des seuils régulièrement espacés
    bins = np.linspace(gray_image.min(), gray_image.max() + 1, n_classes + 1)

    # np.digitize assigne chaque pixel à sa tranche (label)
    labels = np.digitize(gray_image, bins[:-1]) - 1

    # S'assurer que les valeurs restent dans [0, n_classes-1]
    labels = np.clip(labels, 0, n_classes - 1)

    return labels.astype(np.int32)


def _tile_patch_to_size(patch_labels, output_size):
    """
    Répète (carrel) le patch source pour remplir output_size.
    C'est comme poser des carreaux de faïence pour couvrir un mur.

    Si le patch fait 32x32 et qu'on veut 128x128,
    on répète 4 fois en hauteur et 4 fois en largeur.
    """
    H, W = output_size
    ph, pw = patch_labels.shape

    # Calculer combien de répétitions on a besoin (on arrondit au supérieur)
    repeat_h = int(np.ceil(H / ph))
    repeat_w = int(np.ceil(W / pw))

    # np.tile répète le tableau
    tiled = np.tile(patch_labels, (repeat_h, repeat_w))

    # Couper à la bonne taille
    return tiled[:H, :W]


def _labels_to_rgb(labels, n_classes):
    H, W = labels.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Palette de couleurs pour chaque classe (jusqu'à 5 classes)
    palette = [
        [255, 100, 100],   # Classe 0 → rouge clair
        [100, 200, 100],   # Classe 1 → vert
        [100, 150, 255],   # Classe 2 → bleu
        [255, 220, 80],    # Classe 3 → jaune
        [200, 100, 200],   # Classe 4 → violet
    ]

    for class_idx in range(n_classes):
        mask = (labels == class_idx)
        rgb[mask] = palette[class_idx % len(palette)]

    return rgb


def _compute_local_energy(labels, beta):
    # Comparer chaque pixel avec son voisin de droite
    h_diff = np.sum(labels[:, :-1] != labels[:, 1:])
    # Comparer chaque pixel avec son voisin du bas
    v_diff = np.sum(labels[:-1, :] != labels[1:, :])

    return beta * (h_diff + v_diff)


# =============================================================================
# BLOC DE TEST — Lance ce fichier directement pour tester ta partie
# python texture_synth.py
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 50)
    print("TEST DE texture_synth.py")
    print("=" * 50)

    # ------------------------------------------------------------------
    # TEST 1 : Avec un patch simulé (S1 pas encore prêt ? pas grave)
    # On crée un faux patch avec des structures simples
    # ------------------------------------------------------------------
    print("\n--- Test 1 : Patch simulé (rayures horizontales) ---")

    # Créer un patch avec des rayures : classe 0 en haut, classe 1 en bas
    fake_patch = np.zeros((32, 32, 3), dtype=np.uint8)
    fake_patch[:16, :] = [200, 100, 100]   # rouge en haut
    fake_patch[16:, :] = [100, 150, 255]   # bleu en bas

    result = synthesize_texture(fake_patch, output_size=(128, 128), beta=1.5, T=1.0)

    print(f"Résultat shape : {result.shape}")          # doit être (128, 128)
    print(f"Classes présentes : {np.unique(result)}")   # ex: [0 1]

    # ------------------------------------------------------------------
    # TEST 2 : Avec un patch aléatoire (pour voir la convergence)
    # ------------------------------------------------------------------
    print("\n--- Test 2 : Patch aléatoire (3 classes) ---")

    random_patch = np.random.randint(0, 3, (32, 32)).astype(np.uint8)
    # Simuler un patch RGB depuis les labels
    random_patch_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    random_patch_rgb[random_patch == 0] = [255, 100, 100]
    random_patch_rgb[random_patch == 1] = [100, 200, 100]
    random_patch_rgb[random_patch == 2] = [100, 150, 255]

    result2 = synthesize_texture(random_patch_rgb, output_size=(128, 128), beta=2.0, T=0.8)

    # ------------------------------------------------------------------
    # VISUALISATION — afficher patch source vs texture générée
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(fake_patch)
    axes[0].set_title("Patch source\n(32x32 — rayures)")
    axes[0].axis("off")

    axes[1].imshow(result, cmap="tab10", vmin=0, vmax=4)
    axes[1].set_title("Texture générée\n(128x128 — Test 1)")
    axes[1].axis("off")

    axes[2].imshow(result2, cmap="tab10", vmin=0, vmax=4)
    axes[2].set_title("Texture générée\n(128x128 — Test 2)")
    axes[2].axis("off")

    plt.suptitle("texture_synth.py — Résultats des tests", fontweight="bold")
    plt.tight_layout()
    plt.savefig("test_texture_output.png", dpi=100, bbox_inches="tight")
    print("\nImage de test sauvegardée : test_texture_output.png")
    plt.show()
